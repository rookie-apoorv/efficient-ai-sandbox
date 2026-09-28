"""Codec E: uniform step x group RMS, no clipping, escape-coded exact outliers.

For one tensor ``W [rows, cols]`` with group size ``G`` and multiplier ``k``:

    step[r, g] = fp16(k * RMS(W[r, gG:(g+1)G]))
    q          = round(W / step)                     (unbounded integers)
    outliers   = top OUTLIER_FRAC of F * step^2      (stored exactly, fp16)

Symbols fed to rANS are ``q - q_min`` with one extra ESCAPE symbol at outlier
positions. Reconstruction: ``w = (sym + q_min) * step``, escapes replaced by the
stored fp16 values in row-major order.
"""

from __future__ import annotations

import math
import time

import torch

FP16_TINY = 2.0 ** -24


def pick_group(cols, preferred):
    for g in (preferred, 512, 256, 128, 64):
        if g <= cols and cols % g == 0:
            return g
    return cols


def group_steps(W, G, k):
    r, c = W.shape
    rms = W.reshape(r, c // G, G).pow(2).mean(-1).sqrt()
    return (rms * k).clamp_min(FP16_TINY).to(torch.float16).float()


def hb(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def entropy_of(codes):
    c = codes.reshape(-1).long()
    cnt = torch.bincount(c - c.min()).double()
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * p.log2()).sum())


# ---------- planning tables --------------------------------------------------
@torch.no_grad()
def build_tables(names, params, fisher, groups, kgrid, device, outlier_frac, max_rows=2048,
                 log=print):
    """Per tensor and k: estimated bits and Fisher-weighted loss increase."""
    tabs = {}
    t0 = time.time()
    for name in names:
        W = params[name]
        r, c = W.shape
        n = W.numel()
        G = groups[name]
        rows = torch.linspace(0, r - 1, min(r, max_rows), device=W.device).round().long().unique()
        Ws = W[rows].to(device).float()
        Fs = fisher[name][rows].to(device).float()
        scale = r / Ws.shape[0]
        bits = torch.zeros(len(kgrid), dtype=torch.float64)
        loss = torch.zeros(len(kgrid), dtype=torch.float64)
        for ki, k in enumerate(kgrid):
            st = group_steps(Ws, G, k).repeat_interleave(G, 1)
            q = torch.round(Ws / st)
            d2 = (q * st - Ws).pow(2)
            fl = Fs * d2
            if outlier_frac > 0:
                thr = torch.quantile(_subsample(Fs * st.pow(2)), 1 - outlier_frac)
                fl = torch.where(Fs * st.pow(2) > thr, torch.zeros_like(fl), fl)
            bits[ki] = entropy_of(q) * n + 16.0 * n / G + n * (outlier_frac * 16 + hb(outlier_frac))
            loss[ki] = 0.5 * fl.sum().item() * scale
        tabs[name] = {"bits": bits, "loss": loss, "numel": n}
    log(f"  planning tables: {len(names)} tensors x {len(kgrid)} k in {time.time() - t0:.0f}s")
    return tabs


def _subsample(x, m=4_000_000):
    x = x.reshape(-1)
    if x.numel() <= m:
        return x
    g = torch.Generator(device=x.device).manual_seed(0)
    return x[torch.randint(0, x.numel(), (m,), device=x.device, generator=g)]


def allocate(tabs, bits_budget):
    """Per tensor k index minimising sum(loss) s.t. sum(bits) <= budget (Lagrangian)."""
    names = list(tabs)
    B = torch.stack([tabs[n]["bits"] for n in names])
    D = torch.stack([tabs[n]["loss"] for n in names])

    def solve(lam):
        idx = (D + lam * B).argmin(1)
        return idx, B.gather(1, idx[:, None]).sum().item()

    lo, hi = 0.0, 1e-12
    while solve(hi)[1] > bits_budget and hi < 1e30:
        hi *= 4.0
    if solve(hi)[1] > bits_budget:
        raise RuntimeError("budget cannot be met even at the coarsest step in the grid")
    for _ in range(100):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
        if solve(mid)[1] > bits_budget:
            lo = mid
        else:
            hi = mid
    idx, tot = solve(hi)
    return {n: int(idx[i]) for i, n in enumerate(names)}, tot


# ---------- encoding one tensor -----------------------------------------------------
@torch.no_grad()
def quantize_tensor(W, F, G, k, outlier_frac, device, on_device=False):
    """Returns (symbols int64 numpy, n_symbols, q_min, steps fp16, outliers fp16, stats)."""
    W = W.to(device).float()
    r, c = W.shape
    st = group_steps(W, G, k)
    ste = st.repeat_interleave(G, 1)
    q = torch.round(W / ste)
    mask = None
    if outlier_frac > 0 and F is not None:
        sal = F.to(device).float() * ste.pow(2)
        thr = torch.quantile(_subsample(sal), 1 - outlier_frac)
        mask = sal > thr
        del sal
    rec = q * ste
    if mask is not None and mask.any():
        rec = torch.where(mask, W.to(torch.float16).float(), rec)
        base = q[~mask]
    else:
        base = q.reshape(-1)
    qmin = int(base.min().item())
    qmax = int(base.max().item())
    esc = qmax - qmin + 1
    sym = (q - qmin).long()
    outl = torch.empty(0, dtype=torch.float16)
    if mask is not None and mask.any():
        sym = torch.where(mask, torch.full_like(sym, esc), sym)
        outl = W[mask].to(torch.float16).cpu()
    d2 = (rec - W).pow(2)
    stats = {"sse": d2.sum().item(), "wss": W.pow(2).sum().item(),
             "fisher_loss": 0.5 * (d2 * F.to(device).float()).sum().item() if F is not None else 0.0,
             "outliers": int(outl.numel())}
    del rec, d2, ste, q
    if on_device:
        return sym.reshape(-1), esc + 1, qmin, st, outl, stats
    return sym.reshape(-1).cpu().numpy(), esc + 1, qmin, st.to(torch.float16).cpu(), outl, stats


@torch.no_grad()
def exact_bits(W, F, G, k, outlier_frac, device):
    """Ideal entropy-coded size (bits) of one tensor under the plan, computed on device."""
    sym, nsym, _, _, outl, _ = quantize_tensor(W, F, G, k, outlier_frac, device, on_device=True)
    cnt = torch.bincount(sym, minlength=nsym).double()
    p = cnt[cnt > 0] / cnt.sum()
    H = float(-(p * p.log2()).sum())
    n = sym.numel()
    return H * n + 16.0 * n / G + 16.0 * outl.numel()
