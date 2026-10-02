"""Codec E: rate/distortion tables, Fisher allocation, outlier masks, RTN, symbols.

step[r, g] = fp16(k * RMS(W[r, gG:(g+1)G]))  -- computed from the ORIGINAL weights
code       = integer (any size, no clipping); w_hat = code * step
outliers   = top OUTLIER_FRAC of F * RMS^2 per tensor, stored exactly (fp16).
             The ranking does not depend on k (k is constant within a tensor),
             so masks are fixed before planning and before GPTQ.
Symbols for rANS: code - q_min, with one extra ESCAPE symbol at outlier positions.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

FP16_TINY = 2.0 ** -24


def pick_group(cols, preferred):
    for g in (preferred, 512, 256, 128, 64):
        if g <= cols and cols % g == 0:
            return g
    return cols


def group_rms(W, G):
    r, c = W.shape
    return W.reshape(r, c // G, G).pow(2).mean(-1).sqrt()


def group_steps(W, G, k):
    return (group_rms(W, G) * k).clamp_min(FP16_TINY).to(torch.float16).float()


def hb(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _subsample(x, m=4_000_000):
    x = x.reshape(-1)
    if x.numel() <= m:
        return x
    g = torch.Generator(device=x.device).manual_seed(0)
    return x[torch.randint(0, x.numel(), (m,), device=x.device, generator=g)]


# ---------- outliers ---------------------------------------------------------
@torch.no_grad()
def outlier_index(W, F, G, frac, device):
    """Flat (row-major) indices of the outlier weights, int64 on CPU."""
    if frac <= 0:
        return torch.empty(0, dtype=torch.int64)
    W = W.to(device).float()
    sal = F.to(device).float() * group_rms(W, G).pow(2).repeat_interleave(G, 1)
    n_keep = max(1, int(round(frac * W.numel())))
    idx = torch.topk(sal.reshape(-1), n_keep, sorted=False).indices
    return idx.sort().values.cpu()


def mask_from_index(idx, shape, device):
    m = torch.zeros(shape[0] * shape[1], dtype=torch.bool, device=device)
    if idx.numel():
        m[idx.to(device)] = True
    return m.reshape(shape)


# ---------- symbols & sizes --------------------------------------------------------
def to_symbols(q, mask):
    """q: integer codes (any int dtype / float), mask: bool outliers or None."""
    q = q.long()
    base = q[~mask] if mask is not None and mask.any() else q.reshape(-1)
    qmin, qmax = int(base.min()), int(base.max())
    esc = qmax - qmin + 1
    sym = q - qmin
    if mask is not None and mask.any():
        sym = torch.where(mask, torch.full_like(sym, esc), sym)
    return sym.reshape(-1), esc + 1, qmin


def entropy_bits_of(sym, nsym):
    cnt = torch.bincount(sym.reshape(-1), minlength=nsym).double()
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * p.log2()).sum()) * sym.numel()


def stored_bits(sym, nsym, G, n_out):
    n = sym.numel()
    return entropy_bits_of(sym, nsym) + 16.0 * n / G + 16.0 * n_out


# ---------- RTN on the grid ----------------------------------------------------------
@torch.no_grad()
def rtn(W, G, k, mask, device):
    """Round-to-nearest on the codec-E grid. Returns (q int32, w_hat fp32) on device."""
    W = W.to(device).float()
    st = group_steps(W, G, k).repeat_interleave(G, 1)
    q = torch.round(W / st)
    w_hat = q * st
    if mask is not None and mask.any():
        w_hat = torch.where(mask, W.to(torch.float16).float(), w_hat)
        q = torch.where(mask, torch.zeros_like(q), q)
    return q.to(torch.int32), w_hat


# ---------- planning ------------------------------------------------------------------
@torch.no_grad()
def build_tables(names, params, fisher, groups, masks, kgrid, device, max_rows=2048, log=print):
    """Per tensor and k: estimated stored bits and Fisher loss 0.5*sum F*dW^2 (RTN)."""
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
        Ms = masks[name].to(device)[rows] if masks[name] is not None else None
        scale = r / Ws.shape[0]
        n_out = int(masks[name].sum()) if masks[name] is not None else 0
        bits = torch.zeros(len(kgrid), dtype=torch.float64)
        loss = torch.zeros(len(kgrid), dtype=torch.float64)
        for ki, k in enumerate(kgrid):
            q, w_hat = rtn(Ws, G, k, Ms, device)
            sym, nsym, _ = to_symbols(q, Ms)
            H = entropy_bits_of(sym, nsym) / sym.numel()
            bits[ki] = H * n + 16.0 * n / G + 16.0 * n_out
            loss[ki] = 0.5 * (Fs * (w_hat - Ws).pow(2)).sum().item() * scale
        tabs[name] = {"bits": bits, "loss": loss, "numel": n}
    log(f"  tables: {len(names)} tensors x {len(kgrid)} k in {time.time() - t0:.0f}s")
    return tabs


def allocate(tabs, bits_budget, inflate=1.0):
    """k index per tensor minimising sum(loss) s.t. inflate*sum(bits) <= budget."""
    names = list(tabs)
    B = torch.stack([tabs[n]["bits"] for n in names]) * inflate
    D = torch.stack([tabs[n]["loss"] for n in names])

    def solve(lam):
        idx = (D + lam * B).argmin(1)
        return idx, B.gather(1, idx[:, None]).sum().item()

    if solve(1e30)[1] > bits_budget:
        raise RuntimeError("budget cannot be met even at the coarsest step in the grid")
    lo, hi = 0.0, 1e-12
    while solve(hi)[1] > bits_budget:
        hi *= 4.0
    for _ in range(100):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
        if solve(mid)[1] > bits_budget:
            lo = mid
        else:
            hi = mid
    idx, tot = solve(hi)
    return {n: int(idx[i]) for i, n in enumerate(names)}, tot


def kgrid_from(kmin, kmax, steps):
    return np.geomspace(kmin, kmax, steps).tolist()
