"""GPTQ on the codec-E grid.

Standard GPTQ (Frantar et al. 2022) with three changes:

1. **Grid.** Each weight w[r, j] is rounded to an integer multiple of its group's
   step, ``q = round(w / step[r, group(j)])``, with no clipping. The steps are
   computed from the ORIGINAL weights before GPTQ starts and never change, so
   GPTQ only decides which integer each weight gets.
2. **Static groups with act-order.** Columns are processed in order of
   decreasing Hessian diagonal (act-order), but group membership stays in the
   original column order. The stored format therefore needs no permutation and
   the decompressor is identical to the RTN version.
3. **Outliers.** At an outlier position the current (error-compensated) value is
   stored exactly as fp16, so its rounding error is zero and nothing is
   propagated from it.

After fixing column j, its error is spread over the not-yet-quantized columns
along H^-1 (lazy block updates of BLOCK_SIZE columns), which minimises the
layer-output error ||(W - W_hat) X||^2 on the calibration activations.
"""

from __future__ import annotations

import torch


class Hessian:
    def __init__(self, cols, device):
        self.H = torch.zeros((cols, cols), dtype=torch.float32, device=device)
        self.n = 0

    def add(self, inp):
        x = inp.reshape(-1, inp.shape[-1]).to(self.H.device, torch.float32)
        t = x.shape[0]
        if t == 0:
            return
        self.H *= self.n / (self.n + t)
        self.n += t
        x = x * (2.0 / self.n) ** 0.5
        self.H += x.T @ x


@torch.no_grad()
def gptq_on_grid(W, H, steps, G, mask, percdamp=0.01, blocksize=128, act_order=True):
    """W [r,c] (any float, on device), H [c,c] fp32, steps [r, c/G] fp32, mask bool [r,c] or None.

    Returns (q int32 [r,c], w_hat fp32 [r,c]) in the original column order.
    Outlier positions have q = 0 and w_hat = fp16(current value).
    """
    dev = H.device
    W = W.to(dev, torch.float32).clone()
    r, c = W.shape
    S = steps.to(dev, torch.float32).repeat_interleave(G, 1)
    M = mask.to(dev) if mask is not None else torch.zeros((r, c), dtype=torch.bool, device=dev)
    H = H.clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H, S, M = W[:, perm], H[perm][:, perm], S[:, perm], M[:, perm]

    idx = torch.arange(c, device=dev)
    damp = percdamp * torch.mean(torch.diag(H))
    H[idx, idx] += damp
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        H[idx, idx] += 10 * damp
        L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

    Q = torch.zeros((r, c), dtype=torch.int32, device=dev)
    Wh = torch.zeros((r, c), dtype=torch.float32, device=dev)
    for i1 in range(0, c, blocksize):
        i2 = min(i1 + blocksize, c)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            w = W1[:, i]
            s = S[:, col]
            m = M[:, col]
            q = torch.round(w / s)
            dq = q * s
            if m.any():
                dq = torch.where(m, w.to(torch.float16).float(), dq)
                q = torch.where(m, torch.zeros_like(q), q)
            Q[:, col] = q.to(torch.int32)
            Wh[:, col] = dq
            err = (w - dq) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if perm is not None:
        inv = torch.argsort(perm)
        Q, Wh = Q[:, inv], Wh[:, inv]
    return Q, Wh
