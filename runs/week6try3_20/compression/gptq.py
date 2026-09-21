"""GPTQ: error-compensated weight quantization.

Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers* (2022).

Why it beats round-to-nearest
-----------------------------
RTN minimises ``||W - W_hat||``, treating every weight as equally important.
What actually matters is the layer's *output* error ``||WX - W_hat X||``, where
``X`` is the distribution of real activations.  Expanding that objective gives
a quadratic form in the Hessian ``H = 2 X X^T``, so the columns of ``W`` that
multiply high-variance activations deserve smaller errors than the rest.

GPTQ quantizes columns left to right and, after fixing column ``j``, pushes the
rounding error it just committed into the columns that have not been quantized
yet -- along the direction prescribed by ``H^-1``.  The remaining weights
absorb the damage, so the layer output stays close even though individual
weights move further from their originals than RTN would allow.

Implementation notes
--------------------
* ``H`` is accumulated in float32 from hooked layer inputs.
* Dead columns (zero Hessian diagonal, i.e. an input feature that never fires)
  get their diagonal set to 1 and their weights zeroed, which is what the
  reference implementation does to keep the Cholesky factorisation valid.
* Damping ``H += percdamp * mean(diag(H)) * I`` keeps ``H`` positive definite.
  MoE experts see only the tokens routed to them, so their Hessians are often
  rank-deficient and the damping term is doing real work here.
* ``act_order`` quantizes high-Hessian columns first, which measurably helps at
  4 bits.  Columns are permuted, so the permutation is stored alongside the
  codes and undone at restore time.
* Work proceeds in blocks of 128 columns with a lazy batched update, which is
  what makes the O(d^3) algorithm tractable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from compression.quantize import (
    QMAX,
    QuantResult,
    _meta,
    find_qparams,
    n_groups_for,
    pack_codes,
)


class GPTQQuantizer:
    """Accumulates a Hessian for one linear weight, then quantizes it."""

    def __init__(self, weight: torch.Tensor, device: torch.device | str = "cpu"):
        # weight is [out_features, in_features]; columns are input features.
        self.rows, self.cols = weight.shape
        self.device = torch.device(device)
        self.H = torch.zeros((self.cols, self.cols), dtype=torch.float32, device=self.device)
        self.n_samples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate ``H`` from one batch of layer inputs.

        ``inp`` may be ``[..., in_features]``; leading dims are flattened into
        the token axis.  The running mean formulation keeps ``H`` on a stable
        scale regardless of how many calibration tokens arrive.
        """
        x = inp.reshape(-1, inp.shape[-1]).to(self.device, torch.float32)
        if x.shape[0] == 0:
            return
        t = x.shape[0]
        self.H *= self.n_samples / (self.n_samples + t)
        self.n_samples += t
        x = x * (2.0 / self.n_samples) ** 0.5
        self.H += x.T @ x

    def free(self) -> None:
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def quantize(
        self,
        weight: torch.Tensor,
        bits: int,
        group_size: int,
        percdamp: float = 0.01,
        act_order: bool = True,
        blocksize: int = 128,
        mse: bool = True,
    ) -> Tuple[QuantResult, torch.Tensor]:
        """Run GPTQ.

        Returns ``(packed_result, dequantized_weight)``.  The dequantized
        weight is written back into the live model so that later layers see the
        error already committed upstream -- without that feedback the per-layer
        errors compound instead of being partially cancelled.
        """
        if bits not in QMAX:
            raise ValueError(f"unsupported bit width: {bits}")
        maxq = QMAX[bits]

        # Groups must never straddle a block boundary: the codebook for a group
        # is fit on the error-compensated weights held in the block buffer, and
        # columns past the block end have not received the current block's
        # updates yet. Making blocksize a multiple of group_size guarantees
        # every group lies wholly inside one block.
        blocksize = max(blocksize, group_size)
        blocksize -= blocksize % group_size

        orig_shape = list(weight.shape)
        orig_dtype = str(weight.dtype).replace("torch.", "")

        W = weight.detach().to(self.device, torch.float32).clone()
        H = self.H
        cols = self.cols

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        perm = None
        if act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        damp = percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(cols, device=self.device)
        H[idx, idx] += damp

        # Upper Cholesky factor of H^-1: row i of Hinv gives the direction along
        # which column i's error is redistributed over columns i+1...
        try:
            L = torch.linalg.cholesky(H)
        except RuntimeError:
            H[idx, idx] += damp * 10
            L = torch.linalg.cholesky(H)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

        ng = n_groups_for(cols, group_size)
        Q = torch.zeros((self.rows, cols), dtype=torch.uint8, device=self.device)
        scales = torch.zeros((self.rows, ng), dtype=torch.float16, device=self.device)
        zps = torch.zeros((self.rows, ng), dtype=torch.uint8, device=self.device)
        W_hat = torch.zeros_like(W)

        cur_scale: Optional[torch.Tensor] = None
        cur_zp: Optional[torch.Tensor] = None

        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros((self.rows, count), dtype=torch.uint8, device=self.device)
            Deq1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]

                if col % group_size == 0:
                    g = col // group_size
                    hi = min(col + group_size, cols)  # guaranteed <= i2
                    # Fit the codebook on the error-compensated weights, not the
                    # originals: earlier columns in this block have already had
                    # their rounding error pushed into these.
                    ref = W1[:, col - i1 : hi - i1]
                    cur_scale, cur_zp = find_qparams(ref, bits, mse=mse)
                    scales[:, g] = cur_scale
                    zps[:, g] = cur_zp

                s = cur_scale.float()
                z = cur_zp.float()
                q = torch.clamp(torch.round(w / s) + z, 0, maxq)
                dq = (q - z) * s

                Q1[:, i] = q.to(torch.uint8)
                Deq1[:, i] = dq

                err = (w - dq) / d
                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            W_hat[:, i1:i2] = Deq1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        # Restore original column order for the weight handed back to the model.
        if perm is not None:
            inv = torch.argsort(perm)
            W_hat_orig = W_hat[:, inv]
        else:
            W_hat_orig = W_hat

        codes = Q.cpu()
        payload = pack_codes(codes, bits)

        meta = _meta(
            orig_shape,
            orig_dtype,
            self.rows,
            cols,
            bits,
            group_size,
            perm is not None,
        )
        meta["method"] = "gptq"

        result = QuantResult(
            payload,
            scales.cpu(),
            zps.cpu(),
            perm.to(torch.int32).cpu() if perm is not None else None,
            meta,
        )
        return result, W_hat_orig.to(weight.dtype)


def relative_output_error(
    W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor
) -> float:
    """``||WX - W_hat X|| / ||WX||`` -- the quantity GPTQ actually minimises."""
    ref = W.float() @ X.float()
    err = (W.float() - W_hat.float()) @ X.float()
    return (err.norm() / ref.norm().clamp_min(1e-12)).item()
