"""Inverse of the v2 codec.

Self-contained by design: restoring the checkpoint must not depend on anything
in the compression package, on ``transformers``, or on a GPU.

Reconstruction is ``w = (q - zp) * scale`` per group.  When ``act_order`` was
used the columns were permuted before quantization so that the groups are
contiguous in permuted order; the stored permutation is inverted after
dequantization to recover the original column order.

Rows are processed in chunks so that expanding the per-group codebook to a
per-column view never materialises a full float32 copy of a very large tensor.
"""

from __future__ import annotations

import torch

ROW_CHUNK = 4096


def unpack_4bit(packed: torch.Tensor, cols: int) -> torch.Tensor:
    rows = packed.shape[0]
    out = torch.empty((rows, packed.shape[1] * 2), dtype=torch.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    return out[:, :cols]


def restore_tensor(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    perm: torch.Tensor | None,
    meta: dict,
) -> torch.Tensor:
    bits = int(meta["bits"])
    group_size = int(meta["group_size"])
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    shape = tuple(meta["shape"])
    dtype = getattr(torch, meta["dtype"])

    if bits == 4:
        grid = unpack_4bit(codes, cols)
    elif bits == 8:
        grid = codes
    else:
        raise ValueError(f"unsupported bit width in metadata: {bits}")

    if grid.shape != (rows, cols):
        raise ValueError(
            f"code grid {tuple(grid.shape)} does not match metadata ({rows}, {cols})"
        )

    # Column -> group id; handles a short trailing group without padding.
    gidx = torch.arange(cols) // group_size

    out = torch.empty((rows, cols), dtype=torch.float32)
    for r0 in range(0, rows, ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, rows)
        s = scales[r0:r1].float()[:, gidx]
        z = zps[r0:r1].float()[:, gidx]
        out[r0:r1] = (grid[r0:r1].float() - z) * s

    if perm is not None:
        out = out[:, torch.argsort(perm.long())]

    return out.reshape(shape).to(dtype).contiguous()