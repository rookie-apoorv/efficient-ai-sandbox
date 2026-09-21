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


CODES_PER_BYTE = {2: 4, 4: 2, 8: 1}
QMAX = {2: 3, 4: 15, 8: 255}


def unpack_codes(packed: torch.Tensor, cols: int, bits: int) -> torch.Tensor:
    """Expand sub-byte codes, least-significant field first."""
    if bits == 8:
        return packed
    per = CODES_PER_BYTE[bits]
    mask = QMAX[bits]
    rows = packed.shape[0]
    out = torch.empty((rows, packed.shape[1] * per), dtype=torch.uint8)
    for k in range(per):
        out[:, k::per] = (packed >> (bits * k)) & mask
    return out[:, :cols]


def unpack_4bit(packed: torch.Tensor, cols: int) -> torch.Tensor:
    return unpack_codes(packed, cols, 4)


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

    if bits not in CODES_PER_BYTE:
        raise ValueError(f"unsupported bit width in metadata: {bits}")
    grid = unpack_codes(codes, cols, bits)

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
