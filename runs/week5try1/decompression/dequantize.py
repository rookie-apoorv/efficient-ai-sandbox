"""Inverse of ``compression.quantize``.

This module is deliberately self-contained: restoring a checkpoint must not
depend on anything in the compression package, since the graders run
``decompress.py`` against the published Hugging Face checkpoint alone.
"""

from __future__ import annotations

import torch


def unpack_4bit(packed: torch.Tensor, cols_padded: int) -> torch.Tensor:
    """Expand two 4-bit codes per byte (low nibble first)."""
    rows = packed.shape[0]
    out = torch.empty((rows, packed.shape[1] * 2), dtype=torch.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    return out[:, :cols_padded]


def restore_tensor(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """Rebuild one original tensor from its quantized payload.

    ``meta`` is the per-tensor entry of ``compression_config.json``.
    """
    bits = int(meta["bits"])
    group_size = int(meta["group_size"])
    rows = int(meta["rows"])
    cols_padded = int(meta["cols_padded"])
    orig_cols = int(meta["orig_cols"])
    shape = tuple(meta["shape"])
    dtype = getattr(torch, meta["dtype"])

    if bits == 4:
        grid = unpack_4bit(codes, cols_padded)
    elif bits == 8:
        grid = codes
    else:
        raise ValueError(f"unsupported bit width in metadata: {bits}")

    n_groups = cols_padded // group_size
    out = grid.view(rows, n_groups, group_size).float()
    out = out * scales.float().view(rows, n_groups, 1)
    out = out + zeros.float().view(rows, n_groups, 1)
    out = out.view(rows, cols_padded)

    if orig_cols != cols_padded:
        out = out[:, :orig_cols]

    return out.reshape(shape).to(dtype).contiguous()
