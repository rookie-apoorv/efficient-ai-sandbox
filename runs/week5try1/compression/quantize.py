"""Group-wise asymmetric integer quantization primitives.

Scheme (``groupwise-asym-intN-v1``)
----------------------------------
Every quantizable tensor is viewed as a 2-D matrix ``[rows, cols]`` by folding
all leading dimensions into ``rows`` (so a fused MoE expert tensor of shape
``[E, out, in]`` becomes ``[E * out, in]``).  ``cols`` is the *input* dimension
of the corresponding linear layer, which is the axis along which activations
are contracted -- quantizing along it keeps the error of each output unit
independent of the others.

``cols`` is zero-padded to a multiple of ``group_size`` and split into
contiguous groups.  For each group we store an affine (asymmetric) codebook::

    scale = (max - min) / (2**bits - 1)
    zero  = min
    q     = clamp(round((w - zero) / scale), 0, 2**bits - 1)
    w_hat = q * scale + zero

``scale`` and ``zero`` are *rounded to float16 before* the integer codes are
computed, so the encoder and the decoder see bit-identical constants and no
systematic bias is introduced at restore time.

4-bit codes are packed two per byte (low nibble first).

Storage cost per weight:  ``bits + 32 / group_size`` bits.
    4-bit, group 64  -> 4.50 bits/weight (28.1% of float16)
    8-bit, group 64  -> 8.50 bits/weight (53.1% of float16)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

QMAX: Dict[int, int] = {4: 15, 8: 255}
SUPPORTED_BITS = (4, 8)

# float16 cannot represent |x| > 65504; tensors exceeding this are left in their
# original dtype rather than risking an inf in the stored scale/zero-point.
FP16_SAFE_MAX = 6.0e4


def as_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Fold every leading dimension into rows, keeping the last dim as columns."""
    return tensor.reshape(-1, tensor.shape[-1])


def pad_columns(mat: torch.Tensor, group_size: int) -> torch.Tensor:
    """Zero-pad the column axis up to a whole multiple of ``group_size``."""
    remainder = mat.shape[1] % group_size
    if remainder == 0:
        return mat
    return torch.nn.functional.pad(mat, (0, group_size - remainder))


def quantize_2d(mat: torch.Tensor, bits: int, group_size: int):
    """Quantize a padded float32 matrix.

    Returns ``(codes_uint8, scales_fp16, zeros_fp16)`` where ``codes`` are the
    *unpacked* integer codes of shape ``[rows, cols]``.
    """
    rows, cols = mat.shape
    n_groups = cols // group_size
    grouped = mat.view(rows, n_groups, group_size)

    w_min = grouped.amin(dim=-1, keepdim=True)
    w_max = grouped.amax(dim=-1, keepdim=True)
    qmax = QMAX[bits]

    # Round the codebook constants to their stored precision *first*.
    scale = ((w_max - w_min) / qmax).to(torch.float16).float()
    zero = w_min.to(torch.float16).float()

    # A constant group has scale == 0; it is represented exactly by q == 0.
    denom = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.clamp(torch.round((grouped - zero) / denom), 0, qmax).to(torch.uint8)

    return (
        codes.view(rows, cols),
        scale.view(rows, n_groups).to(torch.float16),
        zero.view(rows, n_groups).to(torch.float16),
    )


def dequantize_2d(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Inverse of :func:`quantize_2d` (float32 output)."""
    rows, cols = codes.shape
    n_groups = cols // group_size
    out = codes.view(rows, n_groups, group_size).float()
    out = out * scales.float().view(rows, n_groups, 1) + zeros.float().view(rows, n_groups, 1)
    return out.view(rows, cols)


def pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack an even number of 4-bit codes per row into bytes (low nibble first)."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def unpack_4bit(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """Inverse of :func:`pack_4bit`; ``cols`` is the padded column count."""
    rows = packed.shape[0]
    out = torch.empty((rows, packed.shape[1] * 2), dtype=torch.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    return out[:, :cols].contiguous()


@dataclass
class QuantizedTensor:
    """Payload plus the metadata needed to invert the transform."""

    codes: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    meta: dict


def quantize_tensor(tensor: torch.Tensor, bits: int, group_size: int) -> QuantizedTensor:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bit width: {bits}")

    orig_shape: List[int] = list(tensor.shape)
    orig_dtype = str(tensor.dtype).replace("torch.", "")

    mat = as_2d(tensor).float()
    orig_cols = mat.shape[1]
    mat = pad_columns(mat, group_size)
    rows, cols_padded = mat.shape

    codes, scales, zeros = quantize_2d(mat, bits, group_size)
    payload = pack_4bit(codes) if bits == 4 else codes.contiguous()

    meta = {
        "mode": "quant",
        "bits": bits,
        "group_size": group_size,
        "shape": orig_shape,
        "dtype": orig_dtype,
        "rows": rows,
        "cols_padded": cols_padded,
        "orig_cols": orig_cols,
    }
    return QuantizedTensor(payload, scales, zeros, meta)


def quantized_nbytes(rows: int, cols_padded: int, bits: int, group_size: int) -> int:
    """Bytes on disk for one quantized tensor (codes + fp16 scales + fp16 zeros)."""
    n_groups = cols_padded // group_size
    return (rows * cols_padded * bits) // 8 + 2 * rows * n_groups * 2


def is_quantizable(tensor: torch.Tensor, group_size: int, min_numel: int) -> bool:
    """Only large floating-point matrices are worth quantizing.

    Norms, biases, router logits and other small tensors carry a negligible
    share of the parameter budget but a disproportionate share of the model's
    sensitivity, so they are stored losslessly.
    """
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return False
    if tensor.ndim < 2 or tensor.numel() < min_numel:
        return False
    if tensor.shape[-1] < group_size:
        return False
    if not torch.isfinite(tensor).all():
        return False
    if tensor.abs().max().item() > FP16_SAFE_MAX:
        return False
    return True


def estimate_error(
    tensor: torch.Tensor, bits: int, group_size: int, max_rows: int = 2048
) -> float:
    """Data-free proxy for a tensor's contribution to output error.

    Returns the total squared reconstruction error ``||W - W_hat||_F^2``.  Under
    the (crude but standard) assumption of isotropic layer inputs this is
    proportional to the expected squared error of that layer's output, which
    makes it comparable across tensors of different shapes.

    Rows are strided-subsampled for speed and the result rescaled, which is
    accurate to well under a percent for matrices with thousands of rows.
    """
    mat = pad_columns(as_2d(tensor).float(), group_size)
    rows = mat.shape[0]
    if rows > max_rows:
        idx = torch.linspace(0, rows - 1, max_rows).long()
        sample = mat[idx]
    else:
        sample = mat

    codes, scales, zeros = quantize_2d(sample, bits, group_size)
    recon = dequantize_2d(codes, scales, zeros, group_size)
    sq_err = torch.sum((sample - recon) ** 2).item()
    return sq_err * (rows / sample.shape[0])
