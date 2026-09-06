"""Reconstruct dense tensors from a quantized checkpoint."""

from __future__ import annotations

import torch

from compression.quantize import QuantizedTensor, dequantize_tensor


def rebuild_tensor(
    meta: dict,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Invert one quantized tensor using its ``quant_config.json`` entry."""
    qt = QuantizedTensor(
        packed=packed,
        scale=scale,
        zero=zero,
        shape=tuple(meta["shape"]),
        bits=int(meta["bits"]),
        group_size=int(meta["group_size"]),
        orig_dtype=meta.get("orig_dtype", "bfloat16"),
    )
    return dequantize_tensor(qt, out_dtype=out_dtype)


# fp16 saturates at 65504. bf16 has the same exponent range as fp32, so a bf16
# source tensor can legitimately hold values fp16 cannot represent. Casting
# silently produces inf, which turns into NaN logits on the first matmul, so we
# clamp and count instead of trusting the cast.
FP16_MAX = 65504.0


def cast_dense(tensor: torch.Tensor, out_dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Cast a passthrough tensor to ``out_dtype``, clamping fp16 overflow.

    Returns ``(tensor, n_clamped)``.
    """
    if out_dtype != torch.float16 or tensor.dtype == torch.float16:
        return tensor.to(out_dtype), 0

    as_f32 = tensor.to(torch.float32)
    overflow = int((as_f32.abs() > FP16_MAX).sum().item())
    if overflow:
        as_f32 = as_f32.clamp(-FP16_MAX, FP16_MAX)
    return as_f32.to(torch.float16), overflow
