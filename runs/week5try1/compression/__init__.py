"""Compression pipeline for the Efficient AI (CS6013) project."""

from compression.quantize import (
    QuantizedTensor,
    estimate_error,
    is_quantizable,
    quantize_tensor,
    quantized_nbytes,
)
from compression.planner import format_plan_report, plan_bit_widths

__all__ = [
    "QuantizedTensor",
    "estimate_error",
    "is_quantizable",
    "quantize_tensor",
    "quantized_nbytes",
    "plan_bit_widths",
    "format_plan_report",
]

FORMAT_VERSION = "groupwise-asym-intN-v1"
