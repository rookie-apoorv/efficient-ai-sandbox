"""Decompression pipeline for the Efficient AI (CS6013) project."""

from decompression.dequantize import restore_tensor

__all__ = ["restore_tensor"]

SUPPORTED_FORMATS = ("cs6013-codecE-rans-v1",)
