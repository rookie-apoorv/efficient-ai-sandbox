"""Group-wise affine (asymmetric) integer quantization + bit packing.

The core primitive of the whole pipeline. A 2-D weight ``W`` of shape
``[out_features, in_features]`` is split along its LAST axis (the contraction
axis of ``y = x @ W.T``) into groups of ``group_size`` consecutive values. Each
group gets its own ``scale`` (fp16) and ``zero`` point (uint8), and each weight
is stored as a ``bits``-wide unsigned integer:

    q = clamp(round(w / scale) + zero, 0, 2**bits - 1)
    w = (q - zero) * scale

Storage cost per weight is therefore::

    bits + (16 + 8) / group_size          bits

which is the formula the whole project's size budget rests on. See
``QWEN35_4B_MODEL_NOTES.md`` section 6.

Grouping along the input dimension (rather than per-tensor or per-output-row) is
what makes low bit-widths viable: it lets the scale track local dynamic range,
so a few outlier columns cannot blow up the step size for the entire tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Bit-widths that pack an exact whole number of values into a byte. These get a
# fast vectorised path; everything else goes through the generic bitstream path.
_ALIGNED_BITS = (1, 2, 4, 8)

SUPPORTED_BITS = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass
class QuantizedTensor:
    """A quantized weight plus everything needed to invert the transform."""

    packed: torch.Tensor  # uint8, 1-D bitstream
    scale: torch.Tensor  # float16, [out_features, n_groups]
    zero: torch.Tensor  # uint8,   [out_features, n_groups]
    shape: tuple[int, ...]  # original tensor shape
    bits: int
    group_size: int
    orig_dtype: str

    def nbytes(self) -> int:
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero.numel() * self.zero.element_size()
        )


# --------------------------------------------------------------------------- #
# Bit packing
# --------------------------------------------------------------------------- #


def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a uint8 tensor of ``bits``-wide codes into a dense uint8 bitstream.

    Values are packed little-endian within each byte for the aligned widths, and
    MSB-first as a flat bitstream for the unaligned ones. ``unpack_bits`` is the
    exact inverse; the two are tested together in ``tests/test_roundtrip.py``.
    """
    if bits == 8:
        return values.reshape(-1).contiguous()

    flat = values.reshape(-1)
    if bits in _ALIGNED_BITS:
        per_byte = 8 // bits
        pad = (-flat.numel()) % per_byte
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        chunks = flat.reshape(-1, per_byte).to(torch.uint8)
        out = torch.zeros(chunks.shape[0], dtype=torch.uint8)
        for i in range(per_byte):
            out |= chunks[:, i] << (i * bits)
        return out

    # Generic path (bits in {3, 5, 6, 7}): expand to a bit array, then repack.
    # numpy's (un)packbits is MSB-first, which we mirror on the way back out.
    import numpy as np

    arr = flat.numpy().astype(np.uint8)
    bit_view = np.unpackbits(arr[:, None], axis=1)[:, 8 - bits :]  # [-1, bits]
    stream = bit_view.reshape(-1)
    pad = (-stream.size) % 8
    if pad:
        stream = np.concatenate([stream, np.zeros(pad, dtype=np.uint8)])
    return torch.from_numpy(np.packbits(stream))


def unpack_bits(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Inverse of :func:`pack_bits`. Returns a uint8 tensor of ``numel`` codes."""
    if bits == 8:
        return packed[:numel].clone()

    if bits in _ALIGNED_BITS:
        per_byte = 8 // bits
        mask = (1 << bits) - 1
        cols = [(packed >> (i * bits)) & mask for i in range(per_byte)]
        out = torch.stack(cols, dim=1).reshape(-1)
        return out[:numel].contiguous()

    import numpy as np

    stream = np.unpackbits(packed.numpy())
    need = numel * bits
    stream = stream[:need].reshape(numel, bits)
    # Left-pad each code back out to a full byte before repacking.
    padded = np.concatenate(
        [np.zeros((numel, 8 - bits), dtype=np.uint8), stream], axis=1
    )
    return torch.from_numpy(np.packbits(padded, axis=1).reshape(-1))


# --------------------------------------------------------------------------- #
# Quantize / dequantize
# --------------------------------------------------------------------------- #


def choose_group_size(in_features: int, requested: int) -> int:
    """Largest usable group size <= ``requested`` that divides ``in_features``.

    Falls back to per-row (``group_size == in_features``) when nothing divides,
    so the pipeline never crashes on an awkward shape.
    """
    if requested <= 0 or requested >= in_features:
        return in_features
    if in_features % requested == 0:
        return requested
    for g in range(requested, 0, -1):
        if in_features % g == 0:
            return g
    return in_features


def quantize_tensor(
    weight: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
) -> QuantizedTensor:
    """Group-wise affine quantization of a >=2-D weight along its last axis."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    if weight.ndim < 2:
        raise ValueError(f"expected a >=2-D tensor, got shape {tuple(weight.shape)}")

    orig_dtype = str(weight.dtype).replace("torch.", "")
    shape = tuple(weight.shape)
    in_features = shape[-1]
    g = choose_group_size(in_features, group_size)

    # Work in fp32: bf16 has only 8 mantissa bits, and computing min/max and a
    # reciprocal in bf16 loses more precision than the quantization itself.
    w = weight.detach().to(torch.float32).reshape(-1, in_features)
    rows = w.shape[0]
    n_groups = in_features // g
    wg = w.reshape(rows, n_groups, g)

    qmax = (1 << bits) - 1
    w_min = wg.amin(dim=-1)
    w_max = wg.amax(dim=-1)
    # Include 0 in the range so that an all-positive/all-negative group still has
    # an exactly representable zero -- pruned/masked weights must stay exactly 0.
    w_min = torch.minimum(w_min, torch.zeros_like(w_min))
    w_max = torch.maximum(w_max, torch.zeros_like(w_max))

    scale = (w_max - w_min) / qmax
    # A constant group (scale == 0) is stored with scale 1 and reconstructs to
    # exactly its zero-point value; guards against div-by-zero on dead rows.
    degenerate = scale <= 0
    scale = torch.where(degenerate, torch.ones_like(scale), scale)

    zero = torch.clamp(torch.round(-w_min / scale), 0, qmax)

    q = torch.clamp(
        torch.round(wg / scale.unsqueeze(-1)) + zero.unsqueeze(-1), 0, qmax
    ).to(torch.uint8)

    # Round-trip the scale through fp16 now, so the error we measure at compress
    # time is the error the decompressed model will actually have.
    scale_fp16 = scale.to(torch.float16)

    return QuantizedTensor(
        packed=pack_bits(q, bits),
        scale=scale_fp16,
        zero=zero.to(torch.uint8),
        shape=shape,
        bits=bits,
        group_size=g,
        orig_dtype=orig_dtype,
    )


def dequantize_tensor(
    qt: QuantizedTensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct a dense tensor from a :class:`QuantizedTensor`."""
    shape = tuple(qt.shape)
    in_features = shape[-1]
    numel = 1
    for d in shape:
        numel *= d
    rows = numel // in_features
    n_groups = in_features // qt.group_size

    q = unpack_bits(qt.packed, qt.bits, numel).reshape(rows, n_groups, qt.group_size)
    scale = qt.scale.to(torch.float32).reshape(rows, n_groups, 1)
    zero = qt.zero.to(torch.float32).reshape(rows, n_groups, 1)

    w = (q.to(torch.float32) - zero) * scale
    return w.reshape(shape).to(out_dtype)


def quantization_error(weight: torch.Tensor, qt: QuantizedTensor) -> dict[str, float]:
    """Relative reconstruction error, for logging which layers got hurt."""
    ref = weight.detach().to(torch.float32)
    rec = dequantize_tensor(qt, out_dtype=torch.float32)
    diff = ref - rec
    denom = ref.norm().item()
    return {
        "rel_fro": (diff.norm().item() / denom) if denom > 0 else 0.0,
        "max_abs": diff.abs().max().item(),
    }
