# CS6013 — Compression 40 — week7 (GPTQ, energy-weighted plan)

## Precision

| | dtype |
|---|---|
| Base model | `torch.bfloat16` (as published) |
| Compressed model | group-wise `int4` / `int8`, asymmetric with integer zero-point, group size 64; `float16` scales, `uint8` zero-points, `int32` act-order permutations; small tensors kept at base dtype; vision tower and MTP head stored as shape-only (restored as zeros) |
| Restored model | identical dtype and tensor set to the base checkpoint |

Reconstruction is `w = (q - zp) * scale`, columns un-permuted when act-order was used.

## Setup

```bash
uv sync                       # decompress.py: torch + safetensors only
uv sync --extra compress      # compress.py additionally needs transformers
```

No flash-attn, flash-linear-attention or causal-conv1d is required.

## Compression

```bash
python compress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/base/Qwen3.5-4B \
    --output_path /path/to/compressed
```

All tuning constants are in `compression/config.py`. Calibration data
(`compression/calib_data/traces_all.jsonl`) is bundled and hash-checked; the
run needs no network. A GPU is required in practice. Side outputs (plan,
energies, size report) are written to `<output_path>_artifacts/`, outside the
checkpoint.

`CS6013_SMOKE=1 python compress.py ...` runs a fast structural test on a few
short samples; its output is not a valid submission.

## Decompression

```bash
python decompress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/compressed \
    --output_path /path/to/restored
```

Runs on CPU; output is a standard Hugging Face checkpoint.
