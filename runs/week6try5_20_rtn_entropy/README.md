# CS6013 — Compression 20 — week7 (codec E + Fisher allocation + rANS, RTN)

## Precision

| | dtype |
|---|---|
| Base model | `torch.bfloat16` |
| Compressed model | weights ≥ 1M elements: integer codes `q = round(w / step)` on a uniform grid (no clipping), entropy-coded with rANS; `float16` step per group of 512 columns; 0.25% of weights stored exactly as `float16`. Smaller tensors kept `bfloat16`. Vision tower and MTP head stored as shape only (restored as zeros). |
| Restored model | `torch.bfloat16`, same tensor names and shapes as the base checkpoint |

## Setup

```bash
uv sync                       # decompress.py: torch + numpy + safetensors only
uv sync --extra compress      # compress.py additionally needs transformers
```

No compiled entropy-coding library is used; the rANS coder is plain numpy.

## Compression

```bash
python compress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/base/Qwen3.5-4B \
    --output_path /path/to/compressed
```

Needs a GPU. All constants are in `compression/config.py`; the calibration
traces are bundled in `compression/calib_data/`. Side outputs go to
`<output_path>_artifacts/`. `CS6013_SMOKE=1` runs a fast structural test whose
output is not a submission.

## Decompression

```bash
python decompress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/compressed \
    --output_path /path/to/restored
```

CPU only; a few minutes (rANS decoding runs in numpy).
