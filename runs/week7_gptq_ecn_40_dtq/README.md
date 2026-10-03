# CS6013 — Compression 40 — week8 (codec E + decision-token-weighted Fisher allocation + GPTQ + rANS)

## Precision

| | dtype |
|---|---|
| Base model | `torch.bfloat16` |
| Compressed model | weights ≥ 1M elements: integer codes on a uniform grid (step = k · group RMS, `float16` step per 512 columns, no clipping), chosen by GPTQ and entropy-coded with rANS; 0.25% of weights stored exactly as `float16`. Smaller tensors kept `bfloat16`. Vision tower and MTP head stored as shape only (restored as zeros). |
| Restored model | `torch.bfloat16`, same tensor names and shapes as the base checkpoint |

## Setup

```bash
uv sync                       # decompress.py: torch + numpy + safetensors only
uv sync --extra compress      # compress.py additionally needs transformers
```

## Compression

```bash
python compress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/base/Qwen3.5-4B \
    --output_path /path/to/compressed
```

Needs a GPU. Constants: `compression/config.py`. Calibration data:
`compression/calib_data/aime_calib.npz` (built once with
`python -m compression.build_calib --traces_repo <repo> --tokenizer <base dir>`).
Side outputs go to `<output_path>_artifacts/`. `CS6013_SMOKE=1` runs a fast
structural test whose output is not a submission.

## Decompression

```bash
python decompress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/compressed \
    --output_path /path/to/restored
```

CPU only; a few minutes.
