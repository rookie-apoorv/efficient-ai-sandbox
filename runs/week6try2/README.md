# CS6013 — Compression 40 — Submission 02 (GPTQ)

## Precision

| | dtype |
|---|---|
| Base model | `torch.bfloat16` / `torch.float16` (as published) |
| Compressed model | group-wise `int4` / `int8`, asymmetric with **integer** zero-point, group size 64; `float16` scales, `uint8` zero-points; small tensors kept at base dtype |
| Restored model | identical dtype to the base checkpoint |

Reconstruction is `w = (q - zp) * scale`.

## Setup

Python ≥ 3.10, CUDA 12.6-compatible wheel line.

```bash
pip install -e .
```

`decompress.py` needs only `torch` + `safetensors` and runs on CPU.
`compress.py` additionally needs `transformers` and, in practice, a GPU.

## One-time: build the calibration corpus

Only needed if `compression/calib_data/math_calib.jsonl` is missing from the
clone. Requires network access; `compress.py` itself never uses the network.

```bash
pip install datasets
python -m compression.build_calibration_set
```

## Compression

```bash
python compress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/base/Qwen3.5-4B \
    --output_path /path/to/compressed
```

These three arguments are the entire command line. All tuning constants live in
`compression/config.py` — edit that file and re-run; do not add flags here.

The run prints the bit-width plan, the projected size ratio, and the GPTQ
coverage report before writing anything. Set `METHOD = "rtn"` in
`compression/config.py` for a fast, calibration-free, CPU-only check of the size
budget and file format before committing to a full GPTQ run.

## Decompression

```bash
python decompress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/compressed \
    --output_path /path/to/restored
```

Output is a standard HF checkpoint loadable by `AutoModelForCausalLM.from_pretrained` or vLLM.
`--model_name` is recorded for provenance only; the base weights are never consulted.

## Resource notes

- Peak host RAM during GPTQ ≈ model (~8 GiB bf16) + accumulated payloads (~2.5 GiB) + cached hidden states.
- GPU memory is bounded by `HESSIAN_BUDGET_GB` plus one decoder layer.
- Runtime is dominated by per-layer forward replays; expect hours for a full MoE model on one GPU.
