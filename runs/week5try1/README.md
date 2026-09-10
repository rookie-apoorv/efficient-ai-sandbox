# CS6013 — Compression 40 — Submission 01

## Precision

| | dtype |
|---|---|
| Base model | `torch.bfloat16` / `torch.float16` (as published) |
| Compressed model | group-wise `int4` / `int8` (asymmetric, group size 64) with `float16` scales and zero-points; small tensors kept at base dtype |
| Restored model | identical dtype to the base checkpoint |

## Setup

Python ≥ 3.10, CUDA 12.6-compatible wheel line. No GPU is required to run either
pipeline.

```bash
pip install -e .
# or, without installing the project:
pip install "torch==2.6.0" "safetensors>=0.4.3" numpy
```

Both entry points import the local `compression/` and `decompression/`
packages, so run them from this directory (or add it to `PYTHONPATH`).

## Compression

```bash
python compress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/base/Qwen3.5-4B \
    --output_path /path/to/compressed
```

`--checkpoint_path` must be a **local directory** containing the base
checkpoint (`*.safetensors`, `config.json`, tokenizer files). Nothing is
downloaded from the Hub.

The output directory is the artefact uploaded to Hugging Face. It contains:

```
compressed_model.safetensors        (or compressed_model-0000k-of-0000n.safetensors + index)
compression_config.json
config.json, generation_config.json, tokenizer*, preprocessor_config.json, ...
```

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--target-ratio` | `0.375` | upper bound on compressed / original size |
| `--group-size` | `64` | quantization group length along the input dim |
| `--min-numel` | `1048576` | tensors smaller than this stay at full precision |
| `--dry-run` | off | print the plan and achieved ratio without writing |

## Decompression

```bash
python decompress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/compressed \
    --output_path /path/to/restored
```

`--checkpoint_path` is the compressed directory (the Hugging Face repo).
The output is a standard Hugging Face checkpoint that loads with
`AutoModelForCausalLM.from_pretrained` or vLLM. `--model_name` is recorded for
provenance only — the base weights are never read during decompression.

## Verifying a run

```bash
python compress.py   --model_name Qwen/Qwen3.5-4B --checkpoint_path ./base --output_path ./compressed --dry-run
du -sb ./base/*.safetensors ./compressed/*.safetensors
```

`compress.py` prints the achieved ratio at the end of the run, and
`compression_config.json` records `original_total_bytes`,
`compressed_total_bytes` and `achieved_ratio`.
