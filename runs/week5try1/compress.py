"""Compression entry point.

    python compress.py \
        --model_name <model_name> \
        --checkpoint_path <path to base fp16/bf16 checkpoint dir> \
        --output_path <path for the compressed checkpoint dir>

The base checkpoint is read straight off local disk with ``safetensors``; no
model class is instantiated and no GPU is required.  The output directory is a
self-contained Hugging Face repo: quantized shards, ``compression_config.json``,
and the config / tokenizer files copied from the base checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from compression import FORMAT_VERSION
from compression.io_utils import (
    ShardedSafetensorsWriter,
    copy_auxiliary_files,
    iter_tensors,
    resolve_checkpoint_dir,
)
from compression.planner import format_plan_report, plan_bit_widths
from compression.quantize import (
    estimate_error,
    is_quantizable,
    quantize_tensor,
    quantized_nbytes,
)

SHARD_PREFIX = "compressed_model"


def _dtype_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def build_plan(
    base_dir: Path,
    group_size: int,
    min_numel: int,
    target_ratio: float,
    base_bits: int,
    high_bits: int,
    verbose: bool = True,
):
    """First pass: measure every tensor and choose its bit width."""
    candidates = []
    fixed_bytes = 0
    original_bytes = 0
    n_seen = 0
    t0 = time.time()

    for name, tensor in iter_tensors(base_dir):
        n_seen += 1
        original_bytes += _dtype_bytes(tensor)

        if not is_quantizable(tensor, group_size, min_numel):
            fixed_bytes += _dtype_bytes(tensor)
            continue

        rows = tensor.numel() // tensor.shape[-1]
        cols = tensor.shape[-1]
        cols_padded = cols + (-cols % group_size)

        entry = {
            "name": name,
            "numel": tensor.numel(),
            "bytes": {
                b: quantized_nbytes(rows, cols_padded, b, group_size)
                for b in (base_bits, high_bits)
            },
            "err": {
                b: estimate_error(tensor, b, group_size) for b in (base_bits, high_bits)
            },
        }
        candidates.append(entry)

        if verbose and n_seen % 200 == 0:
            print(f"  scanned {n_seen} tensors ({time.time() - t0:.0f}s)", flush=True)
        del tensor

    bits_by_name, projected = plan_bit_widths(
        candidates,
        fixed_bytes=fixed_bytes,
        original_bytes=original_bytes,
        target_ratio=target_ratio,
        base_bits=base_bits,
        high_bits=high_bits,
    )
    return candidates, bits_by_name, fixed_bytes, original_bytes, projected


def write_compressed(
    base_dir: Path,
    out_dir: Path,
    bits_by_name: dict,
    group_size: int,
    min_numel: int,
    model_name: str,
    target_ratio: float,
    original_bytes: int,
    max_shard_bytes: int,
    verbose: bool = True,
) -> dict:
    """Second pass: quantize and write the compressed repo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(out_dir, SHARD_PREFIX, max_shard_bytes)
    tensor_meta: dict = {}
    n_done = 0
    t0 = time.time()

    for name, tensor in iter_tensors(base_dir):
        if name in bits_by_name:
            packed = quantize_tensor(tensor, bits_by_name[name], group_size)
            writer.add(f"{name}::codes", packed.codes)
            writer.add(f"{name}::scales", packed.scales)
            writer.add(f"{name}::zeros", packed.zeros)
            tensor_meta[name] = packed.meta
        else:
            writer.add(name, tensor)
            tensor_meta[name] = {
                "mode": "raw",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }

        n_done += 1
        if verbose and n_done % 200 == 0:
            print(f"  encoded {n_done} tensors ({time.time() - t0:.0f}s)", flush=True)
        del tensor

    writer.finalize()

    config = {
        "format": FORMAT_VERSION,
        "model_name": model_name,
        "group_size": group_size,
        "min_numel": min_numel,
        "target_ratio": target_ratio,
        "base_dtype": "float16/bfloat16 (as in the base checkpoint)",
        "original_total_bytes": original_bytes,
        "compressed_total_bytes": writer.total_bytes,
        "achieved_ratio": writer.total_bytes / max(original_bytes, 1),
        "shard_prefix": SHARD_PREFIX,
        "tensors": tensor_meta,
    }
    (out_dir / "compression_config.json").write_text(json.dumps(config, indent=2))
    return config


def convert_from_hf_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
    group_size: int = 64,
    min_numel: int = 1 << 20,
    target_ratio: float = 0.375,
    base_bits: int = 4,
    high_bits: int = 8,
    max_shard_bytes: int = 4_000_000_000,
    dry_run: bool = False,
) -> dict:
    base_dir = resolve_checkpoint_dir(checkpoint_path, model_name)
    out_dir = Path(output_path).expanduser()

    print(f"[compress] base checkpoint : {base_dir}")
    print(f"[compress] output          : {out_dir}")
    print(f"[compress] group size      : {group_size}, target ratio {target_ratio}")
    print("[compress] pass 1/2: measuring quantization error ...", flush=True)

    candidates, bits_by_name, fixed_bytes, original_bytes, projected = build_plan(
        base_dir, group_size, min_numel, target_ratio, base_bits, high_bits
    )
    print(
        format_plan_report(
            candidates, bits_by_name, fixed_bytes, original_bytes, projected
        )
    )

    if projected > target_ratio * original_bytes:
        print(
            "[compress] WARNING: even uniform int4 exceeds the target ratio. "
            "Lower --group-size is not enough; consider a lower base bit width.",
            file=sys.stderr,
        )

    if dry_run:
        print("[compress] --dry-run set, nothing written.")
        return {"achieved_ratio": projected / max(original_bytes, 1)}

    print("[compress] pass 2/2: encoding and writing shards ...", flush=True)
    config = write_compressed(
        base_dir,
        out_dir,
        bits_by_name,
        group_size,
        min_numel,
        model_name,
        target_ratio,
        original_bytes,
        max_shard_bytes,
    )

    copied = copy_auxiliary_files(base_dir, out_dir)
    print(f"[compress] copied config/tokenizer files: {', '.join(copied)}")
    print(
        f"[compress] done. compressed size "
        f"{config['compressed_total_bytes'] / 2**30:.3f} GiB, "
        f"ratio {config['achieved_ratio']:.4f}"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress a Hugging Face checkpoint with group-wise integer quantization."
    )
    parser.add_argument("--model_name", "--model-name", dest="model_name", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", type=str, required=True
    )
    parser.add_argument(
        "--output_path", "--output-path", dest="output_path", type=str, required=True
    )
    parser.add_argument("--group-size", dest="group_size", type=int, default=64)
    parser.add_argument("--min-numel", dest="min_numel", type=int, default=1 << 20)
    parser.add_argument(
        "--target-ratio",
        dest="target_ratio",
        type=float,
        default=0.375,
        help="Upper bound on compressed/original size. Kept a little under the "
        "0.40 track target to leave room for tokenizer and config files.",
    )
    parser.add_argument("--base-bits", dest="base_bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--high-bits", dest="high_bits", type=int, default=8, choices=[4, 8])
    parser.add_argument(
        "--max-shard-bytes", dest="max_shard_bytes", type=int, default=4_000_000_000
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")

    args = parser.parse_args()
    convert_from_hf_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        group_size=args.group_size,
        min_numel=args.min_numel,
        target_ratio=args.target_ratio,
        base_bits=args.base_bits,
        high_bits=args.high_bits,
        max_shard_bytes=args.max_shard_bytes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
