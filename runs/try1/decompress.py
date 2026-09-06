#!/usr/bin/env python3
"""Model decompression entry point.

Required by the project spec (Efficient AI CS6013), and the ONLY entry point for
the decompression pipeline. Takes a compressed checkpoint and reconstructs a
standard fp16 HuggingFace checkpoint that loads with ``from_pretrained``:

    python decompress.py \
        --model_name Qwen/Qwen3.5-4B \
        --checkpoint_path /path/to/compressed \
        --output_path /path/to/restored_fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decompression.pipeline import decompress_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", required=True, help="Base model id, e.g. Qwen/Qwen3.5-4B")
    p.add_argument("--checkpoint_path", required=True, help="Path to the compressed checkpoint")
    p.add_argument("--output_path", required=True, help="Where to write the restored checkpoint")
    p.add_argument(
        "--out-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Reconstruction dtype. The spec asks for fp16 (default).",
    )
    p.add_argument(
        "--max-shard-bytes",
        type=int,
        default=None,
        help="Output shard size in bytes (default 2 GiB; lower it on tight RAM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decompress_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        out_dtype=args.out_dtype,
        max_shard_bytes=args.max_shard_bytes,
    )


if __name__ == "__main__":
    main()
