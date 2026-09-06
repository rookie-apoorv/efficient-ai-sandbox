#!/usr/bin/env python3
"""Model compression entry point.

Required by the project spec (Efficient AI CS6013), and the ONLY entry point for
the compression pipeline:

    python compress.py \
        --model_name Qwen/Qwen3.5-4B \
        --checkpoint_path /path/to/base/checkpoint \
        --output_path /path/to/compressed

No argument is hard-coded; all logic lives under ``compression/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compression.pipeline import compress_checkpoint
from compression.policy import PROFILES, QuantPolicy, get_profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", required=True, help="Base model id, e.g. Qwen/Qwen3.5-4B")
    p.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to the base checkpoint. Defaults to --model_name (downloads from the Hub).",
    )
    p.add_argument("--output_path", required=True, help="Where to write the compressed checkpoint")

    p.add_argument(
        "--profile",
        default="int8",
        choices=sorted(PROFILES),
        help="Named recipe from compression/policy.py (default: int8)",
    )
    p.add_argument("--bits", type=int, default=None, help="Override the profile's default bit-width")
    p.add_argument("--group-size", type=int, default=None, help="Override the profile's group size")
    p.add_argument(
        "--min-numel",
        type=int,
        default=None,
        help="Tensors smaller than this stay dense (default: 65536)",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Keep embed_tokens in full precision (ablation knob)",
    )
    p.add_argument(
        "--max-shard-bytes",
        type=int,
        default=None,
        help="Output shard size in bytes (default 2 GiB; lower it on tight RAM)",
    )
    p.add_argument(
        "--no-error-report",
        action="store_true",
        help="Skip per-tensor reconstruction-error measurement (faster)",
    )
    p.add_argument("--report-path", default=None, help="Where to write compression_report.json")
    return p.parse_args()


def build_policy(args: argparse.Namespace) -> QuantPolicy:
    base = get_profile(args.profile)
    policy = QuantPolicy(
        name=base.name,
        bits=args.bits if args.bits is not None else base.bits,
        group_size=args.group_size if args.group_size is not None else base.group_size,
        min_numel=args.min_numel if args.min_numel is not None else base.min_numel,
        skip_patterns=base.skip_patterns,
        overrides=dict(base.overrides),
    )
    if args.bits is not None or args.group_size is not None:
        policy.name = f"{base.name}[b{policy.bits},g{policy.group_size}]"
    if args.skip_embeddings:
        policy.overrides = {"*embed_tokens.weight": {"skip": True}, **policy.overrides}
        policy.name += "+denseemb"
    return policy


def main() -> None:
    args = parse_args()
    compress_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path or args.model_name,
        output_path=args.output_path,
        policy=build_policy(args),
        max_shard_bytes=args.max_shard_bytes,
        measure_error=not args.no_error_report,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
