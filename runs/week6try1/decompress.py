"""Decompression entry point.

    python decompress.py \
        --model_name <model_name> \
        --checkpoint_path <path to the compressed checkpoint dir> \
        --output_path <path for the restored HF checkpoint dir>

Reads the compressed repo and writes a standard Hugging Face checkpoint. Only
``torch`` and ``safetensors`` are required: no model class is instantiated,
nothing is downloaded, and no GPU is used. ``--model_name`` is recorded for
provenance only; the base weights are never consulted.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from decompression import SUPPORTED_FORMATS
from decompression.dequantize import restore_tensor
from decompression.io_utils import (
    CompressedStore,
    ShardedSafetensorsWriter,
    copy_auxiliary_files,
    load_compression_config,
)


MAX_SHARD_BYTES = 4_000_000_000


def convert_to_hf_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
) -> None:
    src = Path(checkpoint_path).expanduser()
    out = Path(output_path).expanduser()

    config = load_compression_config(src)
    fmt = config.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"unknown compression format '{fmt}' in {src}; "
            f"this decompressor supports {SUPPORTED_FORMATS}"
        )

    print(f"[decompress] compressed checkpoint : {src}")
    print(f"[decompress] output                : {out}")
    print(
        f"[decompress] format {fmt}, method {config.get('method', '?')}, "
        f"group {config['group_size']}, act_order {config.get('act_order')}"
    )

    store = CompressedStore(src)
    writer = ShardedSafetensorsWriter(out, "model", MAX_SHARD_BYTES)

    n_done = 0
    t0 = time.time()
    n_zeroed = 0
    for name, meta in config["tensors"].items():
        if meta["mode"] == "raw":
            writer.add(name, store.get(name))
        elif meta["mode"] == "zeros":
            # Stored as shape + dtype only. The key must still exist or strict
            # loading of the restored checkpoint fails.
            writer.add(
                name,
                torch.zeros(tuple(meta["shape"]), dtype=getattr(torch, meta["dtype"])),
            )
            n_zeroed += 1
        else:
            perm_key = f"{name}::perm"
            writer.add(
                name,
                restore_tensor(
                    store.get(f"{name}::codes"),
                    store.get(f"{name}::scales"),
                    store.get(f"{name}::zps"),
                    store.get(perm_key) if perm_key in store else None,
                    meta,
                ),
            )
        n_done += 1
        if n_done % 200 == 0:
            print(f"  restored {n_done} tensors ({time.time() - t0:.0f}s)", flush=True)

    writer.finalize()
    store.close()

    copied = copy_auxiliary_files(src, out)
    print(f"[decompress] copied config/tokenizer files: {', '.join(copied)}")
    if n_zeroed:
        print(f"[decompress] {n_zeroed} tensors reconstructed as zeros (pruned subsystem)")
    print(
        f"[decompress] done. restored {n_done} tensors, "
        f"{writer.total_bytes / 2**30:.3f} GiB written to {out}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Restore a compressed checkpoint into a full Hugging Face model."
    )
    p.add_argument("--model_name", "--model-name", dest="model_name", type=str, required=True)
    p.add_argument("--checkpoint_path", "--checkpoint-path", dest="checkpoint_path",
                   type=str, required=True)
    p.add_argument("--output_path", "--output-path", dest="output_path", type=str, required=True)

    args = p.parse_args()
    convert_to_hf_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()