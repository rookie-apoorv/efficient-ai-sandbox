"""Decompression entry point.

    python decompress.py \
        --model_name <model_name> \
        --checkpoint_path <path to the compressed checkpoint dir> \
        --output_path <path for the restored fp16/bf16 HF checkpoint dir>

Reads the compressed repo produced by ``compress.py`` and writes a standard
Hugging Face checkpoint (``model.safetensors`` plus the config and tokenizer
files).  No model class is instantiated, nothing is downloaded, and no GPU is
used -- only ``torch`` and ``safetensors`` are needed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from decompression import SUPPORTED_FORMATS
from decompression.dequantize import restore_tensor
from decompression.io_utils import (
    CompressedStore,
    ShardedSafetensorsWriter,
    copy_auxiliary_files,
    load_compression_config,
)


def convert_to_hf_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
    max_shard_bytes: int = 4_000_000_000,
) -> None:
    src = Path(checkpoint_path).expanduser()
    out = Path(output_path).expanduser()

    config = load_compression_config(src)
    fmt = config.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unknown compression format '{fmt}' in {src}")

    print(f"[decompress] compressed checkpoint : {src}")
    print(f"[decompress] output                : {out}")
    print(f"[decompress] format {fmt}, group size {config['group_size']}")

    store = CompressedStore(src)
    writer = ShardedSafetensorsWriter(out, "model", max_shard_bytes)

    n_done = 0
    t0 = time.time()
    for name, meta in config["tensors"].items():
        if meta["mode"] == "raw":
            writer.add(name, store.get(name))
        else:
            tensor = restore_tensor(
                store.get(f"{name}::codes"),
                store.get(f"{name}::scales"),
                store.get(f"{name}::zeros"),
                meta,
            )
            writer.add(name, tensor)

        n_done += 1
        if n_done % 200 == 0:
            print(f"  restored {n_done} tensors ({time.time() - t0:.0f}s)", flush=True)

    writer.finalize()
    store.close()

    copied = copy_auxiliary_files(src, out)
    print(f"[decompress] copied config/tokenizer files: {', '.join(copied)}")
    print(
        f"[decompress] done. restored {n_done} tensors, "
        f"{writer.total_bytes / 2**30:.3f} GiB written to {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a compressed checkpoint into a full Hugging Face model."
    )
    parser.add_argument("--model_name", "--model-name", dest="model_name", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", type=str, required=True
    )
    parser.add_argument(
        "--output_path", "--output-path", dest="output_path", type=str, required=True
    )
    parser.add_argument(
        "--max-shard-bytes", dest="max_shard_bytes", type=int, default=4_000_000_000
    )

    args = parser.parse_args()
    convert_to_hf_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        max_shard_bytes=args.max_shard_bytes,
    )


if __name__ == "__main__":
    main()
