"""Decompression pipeline: quantized checkpoint -> full fp16 HF checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from compression.checkpoint_io import (
    ShardWriter,
    copy_aux_files,
    dir_size_bytes,
    resolve_checkpoint,
    weight_files,
)

from .dequantize import cast_dense, rebuild_tensor

QUANT_CONFIG = "quant_config.json"

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_shard_index(ckpt_dir: Path) -> dict[str, Path]:
    """Map every stored tensor name to the shard file containing it."""
    location: dict[str, Path] = {}
    for shard in weight_files(ckpt_dir):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                location[name] = shard
    return location


def decompress_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
    *,
    out_dtype: str = "float16",
    max_shard_bytes: int | None = None,
) -> dict:
    """Rebuild a full-precision HF checkpoint from a compressed one."""
    started = time.time()
    src = resolve_checkpoint(checkpoint_path)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    torch_dtype = DTYPES[out_dtype]

    qc_path = src / QUANT_CONFIG
    if not qc_path.is_file():
        raise FileNotFoundError(
            f"{qc_path} not found -- is {src} a checkpoint produced by compress.py?"
        )
    qc = json.loads(qc_path.read_text())
    suf = qc["suffixes"]
    tensors_meta: dict[str, dict] = qc["tensors"]

    location = _load_shard_index(src)
    print(f"[decompress] source: {src}")
    print(
        f"[decompress] profile={qc.get('profile')} "
        f"quantized={len(tensors_meta)} stored={len(location)} -> {out_dtype}"
    )

    # Group stored tensor names by the shard they live in so we open each shard
    # once and read sequentially, rather than reopening per tensor.
    dense_names = [
        n
        for n in location
        if not n.endswith((suf["packed"], suf["scale"], suf["zero"]))
    ]

    written = 0
    clamped_total = 0
    writer_kwargs = {}
    if max_shard_bytes:
        writer_kwargs["max_shard_bytes"] = max_shard_bytes

    open_files: dict[Path, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = location[name]
        if shard not in open_files:
            open_files[shard] = safe_open(shard, framework="pt", device="cpu").__enter__()
        return open_files[shard].get_tensor(name)

    try:
        with ShardWriter(out, **writer_kwargs) as writer:
            total = len(tensors_meta) + len(dense_names)
            pbar = tqdm(total=total, desc="dequantizing", unit="t")

            for name, meta in tensors_meta.items():
                tensor = rebuild_tensor(
                    meta,
                    get(name + suf["packed"]),
                    get(name + suf["scale"]),
                    get(name + suf["zero"]),
                    out_dtype=torch_dtype,
                )
                writer.add(name, tensor)
                written += 1
                pbar.update(1)
                del tensor

            for name in dense_names:
                tensor, clamped = cast_dense(get(name), torch_dtype)
                clamped_total += clamped
                writer.add(name, tensor)
                written += 1
                pbar.update(1)
                del tensor

            pbar.close()
    finally:
        for f in open_files.values():
            try:
                f.__exit__(None, None, None)
            except Exception:
                pass

    # config.json / tokenizer come from the compressed checkpoint (which is
    # self-contained). Fall back to the base model only if something is missing.
    copied = copy_aux_files(src, out)
    if "config.json" not in copied and model_name:
        base = resolve_checkpoint(model_name)
        copied += copy_aux_files(base, out)

    if clamped_total:
        print(
            f"[decompress] WARNING: clamped {clamped_total:,} values that exceed "
            f"the fp16 range (+/-65504) while casting passthrough tensors."
        )

    report = {
        "base_model": model_name,
        "profile": qc.get("profile"),
        "out_dtype": out_dtype,
        "elapsed_sec": round(time.time() - started, 1),
        "tensors_written": written,
        "tensors_dequantized": len(tensors_meta),
        "fp16_clamped_values": clamped_total,
        "output_weights_bytes": dir_size_bytes(out, "*.safetensors"),
        "output_all_files_bytes": dir_size_bytes(out),
        "aux_files_copied": copied,
    }
    # Written outside the checkpoint, for the same reason as the compression
    # report: produced checkpoints stay clean.
    (out.parent / f"{out.name}_decompression_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print(
        f"[decompress] wrote {written} tensors, "
        f"{report['output_weights_bytes']:,} bytes "
        f"({report['output_weights_bytes'] / 1024**3:.3f} GiB) to {out}"
    )
    return report
