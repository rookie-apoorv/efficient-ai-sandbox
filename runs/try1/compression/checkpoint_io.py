"""Streaming safetensors I/O.

The single most important design decision in this pipeline lives here: we never
instantiate the model. ``compress`` and ``decompress`` walk the checkpoint
**one tensor at a time** through ``safetensors.safe_open``, which memory-maps the
file and materializes only the tensor you ask for.

Why this matters:

* **Memory.** ``AutoModelForCausalLM.from_pretrained`` on Qwen3.5-4B wants ~9.3 GB
  of RAM before you have done any work, and the sample submission then holds a
  second full ``state_dict``. Streaming peaks at roughly one tensor (~47 MB) plus
  one output shard buffer.
* **Robustness.** Nothing here needs the ``qwen3_5`` model class to be registered
  in the installed ``transformers``. That sidesteps the entire class of
  weight-prefix / registration failures the course's Known Issues section warns
  about, and it means compress/decompress run on a laptop CPU with an older
  transformers than the eval needs.
* **Fidelity.** We copy tensors we do not touch byte-for-byte, instead of
  round-tripping them through a model's ``state_dict``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS_INDEX = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_BYTES = 2 * 1024**3  # 2 GiB: fits Kaggle's ~13 GB RAM comfortably

# Files copied verbatim into a produced checkpoint so it is self-contained.
# Deliberately excludes README.md and LICENSE: the project spec forbids shipping
# a README in the HuggingFace checkpoint.
AUX_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def resolve_checkpoint(model_name_or_path: str) -> Path:
    """Return a local directory for a HF repo id or an existing local path."""
    path = Path(model_name_or_path)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    print(f"[io] downloading {model_name_or_path} from the Hub ...")
    return Path(snapshot_download(model_name_or_path))


def weight_files(ckpt_dir: Path) -> list[Path]:
    """The .safetensors shards of a checkpoint, in index order when available."""
    index_path = ckpt_dir / WEIGHTS_INDEX
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
        return [ckpt_dir / n for n in names]

    files = sorted(ckpt_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files under {ckpt_dir}")
    return files


def weights_nbytes(ckpt_dir: Path) -> int:
    """Total bytes of the weight shards (the denominator for the size ratio)."""
    return sum(f.stat().st_size for f in weight_files(ckpt_dir))


def iter_tensors(ckpt_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every tensor, one shard at a time."""
    for shard in weight_files(ckpt_dir):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def tensor_manifest(ckpt_dir: Path) -> dict[str, dict]:
    """``{name: {"shape", "dtype", "numel", "nbytes"}}`` without reading any data."""
    manifest: dict[str, dict] = {}
    for shard in weight_files(ckpt_dir):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                slice_ = f.get_slice(name)
                shape = tuple(slice_.get_shape())
                dtype = slice_.get_dtype()
                numel = 1
                for d in shape:
                    numel *= d
                itemsize = {
                    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
                    "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
                }.get(dtype, 2)
                manifest[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "numel": numel,
                    "nbytes": numel * itemsize,
                }
    return manifest


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


class ShardWriter:
    """Buffer tensors and flush them to numbered safetensors shards.

    Use as a context manager; ``add`` accepts tensors in any order and ``close``
    writes ``model.safetensors.index.json``.
    """

    def __init__(self, out_dir: Path, max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._shards: list[list[str]] = []
        self._tmp_paths: list[Path] = []
        self.total_bytes = 0

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if self._buffer and self._buffer_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self._buffer[name] = tensor
        self._buffer_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        if not self._buffer:
            return
        tmp = self.out_dir / f".shard-{len(self._shards):05d}.tmp"
        save_file(self._buffer, str(tmp), metadata={"format": "pt"})
        self._shards.append(list(self._buffer.keys()))
        self._tmp_paths.append(tmp)
        self._buffer.clear()
        self._buffer_bytes = 0

    def close(self) -> None:
        self._flush()
        n = len(self._shards)
        if n == 0:
            raise RuntimeError("ShardWriter.close() called with no tensors written")

        weight_map: dict[str, str] = {}
        if n == 1:
            final_names = ["model.safetensors"]
        else:
            final_names = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

        for tmp, final, names in zip(self._tmp_paths, final_names, self._shards):
            target = self.out_dir / final
            if target.exists():
                target.unlink()
            tmp.rename(target)
            for name in names:
                weight_map[name] = final

        # A single-shard checkpoint does not need an index, but writing one keeps
        # downstream size accounting and tooling uniform.
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": weight_map,
        }
        (self.out_dir / WEIGHTS_INDEX).write_text(json.dumps(index, indent=2))


def copy_aux_files(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy config/tokenizer files so the produced checkpoint is self-contained."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in AUX_FILES:
        src = Path(src_dir) / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def dir_size_bytes(path: Path, pattern: str = "*") -> int:
    return sum(p.stat().st_size for p in Path(path).rglob(pattern) if p.is_file())
