"""Reading the compressed repo and writing a standard Hugging Face checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".ckpt")
_SKIP_NAMES = {
    "compression_config.json",
    "compressed_model.safetensors.index.json",
    "model.safetensors.index.json",
    ".gitattributes",
    "LICENSE",
    "NOTICE",
}
# ``.txt`` is deliberately absent: Qwen tokenizers ship ``merges.txt``.
_SKIP_SUFFIXES = {".md", ".ipynb", ".py", ".png", ".jpg", ".jpeg", ".gif",
                  ".svg", ".log", ".lock"}


class CompressedStore:
    """Lazy, memory-mapped access to every tensor in the compressed shards."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._handles = []
        self._where: Dict[str, int] = {}

        shards = sorted(self.directory.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors shards found in {directory}")

        for shard in shards:
            handle = safe_open(str(shard), framework="pt", device="cpu")
            idx = len(self._handles)
            self._handles.append(handle)
            for key in handle.keys():
                self._where[key] = idx

    def __contains__(self, key: str) -> bool:
        return key in self._where

    def get(self, key: str) -> torch.Tensor:
        if key not in self._where:
            raise KeyError(f"'{key}' is missing from the compressed checkpoint")
        return self._handles[self._where[key]].get_tensor(key)

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        self._handles = []


def load_compression_config(directory: Path) -> dict:
    path = Path(directory) / "compression_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. --checkpoint_path must point at a directory "
            "produced by compress.py."
        )
    return json.loads(path.read_text())


def copy_auxiliary_files(src: Path, dst: Path) -> List[str]:
    """Carry config / tokenizer files across to the restored checkpoint."""
    copied: List[str] = []
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(Path(src).iterdir()):
        if not item.is_file():
            continue
        if item.suffix in _WEIGHT_SUFFIXES or item.name in _SKIP_NAMES:
            continue
        if item.suffix.lower() in _SKIP_SUFFIXES:
            continue
        shutil.copy2(item, dst / item.name)
        copied.append(item.name)
    return copied


class ShardedSafetensorsWriter:
    """Write ``model.safetensors`` (+ index when sharded) for the restored model."""

    def __init__(
        self, out_dir: Path, prefix: str = "model", max_shard_bytes: int = 4_000_000_000
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_shard_bytes = max_shard_bytes

        self._buffer: Dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._shard_paths: List[Path] = []
        self._pending_names: List[List[str]] = []
        self._weight_map: Dict[str, str] = {}
        self.total_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self._buffer and self._buffer_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self._buffer[name] = tensor.contiguous()
        self._buffer_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._pending_names.append(list(self._buffer.keys()))
        tmp = self.out_dir / f"{self.prefix}-part{len(self._pending_names):05d}.tmp"
        save_file(self._buffer, str(tmp), metadata={"format": "pt"})
        self._shard_paths.append(tmp)
        self._buffer = {}
        self._buffer_bytes = 0

    def finalize(self) -> Dict[str, str]:
        self._flush()
        n = len(self._shard_paths)
        if n == 0:
            raise RuntimeError("no tensors were written")

        for i, tmp in enumerate(self._shard_paths):
            if n == 1:
                target = self.out_dir / f"{self.prefix}.safetensors"
            else:
                target = self.out_dir / f"{self.prefix}-{i + 1:05d}-of-{n:05d}.safetensors"
            tmp.replace(target)
            for name in self._pending_names[i]:
                self._weight_map[name] = target.name

        if n > 1:
            index = {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": self._weight_map,
            }
            (self.out_dir / f"{self.prefix}.safetensors.index.json").write_text(
                json.dumps(index, indent=2)
            )
        return self._weight_map