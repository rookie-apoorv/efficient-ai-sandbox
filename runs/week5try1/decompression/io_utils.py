"""Local checkpoint I/O.

Nothing here instantiates a ``transformers`` model or touches the network: the
base checkpoint is streamed tensor-by-tensor straight off disk.  That keeps peak
memory at roughly one shard, works on a CPU-only node, and is immune to
``vllm`` / ``transformers`` not yet registering the target architecture class.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Files that describe the model but are not weights; these must ride along so
# the compressed repo is self-contained and the restored repo is loadable.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".ckpt")
_SKIP_NAMES = {
    "compression_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "compressed_model.safetensors.index.json",
    ".gitattributes",
    "LICENSE",
    "NOTICE",
}
# Documentation, code and images are explicitly forbidden inside the submitted
# Hugging Face checkpoint. Note that ``.txt`` is NOT skipped: Qwen tokenizers
# ship ``merges.txt``, and dropping it would silently break the tokenizer.
_SKIP_SUFFIXES = {
    ".md",
    ".ipynb",
    ".py",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".log",
    ".lock",
}


def resolve_checkpoint_dir(checkpoint_path: str | None, model_name: str) -> Path:
    """Prefer an explicit local ``--checkpoint_path``, else treat the model name as a path."""
    for candidate in (checkpoint_path, model_name):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir():
                return path
    raise FileNotFoundError(
        "No local checkpoint directory found. Pass --checkpoint_path pointing at "
        "the downloaded base model directory (this pipeline never downloads "
        "weights from the Hugging Face Hub)."
    )


def list_weight_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*.safetensors"))
    if files:
        return files
    files = sorted(directory.glob("*.bin"))
    if files:
        return files
    raise FileNotFoundError(f"No .safetensors or .bin weight files found in {directory}")


def iter_tensors(directory: Path) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every tensor in the checkpoint, in file order."""
    for path in list_weight_files(directory):
        if path.suffix == ".safetensors":
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    yield key, handle.get_tensor(key)
        else:
            shard = torch.load(str(path), map_location="cpu", weights_only=True)
            for key, tensor in shard.items():
                yield key, tensor
            del shard


def copy_auxiliary_files(src: Path, dst: Path) -> List[str]:
    """Copy config / tokenizer / preprocessor files, never weights or docs.

    The assignment forbids shipping READMEs, notebooks, logs or experiment
    outputs inside the Hugging Face checkpoint, so those are filtered out while
    everything needed to load the model is carried across.
    """
    copied: List[str] = []
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
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
    """Accumulate tensors and flush them to size-bounded safetensors shards."""

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        max_shard_bytes: int = 4_000_000_000,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_shard_bytes = max_shard_bytes

        self._buffer: Dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._shard_paths: List[Path] = []
        self._weight_map: Dict[str, str] = {}
        self._pending_names: List[List[str]] = []
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
        tmp_path = self.out_dir / f"{self.prefix}-part{len(self._pending_names):05d}.tmp"
        save_file(self._buffer, str(tmp_path), metadata={"format": "pt"})
        self._shard_paths.append(tmp_path)
        self._buffer = {}
        self._buffer_bytes = 0

    def finalize(self) -> Dict[str, str]:
        """Rename shards to their final names and write the index if sharded."""
        self._flush()
        n = len(self._shard_paths)
        if n == 0:
            raise RuntimeError("nothing was written")

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
            index_path = self.out_dir / f"{self.prefix}.safetensors.index.json"
            index_path.write_text(json.dumps(index, indent=2))

        return self._weight_map
