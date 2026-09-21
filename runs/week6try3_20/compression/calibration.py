"""Calibration data for GPTQ.

The Hessian is only as good as the activations it is built from, so the
calibration set should match the evaluation distribution: maths problems, in
chat format, with thinking-style worked solutions, at a sequence length long
enough to exercise the long-context behaviour the eval uses.

Two sources are supported:

* ``--calib-file`` -- a local ``.jsonl`` (fields ``text``, or ``question`` +
  ``answer``) or plain ``.txt``. Preferred: no network needed, and you control
  exactly what the model sees.
* ``--calib-dataset gsm8k`` -- pulled through ``datasets`` if the machine has
  network access.

**MoE note.** With ``E`` experts and top-``k`` routing, each expert sees roughly
``k/E`` of the calibration tokens. For 128 experts at top-8 that is 1/16, so a
Hessian of dimension 768 needs on the order of 16 x 768 tokens before it is
even nominally full rank. ``suggest_n_samples`` works that budget out; when in
doubt use more sequences, since a rank-deficient expert Hessian quietly falls
back to near-RTN behaviour for that expert.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional

import torch


# The GPTQ paper and every mainstream implementation calibrate on 128 sequences
# of 2048 tokens. Enough data to make the Hessian well conditioned is the floor,
# not the target: past that point more calibration keeps helping, with
# diminishing returns, and it is cheap relative to the quantization itself.
MIN_CALIB_SEQUENCES = 128


def suggest_n_samples(
    n_experts: int, top_k: int, hessian_dim: int, seqlen: int, oversample: int = 4
) -> int:
    """Sequences needed so the average expert sees ~``oversample`` x its dimension.

    For a dense model this is just ``oversample * hessian_dim`` tokens, which is
    a conditioning floor rather than a quality target -- hence the
    ``MIN_CALIB_SEQUENCES`` floor. For an MoE model each expert only receives
    ``top_k / n_experts`` of the tokens, so the requirement scales up by the
    inverse of the routing fan-out.
    """
    if n_experts <= 1 or top_k <= 0:
        tokens = oversample * hessian_dim
    else:
        tokens = oversample * hessian_dim * n_experts / top_k
    return max(MIN_CALIB_SEQUENCES, math.ceil(tokens / seqlen))


def _load_texts_from_file(path: Path) -> List[str]:
    texts: List[str] = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" in row:
                texts.append(row["text"])
            elif "question" in row:
                answer = row.get("answer", row.get("solution", ""))
                texts.append(f"{row['question']}\n\n{answer}".strip())
            elif "problem" in row:
                answer = row.get("solution", row.get("answer", ""))
                texts.append(f"{row['problem']}\n\n{answer}".strip())
    else:
        blocks = [b.strip() for b in path.read_text().split("\n\n\n") if b.strip()]
        texts = blocks if blocks else [path.read_text()]
    if not texts:
        raise ValueError(f"No usable text found in {path}")
    return texts


def _load_texts_from_hub(name: str, limit: int) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "`datasets` is not installed. Install it, or pass --calib-file "
            "with a local jsonl/txt calibration set."
        ) from exc

    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        return [f"{r['question']}\n\n{r['answer']}" for r in ds.select(range(min(limit, len(ds))))]
    ds = load_dataset(name, split="train")
    field = "text" if "text" in ds.column_names else ds.column_names[0]
    return [str(r[field]) for r in ds.select(range(min(limit, len(ds))))]


def build_calibration(
    tokenizer,
    n_samples: int,
    seqlen: int,
    calib_file,
    seed: int = 0,
    use_chat_template: bool = True,
) -> List[torch.Tensor]:
    """Return ``n_samples`` tensors of shape ``[1, seqlen]`` of token ids.

    The corpus is read from the file committed at ``compression/calib_data/``.
    Building it requires network access, but that happens once, offline, via
    ``compression/build_calibration_set.py`` -- ``compress.py`` itself never
    touches the network.
    """
    rng = random.Random(seed)

    path = Path(calib_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration corpus not found at {path}.\n"
            "Build it once with:\n"
            "    python -m compression.build_calibration_set\n"
            "or set compression/config.py METHOD = 'rtn' to skip calibration."
        )
    texts = _load_texts_from_file(path)

    rng.shuffle(texts)

    # Match the evaluation format. The eval runs with thinking mode on, so the
    # activations the model sees there come from chat-formatted prompts; feeding
    # GPTQ raw text would build Hessians for a distribution the model is never
    # actually run on.
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = []
            for text in texts:
                head, _, tail = text.partition("\n\n")
                formatted.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": head},
                            {"role": "assistant", "content": tail or head},
                        ],
                        tokenize=False,
                    )
                )
            texts = formatted
        except Exception:
            pass  # tokenizer has no usable template; raw text is an acceptable fallback

    joined = "\n\n".join(texts)
    ids = tokenizer(joined, return_tensors="pt").input_ids[0]

    if ids.numel() < seqlen + 1:
        raise ValueError(
            f"Calibration corpus has {ids.numel()} tokens but seqlen is {seqlen}. "
            "Supply more calibration text or lower --calib-seqlen."
        )

    samples = []
    for _ in range(n_samples):
        start = rng.randint(0, ids.numel() - seqlen - 1)
        samples.append(ids[start : start + seqlen].unsqueeze(0))
    return samples
