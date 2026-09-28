"""Decide a per-tensor bit width (int4 or int8) subject to a global size budget.

A greedy knapsack: every tensor starts at int4; tensors are promoted to int8
in order of *expected output-error reduction per extra byte* until the budget
is spent.

    gain_i  = (err4_i - err8_i) * E_i          (E_i = input energy, see energy.py)
    extra_i = bytes8_i - bytes4_i
    score_i = gain_i / extra_i

With ``E_i = 1`` for every tensor this is week6's data-free planner, kept as
``PLANNER = "plain"`` for comparison. The greedy solution is optimal for the
continuous relaxation and, since no single tensor is a large fraction of the
budget, within a hair of the integer optimum.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional


def plan_bit_widths(
    candidates: List[dict],
    fixed_bytes: int,
    original_bytes: int,
    target_ratio: float,
    base_bits: int = 4,
    high_bits: int = 8,
    weights: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, int], int]:
    """Assign ``base_bits`` or ``high_bits`` to every candidate.

    Each candidate is a dict with ``name``, ``bytes`` (bits -> byte cost) and
    ``err`` (bits -> squared weight error). ``weights`` maps name -> input
    energy; ``None`` means the data-free planner. Also stores each candidate's
    ``score`` for reporting.
    """
    budget = int(target_ratio * original_bytes)
    bits_by_name: Dict[str, int] = {c["name"]: base_bits for c in candidates}
    total = fixed_bytes + sum(c["bytes"][base_bits] for c in candidates)

    for cand in candidates:
        extra = cand["bytes"][high_bits] - cand["bytes"][base_bits]
        w = 1.0 if weights is None else weights[cand["name"]]
        gain = (cand["err"][base_bits] - cand["err"][high_bits]) * w
        cand["gain"] = gain
        cand["score"] = gain / extra if extra > 0 else 0.0

    if total > budget:
        return bits_by_name, total

    ranked = sorted(
        (i for i, c in enumerate(candidates) if c["score"] > 0),
        key=lambda i: candidates[i]["score"],
        reverse=True,
    )
    for idx in ranked:
        cand = candidates[idx]
        extra = cand["bytes"][high_bits] - cand["bytes"][base_bits]
        if total + extra <= budget:
            total += extra
            bits_by_name[cand["name"]] = high_bits

    return bits_by_name, total


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def role_of(name: str) -> str:
    """Short role label, e.g. ``linear_attn.in_proj_qkv`` or ``embed_tokens``."""
    m = _LAYER_RE.search(name)
    if m:
        return name[m.end():].removesuffix(".weight")
    return name.split(".")[-2] if name.endswith(".weight") else name


def layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def format_plan_report(
    candidates: List[dict],
    bits_by_name: Dict[str, int],
    fixed_bytes: int,
    original_bytes: int,
    projected_bytes: int,
    planner: str = "energy",
    high_bits: int = 8,
) -> str:
    n_high = sum(1 for c in candidates if bits_by_name[c["name"]] == high_bits)
    n_low = len(candidates) - n_high
    w_high = sum(c["numel"] for c in candidates if bits_by_name[c["name"]] == high_bits)
    w_low = sum(c["numel"] for c in candidates if bits_by_name[c["name"]] != high_bits)
    residual = sum(
        c["gain"] for c in candidates if bits_by_name[c["name"]] != high_bits
    )  # error left on the table by int4 tensors, in planner units
    total_gain = sum(c["gain"] for c in candidates) or 1.0

    lines = [
        "",
        "=" * 72,
        f"Compression plan  (planner = {planner})",
        "=" * 72,
        f"  tensors @ int4          : {n_low:>6}  ({w_low / 1e9:.3f} B weights)",
        f"  tensors @ int8          : {n_high:>6}  ({w_high / 1e9:.3f} B weights)",
        f"  stored losslessly + perm: {fixed_bytes / 2**20:>9.1f} MiB",
        f"  original TEXT tower     : {original_bytes / 2**30:>9.4f} GiB",
        f"  projected compressed    : {projected_bytes / 2**30:>9.4f} GiB",
        f"  projected size_frac     : {projected_bytes / max(original_bytes, 1):>9.4f}",
        f"  promotable gain captured: {(1 - residual / total_gain) * 100:>8.1f} %",
        "-" * 72,
        f"  {'role':<28}{'n':>4}{'@int8':>7}{'MiB@int8':>11}{'mean score':>14}",
    ]
    by_role = defaultdict(list)
    for c in candidates:
        by_role[role_of(c["name"])].append(c)
    for role, cs in sorted(by_role.items(), key=lambda kv: -max(c["score"] for c in kv[1])):
        hi = [c for c in cs if bits_by_name[c["name"]] == high_bits]
        mib = sum(c["bytes"][high_bits] for c in hi) / 2**20
        mean_score = sum(c["score"] for c in cs) / len(cs)
        lines.append(f"  {role:<28}{len(cs):>4}{len(hi):>7}{mib:>11.1f}{mean_score:>14.4g}")

    promoted_layers = sorted(
        {layer_of(c["name"]) for c in candidates
         if bits_by_name[c["name"]] == high_bits and layer_of(c["name"]) is not None}
    )
    lines += ["-" * 72, f"  layers with any int8    : {promoted_layers}", "=" * 72]
    return "\n".join(lines)
