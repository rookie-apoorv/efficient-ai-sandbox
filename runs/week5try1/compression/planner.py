"""Decide a per-tensor bit width subject to a global size budget.

The 40% target leaves real headroom over uniform 4-bit: at group size 64,
uniform int4 lands near 28% of the float16 baseline, so roughly 45% of the
weights can be promoted to int8 for free.  Spending that headroom uniformly is
wasteful -- most tensors quantize almost losslessly at 4 bits, while a minority
(typically the first and last blocks, the attention output projection and the
MoE down-projections) dominate the total error.

This planner therefore solves a small knapsack: promote the tensors with the
largest *error reduction per extra byte* until the budget is exhausted.  The
greedy solution is optimal for the continuous relaxation and, because no single
tensor is a large fraction of the budget, within a hair of optimal here.

Everything is data-free -- no calibration set is required, which keeps
``compress.py`` a pure function of the checkpoint.
"""

from __future__ import annotations

from typing import Dict, List


def plan_bit_widths(
    candidates: List[dict],
    fixed_bytes: int,
    original_bytes: int,
    target_ratio: float,
    base_bits: int = 4,
    high_bits: int = 8,
) -> tuple[Dict[str, int], int]:
    """Assign ``base_bits`` or ``high_bits`` to every candidate tensor.

    Parameters
    ----------
    candidates:
        One dict per quantizable tensor with keys ``name``, ``bytes`` (a mapping
        ``bits -> byte cost``) and ``err`` (a mapping ``bits -> squared error``).
    fixed_bytes:
        Total size of everything stored losslessly (norms, biases, routers).
    original_bytes:
        Size of the base checkpoint's tensors, the denominator of the ratio.
    target_ratio:
        Upper bound on ``compressed_bytes / original_bytes``.

    Returns
    -------
    ``(bits_by_name, projected_total_bytes)``
    """
    budget = int(target_ratio * original_bytes)
    bits_by_name: Dict[str, int] = {c["name"]: base_bits for c in candidates}
    total = fixed_bytes + sum(c["bytes"][base_bits] for c in candidates)

    if total > budget:
        # Nothing to promote; the caller decides whether to warn or hard-fail.
        return bits_by_name, total

    ranked = []
    for idx, cand in enumerate(candidates):
        extra = cand["bytes"][high_bits] - cand["bytes"][base_bits]
        gain = cand["err"][base_bits] - cand["err"][high_bits]
        if extra > 0 and gain > 0:
            ranked.append((gain / extra, idx))
    ranked.sort(reverse=True)

    for _, idx in ranked:
        cand = candidates[idx]
        extra = cand["bytes"][high_bits] - cand["bytes"][base_bits]
        if total + extra <= budget:
            total += extra
            bits_by_name[cand["name"]] = high_bits

    return bits_by_name, total


def format_plan_report(
    candidates: List[dict],
    bits_by_name: Dict[str, int],
    fixed_bytes: int,
    original_bytes: int,
    projected_bytes: int,
) -> str:
    """Human-readable summary printed by ``compress.py``."""
    n_high = sum(1 for b in bits_by_name.values() if b == 8)
    n_low = len(bits_by_name) - n_high
    weights_high = sum(
        c["numel"] for c in candidates if bits_by_name[c["name"]] == 8
    )
    weights_low = sum(c["numel"] for c in candidates if bits_by_name[c["name"]] == 4)
    lines = [
        "",
        "=" * 68,
        "Compression plan",
        "=" * 68,
        f"  tensors @ int4          : {n_low:>6}  ({weights_low / 1e9:.3f} B weights)",
        f"  tensors @ int8          : {n_high:>6}  ({weights_high / 1e9:.3f} B weights)",
        f"  stored losslessly       : {fixed_bytes / 2**20:>9.1f} MiB",
        f"  original checkpoint     : {original_bytes / 2**30:>9.3f} GiB",
        f"  projected compressed    : {projected_bytes / 2**30:>9.3f} GiB",
        f"  projected ratio         : {projected_bytes / max(original_bytes, 1):>9.4f}",
        "=" * 68,
    ]
    return "\n".join(lines)
