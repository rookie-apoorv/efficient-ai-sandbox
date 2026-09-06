"""Which tensor gets how many bits.

Everything strategic about a compression recipe that isn't the quantizer itself
lives here: eligibility rules and a per-tensor bit/group assignment. A recipe is
a named :class:`QuantPolicy`; ``compress.py --profile <name>`` selects one.

Keeping this separate from ``quantize.py`` is deliberate. Sensitivity-driven
mixed precision -- the main lever for the 20% target -- is a change to this file
alone, with no change to the quantizer or the checkpoint format.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

# Tensors that are never quantized, regardless of profile.
#
# All of these are either <2-D, numerically delicate, or so small that
# quantizing them saves nothing. Combined they are well under 0.1% of the
# checkpoint. See QWEN35_4B_MODEL_NOTES.md section 3, conclusion 5.
DEFAULT_SKIP_PATTERNS: tuple[str, ...] = (
    "*norm*",  # all RMSNorm / LayerNorm weights and biases
    "*.bias",
    "*.A_log",  # DeltaNet decay, fp32, 32 values per layer
    "*.dt_bias",  # DeltaNet timestep bias, fp32
    "*conv1d*",  # depthwise conv, 3-D, [8192, 1, 4]
    "*in_proj_a*",  # [32, 2560] -- tiny, drives the recurrent gate
    "*in_proj_b*",  # [32, 2560] -- tiny, drives the recurrent gate
    "*pos_embed*",  # vision positional table
    "*patch_embed*",  # vision stem
)

# Module families, for reporting and for writing readable per-family overrides.
FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("embedding", r"embed_tokens"),
    ("text_mlp", r"language_model\.layers\.\d+\.mlp\."),
    ("linear_attn", r"language_model\.layers\.\d+\.linear_attn\."),
    ("full_attn", r"language_model\.layers\.\d+\.self_attn\."),
    ("vision", r"^model\.visual\."),
    ("mtp", r"^mtp\."),
)


def tensor_family(name: str) -> str:
    for family, pattern in FAMILY_PATTERNS:
        if re.search(pattern, name):
            return family
    return "other"


@dataclass
class QuantPolicy:
    """A compression recipe.

    Attributes
    ----------
    bits, group_size:
        Defaults applied to every eligible tensor.
    overrides:
        ``{glob_pattern: {"bits": int, "group_size": int, "skip": bool}}``.
        The FIRST matching pattern wins, so order them most-specific first.
    min_numel:
        Tensors smaller than this stay in full precision. Quantizing a 100K-param
        tensor saves ~100 KB and risks real damage; not worth it.
    skip_patterns:
        Never-quantize globs, checked before ``overrides``.
    """

    name: str = "int8"
    bits: int = 8
    group_size: int = 128
    min_numel: int = 1 << 16  # 65,536
    skip_patterns: tuple[str, ...] = DEFAULT_SKIP_PATTERNS
    overrides: dict[str, dict] = field(default_factory=dict)

    def plan(self, name: str, shape: tuple[int, ...]) -> dict | None:
        """Return ``{"bits", "group_size"}`` for this tensor, or None to keep it dense."""
        numel = 1
        for d in shape:
            numel *= d

        if len(shape) < 2 or numel < self.min_numel:
            return None
        if any(fnmatch.fnmatch(name, pat) for pat in self.skip_patterns):
            return None

        bits, group_size = self.bits, self.group_size
        for pattern, cfg in self.overrides.items():
            if fnmatch.fnmatch(name, pattern):
                if cfg.get("skip"):
                    return None
                bits = cfg.get("bits", bits)
                group_size = cfg.get("group_size", group_size)
                break

        return {"bits": bits, "group_size": group_size}


# --------------------------------------------------------------------------- #
# Named profiles
# --------------------------------------------------------------------------- #
#
# Predicted ratios below assume bf16 baseline and are what
# `scripts/estimate_ratio.py` computes analytically. Always confirm against the
# MEASURED ratio that compress.py prints -- prediction is a planning tool, the
# bytes on disk are the grade.

PROFILES: dict[str, QuantPolicy] = {
    # Plumbing baseline. ~51% -> hits NO target. Correctness first.
    "int8": QuantPolicy(name="int8", bits=8, group_size=128),
    # First genuinely submittable recipe: ~26%, clears the 40% target with slack.
    "int4": QuantPolicy(name="int4", bits=4, group_size=128),
    # 40% target, spending the slack on quality: 4-bit bulk, 8-bit where it hurts.
    # Predicted 34.63% -- verified against real Qwen3.5-4B shapes.
    "mixed40": QuantPolicy(
        name="mixed40",
        bits=4,
        group_size=128,
        overrides={
            # down_proj reads post-SiLU activations with heavy outliers.
            "*mlp.down_proj.weight": {"bits": 8, "group_size": 128},
            # Tied embedding sets both input geometry and output logits.
            "*embed_tokens.weight": {"bits": 8, "group_size": 128},
            # Only 8 full-attention layers exist; keeping them at 8 bits is cheap.
            "*self_attn.*": {"bits": 8, "group_size": 128},
            # Off the math path entirely -- free budget.
            "model.visual.*": {"bits": 2, "group_size": 128},
            "mtp.*": {"bits": 2, "group_size": 128},
        },
    ),
    # 20% target. Predicted 19.47% -- only 0.53 points of headroom, so verify the
    # MEASURED ratio before submitting.
    #
    # This one is genuinely tight, and the arithmetic is instructive: uniform
    # 3-bit/G=128 across the whole model is 20.08% and MISSES. The budget only
    # closes by spending the free 9.75% (vision + MTP, off the math path) and
    # taking the tied embedding down to 2 bits. Candidates measured with
    # scripts/estimate_ratio.py:
    #     3-bit g128 uniform ......................... 20.08%  OVER
    #     + vis/mtp 2-bit, emb 3-bit, down_proj 4-bit  20.82%  OVER
    #     + emb 3-bit, down_proj 3-bit g64 ........... 19.97%  ok, no headroom
    #     this profile ............................... 19.47%  ok
    #
    # The 2-bit tied embedding is the risky part -- it sets both input geometry
    # and output logits. Ablate it first if quality collapses; the fallback is
    # the 19.97% variant with a 3-bit embedding.
    "mixed20": QuantPolicy(
        name="mixed20",
        bits=3,
        group_size=128,
        overrides={
            # Smaller groups buy accuracy back on the post-SiLU outlier layer
            # more cheaply than a whole extra bit would.
            "*mlp.down_proj.weight": {"bits": 3, "group_size": 64},
            "*embed_tokens.weight": {"bits": 2, "group_size": 256},
            # k/v are 2.6M params each across only 8 layers -- 8 bits is free.
            "*self_attn.k_proj.weight": {"bits": 8, "group_size": 128},
            "*self_attn.v_proj.weight": {"bits": 8, "group_size": 128},
            "*self_attn.q_proj.weight": {"bits": 4, "group_size": 128},
            "*self_attn.o_proj.weight": {"bits": 4, "group_size": 128},
            "model.visual.*": {"bits": 2, "group_size": 256},
            "mtp.*": {"bits": 2, "group_size": 256},
        },
    ),
    # Deliberately NO "mixed10" profile. The 10% target is 1.60 bits/param, and
    # uniform 2-bit scalar quantization floors at 12.96% even with G=512. Getting
    # under 10% requires vector quantization (AQLM / QuIP#), structured pruning,
    # or low-rank + low-bit residual -- a new method, not a new bit-width. See
    # QWEN35_4B_MODEL_NOTES.md section 6.
}


def get_profile(name: str) -> QuantPolicy:
    if name not in PROFILES:
        raise SystemExit(
            f"Unknown profile '{name}'. Available: {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[name]
