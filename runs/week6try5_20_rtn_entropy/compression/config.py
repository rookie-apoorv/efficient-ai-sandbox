"""All tuning constants (week7_ec_rtn_20).

Method: codec E + Fisher-weighted bit allocation + Fisher outliers + rANS.

* Every quantizable weight is stored as ``q = round(w / step)`` with
  ``step = k_t * RMS(group)`` -- a uniform grid with **no clipping** -- and the
  integer codes are entropy-coded (rANS), so rare large codes simply cost more
  bits instead of being clipped.
* ``k_t`` is chosen per tensor by a Lagrangian allocation that minimises the
  Fisher-predicted loss increase ``0.5 * sum F * dW^2`` under the size budget.
* The ``OUTLIER_FRAC`` weights with the largest ``F * step^2`` are stored
  exactly (fp16) and marked in the code stream with an escape symbol.
* RTN (no GPTQ) in this version.

Measured on the real model in codec_lab2 (RTN, size_frac 0.195): KL 0.033 vs
0.216 for uniform int3/g128.
"""

from __future__ import annotations

import os
from pathlib import Path

SMOKE = os.environ.get("CS6013_SMOKE", "0") == "1"

# ---- size budget ---------------------------------------------------------------
TARGET_RATIO = 0.198          # planned size_frac; the graders fail anything > 0.20
HARD_LIMIT = 0.200
GRADER_TEXT_GIB = 8.0585      # graders' denominator, used for the final check

# ---- codec E -------------------------------------------------------------------
GROUP_SIZE = 512              # columns per step (falls back to a divisor of cols)
MIN_NUMEL = 1 << 20           # smaller tensors stay bf16
K_MIN, K_MAX, K_STEPS = 0.2, 1.6, 64   # step multipliers searched by the planner
OUTLIER_FRAC = 0.0025         # exact fp16 weights per tensor, chosen by F * step^2

# ---- Fisher --------------------------------------------------------------------
CALIB_FILE = Path(__file__).parent / "calib_data" / "traces_all.jsonl"
CALIB_SHA256 = "a52f276deee2613486e5c1ed4addd0555915ce2c7481f971dda79c69c61d88d8"
FISHER_SAMPLES = 64
FISHER_MAX_TOKENS = 2048
# The same held-out traces the codec lab scored KL on are excluded from the
# Fisher set, so the verification KL in the notebook is out-of-sample.
HOLDOUT_MAX_LEN = 4096
HOLDOUT_TOKENS = 24_000
SEED = 0

# ---- planner -------------------------------------------------------------------
TABLE_MAX_ROWS = 2048         # rows sampled per tensor to build rate/distortion tables
REPLAN_ROUNDS = 4             # re-plan if the exact size overshoots the target

# ---- pruning -------------------------------------------------------------------
DROP_COMPONENTS = ("visual", "vision_tower", "vision_model", "video_tower",
                   "image_newline", "multi_modal_projector", "mtp")

# ---- output --------------------------------------------------------------------
SHARD_PREFIX = "compressed_model"
MAX_SHARD_BYTES = 4_000_000_000

if SMOKE:
    FISHER_SAMPLES = 4
    FISHER_MAX_TOKENS = 512
