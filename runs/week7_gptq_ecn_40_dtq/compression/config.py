"""All tuning constants (week8_ec_gptq_40_dtw: decision-token-weighted Fisher).

Method: codec E grid + Fisher-weighted bit allocation + Fisher outliers + GPTQ
on that grid + rANS entropy coding. Calibration: base-model AIME 2022-2026
traces (grader sampling settings), bundled as token ids.

Codec E for one weight tensor W, per row and group of GROUP_SIZE columns:
    step  = fp16(k_t * RMS(group))                (k_t chosen per tensor)
    code  = integer chosen by GPTQ on the grid {..., -2, -1, 0, 1, 2, ...} * step
            (no clipping: any integer is allowed)
    w_hat = code * step
The integer codes are entropy-coded with rANS. OUTLIER_FRAC of the weights
(largest Fisher x RMS^2) are stored exactly as fp16.

``CS6013_SMOKE=1`` shrinks only the Fisher pass and the GPTQ calibration set;
planning, GPTQ over every layer, encoding and the size check all run for real.
Never submit a smoke output.
"""

from __future__ import annotations

import os
from pathlib import Path

SMOKE = os.environ.get("CS6013_SMOKE", "0") == "1"

# ---- size budget ---------------------------------------------------------------
TARGET_RATIO = 0.39           # planned size_frac
HARD_LIMIT = 0.40             # compress.py refuses to finish above this
GRADER_TEXT_GIB = 8.0585      # graders' denominator

# ---- codec E -------------------------------------------------------------------
GROUP_SIZE = 512
MIN_NUMEL = 1 << 20           # smaller tensors stay bf16
K_MIN, K_MAX, K_STEPS = 0.02, 1.6, 96   # step multipliers searched by the planner
OUTLIER_FRAC = 0.0025

# ---- calibration data (built once by compression/build_calib.py) -----------------
CALIB_FILE = Path(__file__).parent / "calib_data" / "aime_calib.npz"

# ---- Fisher --------------------------------------------------------------------
FISHER_SAMPLES = 256          # loss on the completion tokens only
FISHER_MAX_TOKENS = 4096
# Decision-token weighting: in the Fisher loss, positions where the model decides
# whether to second-guess itself count DECISION_WEIGHT times more. A position is a
# decision point when the target token is a decision word (Wait, But, Alternatively,
# Hmm, Actually, However, Hold, </think>) or the previous token ends a sentence or
# paragraph. Bits then flow to the weights that control these choices.
DECISION_WEIGHT = 5.0

# ---- planner -------------------------------------------------------------------
TABLE_MAX_ROWS = 2048

# ---- GPTQ ----------------------------------------------------------------------
METHOD = "gptq"               # "gptq" | "rtn"
GPTQ_MAX_TOKENS = 2_000_000   # calibration tokens pushed through every layer
PERCDAMP = 0.01
ACT_ORDER = True              # static groups: steps fixed in original column order
BLOCK_SIZE = 128
HESSIAN_BUDGET_GB = 8.0
CACHE_DEVICE = "cuda"         # per-sample hidden states (about 5 KiB per token)
# Re-plan the remaining layers when the bits actually used by GPTQ exceed the
# plan by more than this fraction.
RATE_TOLERANCE = 0.002

# ---- pruning -------------------------------------------------------------------
DROP_COMPONENTS = ("visual", "vision_tower", "vision_model", "video_tower",
                   "image_newline", "multi_modal_projector", "mtp")

# ---- output --------------------------------------------------------------------
SHARD_PREFIX = "compressed_model"
MAX_SHARD_BYTES = 4_000_000_000
SEED = 0

if SMOKE:
    FISHER_SAMPLES = 4
    FISHER_MAX_TOKENS = 512
    GPTQ_MAX_TOKENS = 8192
