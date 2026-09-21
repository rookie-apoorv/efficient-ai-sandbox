"""Build the math calibration corpus. Run once, offline, then commit the output.

    python -m compression.build_calibration_set

Writes ``compression/calib_data/math_calib.jsonl``, which ``compress.py`` reads.
Keeping the corpus in the repository means the compression run itself needs no
network access and is byte-for-byte reproducible by anyone who clones it.

What goes in and why
--------------------
GPTQ builds its Hessians from whatever text you feed the model, so the
calibration corpus should look like the activations the model will actually
produce at evaluation time. For this project that means competition-style maths
with long worked solutions, formatted as chat turns.

Three source groups, mixed:

* **Competition problems** (AIME, AMC) -- match the problem *style* and the
  notation-heavy token distribution of olympiad maths.
* **Long chain-of-thought solutions** (NuminaMath-CoT) -- the eval runs with
  thinking mode on and a 32K token budget, so most tokens generated at eval time
  are mid-reasoning tokens. Calibrating only on short answers would build
  Hessians for a distribution the model barely visits.
* **MATH** -- broader topic coverage so that experts specialising in algebra,
  geometry, number theory and combinatorics all receive traffic. With top-k
  routing, a narrow corpus leaves some experts with near-empty Hessians, and
  those experts silently degrade to plain RTN.

A note on contamination
-----------------------
Calibrating on *public* competition problems is standard domain-specific PTQ and
is fine. Calibrating on the actual hidden evaluation questions would not be, and
is also unnecessary -- GPTQ only needs the activation statistics of the domain,
not the specific problems.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).parent / "calib_data" / "math_calib.jsonl"

# (hf_id, config, split, problem_field, solution_field, max_rows)
SOURCES = [
    ("AI-MO/aimo-validation-aime", None, "train", "problem", "solution", 200),
    ("AI-MO/aimo-validation-amc", None, "train", "problem", "solution", 200),
    ("AI-MO/NuminaMath-CoT", None, "train", "problem", "solution", 1500),
    ("HuggingFaceH4/MATH-500", None, "test", "problem", "solution", 500),
]

MIN_SOLUTION_CHARS = 200  # drop bare answers; we want reasoning traces
TARGET_ROWS = 2000


def _first_present(row: dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def collect() -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    for hf_id, cfg, split, pfield, sfield, limit in SOURCES:
        try:
            ds = load_dataset(hf_id, cfg, split=split) if cfg else load_dataset(hf_id, split=split)
        except Exception as exc:
            print(f"  [skip] {hf_id}: {exc}")
            continue

        kept = 0
        for row in ds:
            problem = _first_present(row, pfield, "problem", "question")
            solution = _first_present(row, sfield, "solution", "answer", "cot")
            if not problem or len(solution) < MIN_SOLUTION_CHARS:
                continue
            rows.append({"question": problem, "answer": solution, "source": hf_id})
            kept += 1
            if kept >= limit:
                break
        print(f"  [ok]   {hf_id}: {kept} rows")

    return rows


def main() -> None:
    print("Building math calibration corpus ...")
    rows = collect()
    if not rows:
        raise SystemExit(
            "No rows collected. Check network access, or assemble "
            f"{OUT_PATH} by hand (one JSON object per line with "
            '"question" and "answer" fields).'
        )

    random.Random(0).shuffle(rows)
    rows = rows[:TARGET_ROWS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    chars = sum(len(r["question"]) + len(r["answer"]) for r in rows)
    print(
        f"\nWrote {len(rows)} rows to {OUT_PATH} "
        f"({chars / 1e6:.1f} M chars, roughly {chars / 4 / 1e3:.0f}K tokens)."
    )
    print("Commit this file to the repository.")


if __name__ == "__main__":
    main()
