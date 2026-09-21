"""Build the CS6013 local math benchmark suites, and score their outputs.

Design
------
Five suites forming a **difficulty ladder** rather than five interchangeable
datasets. Quantization damage does not appear uniformly: a 4-bit model usually
still solves grade-school arithmetic perfectly while falling apart on problems
needing ten chained steps. A single aggregate number hides exactly the signal
worth having, so each suite is scored separately and the *shape* of the curve is
the diagnostic.

    smoke    8 problems, trivial       2 min      "is anything catastrophically broken"
    gsm8k    100, easy word problems   ~10 min    floor check; should stay near baseline
    math500  150, broad topics         ~25 min    topic coverage across all experts
    amc      83, competition           ~25 min    where quantization starts to bite
    aime     90, olympiad              ~45 min    hardest; long reasoning, token-limit stress

The ladder also isolates the token-budget problem. AIME solutions routinely run
past 10k thinking tokens, so a model that degrades into rambling gets truncated
before emitting ``\\boxed{}``. That shows up as a collapsing ``truncation_rate``
rather than as wrong answers, and the two failure modes call for different
fixes, so they are reported separately.

A note on contamination: GSM8K and MATH-500 are almost certainly in Qwen's
training data. That is fine here. These suites measure *compressed vs baseline
on identical inputs*, and contamination shifts both arms equally. Do not read
the absolute numbers as a capability claim.

Schema matches the graders' loader exactly: ``problem_idx`` (int64),
``problem`` (string), ``answer`` (string), saved via ``DatasetDict.save_to_disk``
so ``load_dataset(path, split="train")`` reads it back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA = ["problem_idx", "problem", "answer"]

# Eight problems with unambiguous integer answers, verified by hand. The point
# is a fast binary signal: if a model misses these, something is badly wrong and
# there is no reason to spend an hour on AIME.
SMOKE_PROBLEMS = [
    ("What is the sum of all factors of $100$?", "217"),
    ("What is the sum of all prime factors of $100$?", "7"),
    ("Compute $17 \\times 23$.", "391"),
    ("What is the remainder when $2^{10}$ is divided by $7$?", "2"),
    ("How many positive divisors does $36$ have?", "9"),
    ("What is the value of $5!$?", "120"),
    ("Solve for $x$: $3x + 7 = 22$.", "5"),
    ("What is the greatest common divisor of $48$ and $180$?", "12"),
]


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------
def boxed_answer(text: str) -> str | None:
    """Last balanced ``\\boxed{...}`` payload, matching the graders' extractor."""
    i = text.rfind(r"\boxed")
    if i < 0:
        return None
    start = text.find("{", i)
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j].strip()
    return None


def _clean(value: Any) -> str | None:
    """Coerce a gold answer to a clean string.

    Competition sets store answers as ints or floats, so ``42.0`` must become
    ``"42"`` -- the graders compare normalized strings before falling back to
    symbolic equivalence, and a stray ``.0`` costs an exact match.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def extract_gsm8k(row: dict) -> str | None:
    """GSM8K stores the final answer after a ``####`` marker."""
    raw = row.get("answer") or ""
    if "####" in raw:
        return _clean(raw.split("####")[-1].replace(",", ""))
    return None


def extract_generic(row: dict) -> str | None:
    """Try the common field names, then fall back to a boxed solution."""
    for field in ("answer", "final_answer", "Answer", "solution_answer"):
        value = _clean(row.get(field))
        if value is not None and "####" not in str(value):
            return value
    for field in ("solution", "Solution", "cot", "reasoning"):
        text = row.get(field)
        if isinstance(text, str):
            value = _clean(boxed_answer(text))
            if value is not None:
                return value
    return None


def get_problem(row: dict) -> str | None:
    for field in ("problem", "question", "Problem", "Question", "query"):
        text = row.get(field)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


# --------------------------------------------------------------------------
# Suite specifications
# --------------------------------------------------------------------------
class Suite:
    def __init__(
        self,
        name: str,
        sources: list[tuple[str, str | None, str]],
        limit: int,
        extractor: Callable[[dict], str | None] = extract_generic,
        note: str = "",
    ):
        self.name = name
        self.sources = sources  # (hf_id, config, split)
        self.limit = limit
        self.extractor = extractor
        self.note = note


SUITES = [
    Suite("smoke", [], 8, note="hand-written sanity check, ~2 min"),
    Suite(
        "gsm8k",
        [("openai/gsm8k", "main", "test")],
        100,
        extractor=extract_gsm8k,
        note="easy multi-step arithmetic; the floor",
    ),
    Suite(
        "math500",
        [("HuggingFaceH4/MATH-500", None, "test")],
        150,
        note="broad topic coverage, mixed difficulty",
    ),
    Suite(
        "amc",
        [("AI-MO/aimo-validation-amc", None, "train")],
        83,
        note="AMC12 2022-2023, integer answers",
    ),
    Suite(
        "aime",
        [
            ("AI-MO/aimo-validation-aime", None, "train"),
            ("MathArena/aime_2025", None, "train"),
        ],
        90,
        note="hardest; long reasoning, stresses the 32k token budget",
    ),
]


def rows_for_suite(suite: Suite, load_dataset, log=print) -> list[dict]:
    """Collect and normalize rows for one suite."""
    if suite.name == "smoke":
        return [
            {"problem_idx": i + 1, "problem": p, "answer": a}
            for i, (p, a) in enumerate(SMOKE_PROBLEMS)
        ]

    collected: list[dict] = []
    for hf_id, config, split in suite.sources:
        try:
            ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
        except Exception as exc:
            log(f"    [skip] {hf_id}: {type(exc).__name__}: {exc}")
            continue

        kept = dropped = 0
        for row in ds:
            if len(collected) >= suite.limit:
                break
            problem = get_problem(row)
            answer = suite.extractor(row)
            if not problem or answer is None:
                dropped += 1
                continue
            collected.append(
                {"problem_idx": len(collected) + 1, "problem": problem, "answer": answer}
            )
            kept += 1
        log(f"    [ok]   {hf_id}: kept {kept}, dropped {dropped}")
        if len(collected) >= suite.limit:
            break

    return collected


def build_suites(out_root, load_dataset, Dataset, DatasetDict, log=print) -> dict:
    """Build every suite under ``out_root``. Returns {name: (path, n_rows)}."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    built = {}

    for suite in SUITES:
        log(f"  building '{suite.name}' ({suite.note}) ...")
        rows = rows_for_suite(suite, load_dataset, log=log)
        if not rows:
            log(f"    [FAIL] no rows for '{suite.name}'")
            continue
        path = out_root / f"CS6013_{suite.name}"
        ds = Dataset.from_list(rows)
        # Column order must match the graders' loader expectations.
        ds = ds.select_columns(SCHEMA)
        DatasetDict({"train": ds}).save_to_disk(str(path))
        built[suite.name] = (str(path), len(rows))
        log(f"    -> {len(rows)} problems at {path}")

    return built


def verify_suites(built: dict, load_dataset, log=print) -> list[str]:
    """Round-trip every suite through the graders' loader. Returns failures."""
    failures = []
    for name, (path, n) in sorted(built.items()):
        try:
            ds = load_dataset(path, split="train")
            if ds.column_names != SCHEMA:
                failures.append(f"{name}: columns {ds.column_names} != {SCHEMA}")
            elif len(ds) != n:
                failures.append(f"{name}: {len(ds)} rows != {n}")
            else:
                row = ds[0]
                assert isinstance(row["answer"], str)
                log(f"  {name:<9} OK  {len(ds):>4} rows   e.g. answer={row['answer']!r}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_output(json_path) -> dict:
    """Read a run_eval.py output file and add truncation analysis.

    ``parse_rate`` alone conflates two very different failures: a model that
    answers confidently but wrongly, and one that never stops reasoning and gets
    cut off at the token limit. The second is the characteristic failure of an
    over-quantized model on hard problems, so it is measured separately.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    summary = dict(data["summary"])
    predictions = data.get("predictions", [])

    n = len(predictions)
    no_boxed = 0
    lengths = []
    for pred in predictions:
        response = pred.get("response") or ""
        lengths.append(len(response))
        if r"\boxed" not in response:
            no_boxed += 1

    summary["num_examples"] = summary.get("num_examples", n)
    summary["truncation_rate"] = (no_boxed / n) if n else 0.0
    summary["num_no_boxed"] = no_boxed
    summary["mean_response_chars"] = (sum(lengths) / n) if n else 0
    summary["max_response_chars"] = max(lengths) if lengths else 0
    return summary


def format_results_table(rows: list[dict]) -> str:
    """Markdown table of per-suite results."""
    header = (
        "| suite | n | accuracy | parse rate | no-\\boxed | mean chars |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| `{r['suite']}` | {r['num_examples']} | **{r['accuracy']:.3f}** | "
            f"{r['parse_rate']:.3f} | {r['truncation_rate']:.3f} | "
            f"{r['mean_response_chars']:.0f} |\n"
        )
    return header + body


def format_comparison_table(compressed: list[dict], baseline: list[dict]) -> str:
    """Side-by-side compressed vs baseline, with the delta that matters."""
    base = {r["suite"]: r for r in baseline}
    out = (
        "| suite | n | baseline | compressed | delta | rel. retained |\n"
        "|---|---|---|---|---|---|\n"
    )
    for r in compressed:
        b = base.get(r["suite"])
        if not b:
            continue
        delta = r["accuracy"] - b["accuracy"]
        retained = (r["accuracy"] / b["accuracy"]) if b["accuracy"] else float("nan")
        out += (
            f"| `{r['suite']}` | {r['num_examples']} | {b['accuracy']:.3f} | "
            f"{r['accuracy']:.3f} | {delta:+.3f} | {retained:.1%} |\n"
        )
    return out