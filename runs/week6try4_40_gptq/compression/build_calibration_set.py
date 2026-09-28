"""Build the math calibration corpus from the BASE MODEL'S OWN thinking traces.

Run once, offline, on a GPU box (molab). Commit the output JSONL to the repo so
`compress.py` never needs the network and the run is byte-for-byte reproducible.

    python -m compression.build_calibration_set

WHY SELF-GENERATED TRACES
-------------------------
GPTQ builds Hessians H = 2*E[x x^T] from whatever text you push through the
model. The eval runs with thinking ON, temperature 1.0, up to 32K tokens -- so
the overwhelming majority of tokens the compressed model will ever process are
its OWN <think> tokens: hedging, restarts, self-correction, long arithmetic.

Human-written dataset solutions (NuminaMath, MATH) are terse, polished LaTeX.
Calibrating on those estimates H for a distribution the model never visits.
Generating the corpus with the bf16 base model, under the eval's exact sampling
parameters, matches the distribution by construction. No labels are used, so
there is nothing to contaminate.

WHAT IS AND IS NOT FILTERED
---------------------------
* NOT filtered on correctness. The compressed model will get plenty of eval
  problems wrong; calibrating only on correct traces fits a sub-distribution.
  Worse, correctness filtering keeps the EASY problems (those are the ones the
  model solves), which is the opposite of what we want.
* NOT filtered on truncation. A trace that hit the token cap is the most
  valuable data we have -- it is the deep-recurrent regime that short windows
  never reach.
* IS filtered on DEGENERACY. The real reason many traces truncate is a
  repetition loop, and degenerate repetition has pathological (low-entropy,
  self-similar) activation statistics that would poison H. Filter the
  pathology, not the outcome.

Correctness and truncation are still RECORDED per row, so later experiments can
slice on them without regenerating.

OUTPUT
------
`compression/calib_data/math_traces.jsonl`, one object per trace:
    {"prompt": <rendered user turn>, "trace": <raw model output>,
     "n_tokens": int, "source": str, "level": str|None,
     "truncated": bool, "has_boxed": bool, "correct": bool|None,
     "rep_frac": float, "distinct_frac": float}

The script ends by printing a TOKEN-LENGTH HISTOGRAM. That histogram is what
decides CALIB_SEQLEN -- do not pick the sequence length before reading it.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

OUT_PATH = Path(__file__).parent / "calib_data" / "math_traces.jsonl"

MODEL_PATH = "Qwen/Qwen3.5-4B"   # local dir is faster if already downloaded

# --------------------------------------------------------------------------
# Eval-exact prompt and sampling. Do not "clean up" the instruction string:
# it is copied byte for byte from configs/eval_config.yaml, trailing spaces
# included, because those affect tokenisation.
# --------------------------------------------------------------------------
MATH_INSTRUCTION = (
    "Answer the following mathematics problem. \n"
    "Please reason step by step, but keep the reasoning concise. \n"
    "After reasoning, output the final answer on its own last line using "
    "exactly this format: \\boxed{X}, \n"
    "where X is the final mathematical answer. Generate \\boxed{X} exactly once. \n"
    "Do not write anything after that line.\n"
)

SAMPLING = dict(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,     # eval sets this; it strongly reshapes the token
    repetition_penalty=1.0,   # distribution, so calibration must match it
    max_tokens=32000,
)

N_PROMPTS = 1500
SEED = 0

# Degeneracy thresholds. Chosen conservatively; the script prints the full
# distribution of both statistics so you can retune from data.
REP_NGRAM = 25          # word n-gram length for the repetition detector
REP_FRAC_MAX = 0.30     # drop if the top n-gram covers >30% of the trace
DISTINCT_FRAC_MIN = 0.08  # drop if <8% of words are unique

# Stratified prompt sources: (hf_id, config, split, problem_field, answer_field,
# level_field, n_wanted). Difficulty SPREAD is deliberate -- the Hessian sums
# over tokens, so a hard problem emitting a 12K-token trace already contributes
# 30x what an easy 400-token one does. Over-selecting hard problems on top of
# that risks mismatching a hidden set whose shipped format sample looks like
# "What is the sum of all factors of 100?".
SOURCES = [
    ("EleutherAI/hendrycks_math", "algebra",              "train", "problem", "solution", "level", 200),
    ("EleutherAI/hendrycks_math", "number_theory",        "train", "problem", "solution", "level", 200),
    ("EleutherAI/hendrycks_math", "counting_and_probability", "train", "problem", "solution", "level", 150),
    ("EleutherAI/hendrycks_math", "geometry",             "train", "problem", "solution", "level", 150),
    ("EleutherAI/hendrycks_math", "intermediate_algebra", "train", "problem", "solution", "level", 200),
    ("EleutherAI/hendrycks_math", "precalculus",          "train", "problem", "solution", "level", 100),
    ("AI-MO/NuminaMath-CoT",      None,                   "train", "problem", "solution", None,    400),
    ("AI-MO/aimo-validation-amc", None,                   "train", "problem", "answer",   None,    60),
    ("AI-MO/aimo-validation-aime", None,                  "train", "problem", "answer",   None,    40),
]


# ==========================================================================
# helpers
# ==========================================================================
def extract_boxed(text: str) -> str | None:
    """Last balanced \\boxed{...}; mirrors evaluation/common.py."""
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


def rep_stats(text: str, n: int = REP_NGRAM) -> tuple[float, float]:
    """(fraction of words covered by the most frequent n-gram, distinct-word fraction)."""
    words = text.split()
    if len(words) < 2 * n:
        return 0.0, 1.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    _, cnt = grams.most_common(1)[0]
    return min(1.0, cnt * n / len(words)), len(set(words)) / len(words)


def norm_answer(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s or None


# ==========================================================================
# 1. collect prompts
# ==========================================================================
def collect_prompts() -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    for hf_id, cfg, split, pf, af, lf, want in SOURCES:
        try:
            ds = load_dataset(hf_id, cfg, split=split) if cfg else load_dataset(hf_id, split=split)
        except Exception as exc:
            print(f"  [skip] {hf_id} {cfg or ''}: {exc}")
            continue
        kept = 0
        for row in ds:
            problem = (row.get(pf) or "").strip()
            if not problem:
                continue
            rows.append({
                "problem": problem,
                "ref_answer": (row.get(af) or "").strip() or None,
                "source": f"{hf_id}{'/' + cfg if cfg else ''}",
                "level": str(row.get(lf)) if lf and row.get(lf) is not None else None,
            })
            kept += 1
            if kept >= want:
                break
        print(f"  [ok]   {hf_id} {cfg or '':<26} {kept:>5} prompts")

    random.Random(SEED).shuffle(rows)
    return rows[:N_PROMPTS]


# ==========================================================================
# 2. generate traces with vLLM
# ==========================================================================
def generate(prompts: list[dict]) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    rendered = []
    for r in prompts:
        # Byte-identical to evaluation/common.py build_math_prompt()
        user = f"{MATH_INSTRUCTION}\n\nProblem:\n{r['problem'].strip()}"
        rendered.append(tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))
        r["prompt"] = user

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=33000, gpu_memory_utilization=0.90)
    outs = llm.generate(rendered, SamplingParams(**SAMPLING))

    for r, o in zip(prompts, outs):
        gen = o.outputs[0]
        r["trace"] = gen.text
        r["n_tokens"] = len(gen.token_ids)
        r["truncated"] = gen.finish_reason == "length"
        pred = extract_boxed(gen.text)
        r["has_boxed"] = pred is not None
        ref = norm_answer(r.get("ref_answer"))
        r["correct"] = (norm_answer(pred) == ref) if (pred and ref) else None
        r["rep_frac"], r["distinct_frac"] = rep_stats(gen.text)
    return prompts


# ==========================================================================
# 3. filter, report, write
# ==========================================================================
def histogram(values, edges, label):
    print(f"\n  {label}")
    total = len(values) or 1
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = sum(1 for v in values if lo <= v < hi)
        bar = "#" * int(60 * n / total)
        print(f"    [{lo:>6} , {hi:>6}) {n:>5} {bar}")
    n = sum(1 for v in values if v >= edges[-1])
    print(f"    [{edges[-1]:>6} ,    inf) {n:>5} {'#' * int(60 * n / total)}")


def main() -> None:
    print("Collecting prompts ...")
    prompts = collect_prompts()
    print(f"  -> {len(prompts)} prompts\n")

    print(f"Generating traces with vLLM (cap {SAMPLING['max_tokens']} tokens) ...")
    rows = generate(prompts)

    degenerate = [r for r in rows
                  if r["rep_frac"] > REP_FRAC_MAX or r["distinct_frac"] < DISTINCT_FRAC_MIN]
    keep = [r for r in rows if r not in degenerate]

    n_tok = [r["n_tokens"] for r in keep]
    total_tokens = sum(n_tok)

    print("\n" + "=" * 70)
    print("CORPUS REPORT")
    print("=" * 70)
    print(f"  traces generated        : {len(rows)}")
    print(f"  dropped as degenerate   : {len(degenerate)}")
    print(f"  kept                    : {len(keep)}")
    print(f"  total tokens kept       : {total_tokens:,}")
    print(f"  truncated (hit 32K cap) : {sum(r['truncated'] for r in keep)}")
    print(f"  emitted \\boxed{{}}        : {sum(r['has_boxed'] for r in keep)}")
    scored = [r for r in keep if r["correct"] is not None]
    if scored:
        print(f"  correct (where scorable): {sum(r['correct'] for r in scored)}/{len(scored)} "
              f"= {sum(r['correct'] for r in scored)/len(scored)*100:.1f}%")

    histogram(n_tok, [0, 512, 1024, 2048, 4096, 8192, 16384, 32000],
              "TRACE LENGTH (tokens) -- THIS DECIDES CALIB_SEQLEN")
    for L in (2048, 4096, 8192, 16384):
        n = sum(1 for t in n_tok if t >= L)
        print(f"    traces >= {L:>5} tokens: {n:>5}  ({n/max(len(n_tok),1)*100:5.1f}%)  "
              f"-> {sum(t for t in n_tok if t >= L):,} tokens live at or above this length")

    histogram([round(r["rep_frac"], 2) for r in rows], [0, 0.1, 0.2, 0.3, 0.5, 0.8],
              "repetition fraction (top-25gram coverage) -- retune REP_FRAC_MAX from this")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        for r in keep:
            fh.write(json.dumps({
                "prompt": r["prompt"], "trace": r["trace"], "n_tokens": r["n_tokens"],
                "source": r["source"], "level": r["level"], "truncated": r["truncated"],
                "has_boxed": r["has_boxed"], "correct": r["correct"],
                "rep_frac": round(r["rep_frac"], 4),
                "distinct_frac": round(r["distinct_frac"], 4),
            }, ensure_ascii=False) + "\n")

    print(f"\n  wrote {len(keep)} rows -> {OUT_PATH}")
    print("  COMMIT THIS FILE to the repository.")
    print("=" * 70)


if __name__ == "__main__":
    main()
