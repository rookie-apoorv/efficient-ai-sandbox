import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # CS6013 sandbox — eval on molab

        Pulls a restored fp16 checkpoint from HuggingFace, serves it with vLLM
        using the **TA's exact generation path and sampling config**, evaluates
        it on math, and pushes the results back to the sandbox git repo.

        Pairs with `kaggle_sandbox.ipynb`, which produced the checkpoint.

        **To reuse:** edit the CONFIG cell, then Run All.
        To switch models mid-session: `stop_server()` → edit CONFIG → re-run
        from *Download*.

        Fixes baked in (all previously hit):

        - `VLLM_USE_FLASHINFER_SAMPLER=0` — molab has no `nvcc`
        - `start_new_session=True` — server survives cell interrupts
        - live log tailing in the readiness poll
        - **model downloaded to a local dir** — vLLM cannot read a repo
          subdirectory, and this also lets us verify before serving
        - `DTYPE="float16"` — avoids the silent fp16→bf16 upcast
        - `chat_template.jinja` presence check — without it `enable_thinking`
          silently does nothing

        **Attach the GPU via the notebook specs button before running.**
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md("## 0. CONFIG — the only cell you normally edit")
    return


@app.cell
def _():
    # ------------------------------------------------------------------- RUN
    RUN_ID = "w01_c40_int4"

    # HF repo holding the restored fp16 model (files at the repo ROOT).
    # Use "Qwen/Qwen3.5-4B" to measure the baseline.
    HF_EVAL_REPO = "your-hf-user/qwen35-w01_c40_int4-fp16"

    # ---------------------------------------------------------------- GIT
    GITHUB_REPO = "your-github-user/cs6013-sandbox"
    GIT_BRANCH = "main"
    PUSH_RESULTS = True

    # ------------------------------------------------------------ BENCHMARK
    # smoke   10 offline problems, no download   -> always start here
    # gsm8k   openai/gsm8k, free-form numeric
    # math500 HuggingFaceH4/MATH-500 (most discriminative)
    # gpqa    Idavidrein/gpqa (gated) -- TA parity check, MCQ grading
    BENCHMARK = "smoke"
    LIMIT = 5

    # ------------------------------------------------------------- RUNTIME
    # "float16" matches what decompress.py writes. Use "auto" only for the
    # untouched base model.
    DTYPE = "float16"
    MAX_MODEL_LEN = 16000
    MAX_CONCURRENCY = 8

    # ------------------------------------ SAMPLING (TA graded config, fixed)
    # cs6013-main/evaluation/configs/config.yaml. Matching these is the point.
    # The TA pair is 32000/33000; 8000/16000 iterates much faster. Raise both
    # for a final pre-submission run. MAX_NEW_TOKENS must stay < MAX_MODEL_LEN.
    MAX_NEW_TOKENS = 8000
    TEMPERATURE = 1.0
    TOP_P = 0.95
    TOP_K = 20
    MIN_P = 0.0
    PRESENCE_PENALTY = 1.5
    REPETITION_PENALTY = 1.0
    ENABLE_THINKING = True

    RUN_TAG = f"{RUN_ID}_{BENCHMARK}"

    print(f"run        {RUN_ID}")
    print(f"model      {HF_EVAL_REPO}")
    print(f"benchmark  {BENCHMARK} (limit={LIMIT})")
    print(f"tokens     {MAX_NEW_TOKENS} new / {MAX_MODEL_LEN} ctx")
    print(f"tag        {RUN_TAG}")
    return (
        BENCHMARK,
        DTYPE,
        ENABLE_THINKING,
        GITHUB_REPO,
        GIT_BRANCH,
        HF_EVAL_REPO,
        LIMIT,
        MAX_CONCURRENCY,
        MAX_MODEL_LEN,
        MAX_NEW_TOKENS,
        MIN_P,
        PRESENCE_PENALTY,
        PUSH_RESULTS,
        REPETITION_PENALTY,
        RUN_ID,
        RUN_TAG,
        TEMPERATURE,
        TOP_K,
        TOP_P,
    )


@app.cell
def _(mo):
    mo.md("## 1. Dependencies")
    return


@app.cell
def _():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "vllm", "datasets", "openai", "pyyaml", "huggingface_hub"],
        check=True,
    )

    _v = subprocess.run(
        [sys.executable, "-c",
         "import vllm, torch; print(vllm.__version__, torch.__version__)"],
        capture_output=True, text=True)
    print("vllm / torch:", _v.stdout.strip() or _v.stderr[-400:])

    import shutil
    _t, _u, _f = shutil.disk_usage("/")
    print(f"disk: {_f/1024**3:.1f} GiB free of {_t/1024**3:.1f} GiB")
    return shutil, subprocess, sys


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Tokens

        **HF token** — needed for a private model repo (yours are private) and
        for gated datasets (GPQA).
        **GitHub PAT** — only if `PUSH_RESULTS` is on.
        """
    )
    return


@app.cell
def _(mo):
    hf_token_input = mo.ui.text(label="HF_TOKEN", kind="password")
    gh_pat_input = mo.ui.text(label="GITHUB_PAT (optional)", kind="password")
    mo.vstack([hf_token_input, gh_pat_input])
    return gh_pat_input, hf_token_input


@app.cell
def _(gh_pat_input, hf_token_input):
    import os

    HF_TOKEN = hf_token_input.value.strip()
    GH_PAT = gh_pat_input.value.strip()

    if HF_TOKEN:
        from huggingface_hub import login
        os.environ["HF_TOKEN"] = HF_TOKEN
        login(token=HF_TOKEN)
        print("HF: logged in")
    else:
        print("HF: no token (public repos only)")
    print("GitHub PAT:", "set" if GH_PAT else "not set")
    return GH_PAT, HF_TOKEN, os


@app.cell
def _(mo):
    mo.md("## 3. Workspace + sandbox repo")
    return


@app.cell
def _(GH_PAT, GITHUB_REPO, GIT_BRANCH, subprocess):
    import pathlib

    WORKDIR = pathlib.Path("/root/cs6013_eval")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    SANDBOX = pathlib.Path("/root/sandbox")

    def sh(cmd, cwd=None, check=True, quiet=False):
        r = subprocess.run([str(c) for c in cmd], cwd=cwd,
                           capture_output=True, text=True)
        shown = " ".join(str(c).replace(GH_PAT, "***") if GH_PAT else str(c) for c in cmd)
        if not quiet:
            print(f"$ {shown}")
            if r.stdout.strip():
                print(r.stdout[-2500:])
        if r.returncode != 0:
            msg = r.stderr[-2500:]
            print("STDERR:", msg.replace(GH_PAT, "***") if GH_PAT else msg)
            if check:
                raise RuntimeError(f"failed: {shown}")
        return r

    if GH_PAT:
        _remote = f"https://{GH_PAT}@github.com/{GITHUB_REPO}.git"
        if SANDBOX.exists():
            sh(["git", "fetch", "origin"], cwd=SANDBOX)
            sh(["git", "reset", "--hard", f"origin/{GIT_BRANCH}"], cwd=SANDBOX)
        else:
            sh(["git", "clone", "--branch", GIT_BRANCH, _remote, str(SANDBOX)])
    else:
        print("no PAT — skipping sandbox clone; results stay local")

    print("WORKDIR:", WORKDIR)
    return SANDBOX, WORKDIR, pathlib, sh


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Download the model to a LOCAL directory

        vLLM cannot read a repo subdirectory — `config.json` must be at the
        root of whatever you point it at. Downloading first also lets us verify
        the checkpoint before spending ~3 minutes on engine startup.
        """
    )
    return


@app.cell
def _(HF_EVAL_REPO, HF_TOKEN, pathlib):
    from huggingface_hub import snapshot_download

    MODEL_DIR = pathlib.Path("/root/model") / HF_EVAL_REPO.replace("/", "__")
    _p = snapshot_download(
        HF_EVAL_REPO,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN or None,
    )
    MODEL_DIR = pathlib.Path(_p)

    # If someone uploaded into a subfolder, find the real root.
    if not (MODEL_DIR / "config.json").is_file():
        _hits = [c.parent for c in MODEL_DIR.rglob("config.json")
                 if list(c.parent.glob("*.safetensors"))]
        if _hits:
            MODEL_DIR = _hits[0]
            print(f"config.json was nested; using {MODEL_DIR}")
        else:
            raise SystemExit(f"No config.json with weights under {MODEL_DIR}")

    print("MODEL_DIR:", MODEL_DIR)
    for _f in sorted(MODEL_DIR.iterdir()):
        if _f.is_file():
            print(f"  {_f.stat().st_size:>15,}  {_f.name}")
    return MODEL_DIR, snapshot_download


@app.cell
def _(mo):
    mo.md("## 5. Verify before serving")
    return


@app.cell
def _(MODEL_DIR):
    import json as _json

    _req = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for _f in _req:
        print(f"  {'OK     ' if (MODEL_DIR/_f).is_file() else 'MISSING'} {_f}")

    if not (MODEL_DIR / "chat_template.jinja").is_file():
        print("\n  WARNING: no chat_template.jinja. enable_thinking will silently")
        print("  do nothing and the model will stop emitting <think> blocks —")
        print("  your numbers will not match the graded config.")

    _w = sorted(MODEL_DIR.glob("*.safetensors"))
    _tot = sum(f.stat().st_size for f in _w)
    print(f"\n{len(_w)} shard(s), {_tot:,} bytes ({_tot/1024**3:.3f} GiB)")
    print("base fp16 reference: 9,319,737,856")

    _cfg = _json.loads((MODEL_DIR / "config.json").read_text())
    print("architectures:", _cfg.get("architectures"))
    CFG_DTYPE = _cfg.get("dtype") or _cfg.get("text_config", {}).get("dtype")
    print("config dtype :", CFG_DTYPE)

    from safetensors import safe_open
    with safe_open(str(_w[0]), framework="pt") as _sf:
        _k = next(iter(_sf.keys()))
        REAL_DTYPE = _sf.get_slice(_k).get_dtype()
    print("tensor dtype :", REAL_DTYPE)

    if REAL_DTYPE in ("F16", "float16") and str(CFG_DTYPE).lower() == "bfloat16":
        print("\n  MISMATCH: fp16 weights but config says bfloat16.")
        print("  Keep DTYPE='float16' or vLLM will upcast and lose precision.")
    return CFG_DTYPE, REAL_DTYPE, safe_open


@app.cell
def _(mo):
    mo.md("## 6. Write the eval library")
    return


@app.cell
def _(WORKDIR):
    # Generation path is VERBATIM from cs6013-main/evaluation/common.py.
    # ADDED: math datasets + free-form grading (the TA matcher is A-D only and
    # returns garbage on a value like \frac{3}{4}).
    eval_lib_py = r'''"""TA generation path + math datasets/grading."""

from __future__ import annotations

import asyncio, json, random, re, traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI

_VALID_CHOICES = ("A", "B", "C", "D")


@dataclass
class EvalExample:
    example_id: str
    prompt: str
    gold: str
    metadata: dict


@dataclass
class EvalPrediction:
    example_id: str
    gold: str
    prediction: str | None
    prediction_normalized: str | None
    prediction_parsed: list
    gold_normalized: str
    gold_parsed: list
    correct: bool
    response: str
    metadata: dict


# ---------------------- VERBATIM TA: prompt + extraction --------------------

def build_math_prompt(problem: str, math_instruction: str) -> str:
    return f"{math_instruction}\n\nProblem:\n{problem.strip()}"


def extract_boxed_answer(text: str) -> str | None:
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
                return text[start + 1:j].strip()
    return None


# ---------------------- VERBATIM TA: MCQ grading ----------------------------

def normalize_answer(answer: str) -> str:
    text = answer.strip().strip("$").strip()
    text = re.sub(r"[\\{}()]", "", text).strip()
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    for ch in text.upper():
        if ch in _VALID_CHOICES:
            return ch
    return ""


def answers_match(prediction, gold):
    gn = normalize_answer(gold)
    gp = [gn] if gn in _VALID_CHOICES else []
    if prediction is None:
        return False, None, [], gn, gp
    pn = normalize_answer(prediction)
    pp = [pn] if pn in _VALID_CHOICES else []
    return (pn in _VALID_CHOICES and pn == gn), pn or None, pp, gn, gp


# ---------------------- ADDED: free-form math grading -----------------------

_STRIP = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "\\$",
          "\\text{ }", "^\\circ", "^{\\circ}", "\\%", "%", "\\quad", "\\qquad")


def normalize_math_answer(ans):
    if ans is None:
        return None
    s = ans.strip()
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", s)
    for t in _STRIP:
        s = s.replace(t, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\1/\2", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
    s = s.rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.lstrip("+")
    if s.startswith("(") and s.endswith(")") and s.count("(") == 1:
        s = s[1:-1]
    return s.lower()


def _as_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", s)
    if m:
        try:
            d = float(m.group(2))
            return float(m.group(1)) / d if d else None
        except ValueError:
            return None
    return None


def math_answers_match(prediction, gold):
    gn = normalize_math_answer(gold) or ""
    gp = [gn] if gn else []
    if prediction is None:
        return False, None, [], gn, gp
    pn = normalize_math_answer(prediction)
    pp = [pn] if pn else []
    correct = False
    if pn and gn:
        if pn == gn:
            correct = True
        else:
            pf, gf = _as_float(pn), _as_float(gn)
            if pf is not None and gf is not None:
                correct = abs(pf - gf) <= 1e-6 * max(1.0, abs(gf))
    return correct, pn or None, pp, gn, gp


def repetition_score(text, n=12):
    w = text.split()
    if len(w) < n * 2:
        return 0.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - (len(set(g)) / len(g))


# ---------------------- VERBATIM TA: generation -----------------------------

async def _generate_one_response(client, semaphore, index, total, prompt, *,
                                 model_name, max_new_tokens, temperature, top_p,
                                 top_k, min_p, presence_penalty,
                                 repetition_penalty, enable_thinking):
    async with semaphore:
        print(flush=True)
        print("=" * 80, flush=True)
        print(f"Generating response {index}/{total}", flush=True)
        print("=" * 80, flush=True)
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                seed=42,
                extra_body={
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
        except Exception:
            print(f"FAILED to generate response {index}/{total}", flush=True)
            traceback.print_exc()
            raise
        text = (response.choices[0].message.content or "").strip()
        print(flush=True)
        print(f"Model Response {index}/{total}:", flush=True)
        print(text, flush=True)
        print(flush=True)
        return text


async def generate_responses(client, model_name, prompts, *, max_new_tokens,
                             temperature, top_p, top_k, min_p, presence_penalty,
                             repetition_penalty, enable_thinking,
                             max_concurrency=8):
    total = len(prompts)
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [
        _generate_one_response(
            client, sem, i, total, p, model_name=model_name,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, min_p=min_p, presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty, enable_thinking=enable_thinking)
        for i, p in enumerate(prompts, start=1)
    ]
    return list(await asyncio.gather(*tasks))


# ---------------------- Datasets --------------------------------------------

FREEFORM_INSTRUCTION = (
    "Solve the following math problem. Reason carefully, but keep the "
    "reasoning concise. After reasoning, output the final answer on its own "
    "last line using exactly this format: \\boxed{ANSWER}. Put only the final "
    "value inside the box, with no units and no explanation. Do not write "
    "anything after that line."
)

MCQ_INSTRUCTION = (
    "Answer the following multiple-choice question. Reason carefully, but keep "
    "the reasoning concise. After reasoning, output the final answer on its own "
    "last line using exactly this format: \\boxed{X} where X is one of "
    "(A, B, C, D). Do not write anything after that line. Example last line: "
    "\\boxed{A}."
)

_SMOKE = [
    ("What is 17 * 24?", "408"),
    ("If 3x + 7 = 25, what is x?", "6"),
    ("What is the sum of the first 20 positive integers?", "210"),
    ("A shirt costs $40 and is discounted by 25%. What is the sale price in dollars?", "30"),
    ("What is the greatest common divisor of 84 and 132?", "12"),
    ("Compute the derivative of f(x) = 3x^2 + 5x - 2 at x = 2.", "17"),
    ("How many distinct arrangements are there of the letters in the word LEVEL?", "30"),
    ("What is the remainder when 2^10 is divided by 7?", "2"),
    ("A right triangle has legs of length 9 and 12. What is the length of the hypotenuse?", "15"),
    ("Solve for x: x^2 - 5x + 6 = 0. Give the larger root.", "3"),
]


def load_smoke(instr, dataset_path=None, dataset_config=None, limit=None, seed=42):
    items = _SMOKE if limit is None else _SMOKE[:limit]
    return [EvalExample(str(i), build_math_prompt(q, instr), a,
                        {"problem_type": "smoke"})
            for i, (q, a) in enumerate(items)]


def _select(ds, limit):
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"Using first {len(ds)} examples.", flush=True)
    return ds


def load_gsm8k(instr, dataset_path="openai/gsm8k", dataset_config="main",
               limit=None, seed=42):
    from datasets import load_dataset
    ds = load_dataset(dataset_path or "openai/gsm8k", dataset_config or "main",
                      split="test")
    print(f"Dataset size: {len(ds)}", flush=True)
    ds = _select(ds, limit)
    out = []
    for i, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(EvalExample(str(i), build_math_prompt(row["question"], instr),
                               gold, {"problem_type": "gsm8k"}))
    return out


def load_math500(instr, dataset_path="HuggingFaceH4/MATH-500",
                 dataset_config=None, limit=None, seed=42):
    from datasets import load_dataset
    ds = load_dataset(dataset_path or "HuggingFaceH4/MATH-500", split="test")
    print(f"Dataset size: {len(ds)}", flush=True)
    ds = _select(ds, limit)
    out = []
    for i, row in enumerate(ds):
        gold = str(row.get("answer") or "").strip()
        if not gold and row.get("solution"):
            gold = (extract_boxed_answer(row["solution"]) or "").strip()
        if not gold:
            continue
        out.append(EvalExample(str(i), build_math_prompt(row["problem"], instr),
                               gold, {"problem_type": row.get("subject"),
                                      "level": row.get("level")}))
    return out


def load_gpqa(instr, dataset_path="Idavidrein/gpqa",
              dataset_config="gpqa_diamond", limit=None, seed=42):
    from datasets import load_dataset
    ds = (load_dataset(dataset_path, dataset_config, split="train")
          if dataset_config else load_dataset(dataset_path, split="train"))
    print(f"Dataset size: {len(ds)}", flush=True)
    ds = _select(ds, limit)
    out = []
    for ri, row in enumerate(ds):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        rng = random.Random(f"{seed}:{ri}")
        idxs = list(range(4))
        rng.shuffle(idxs)
        shuffled = [choices[i] for i in idxs]
        gold = "ABCD"[idxs.index(0)]
        opts = "\n".join(f"{L}) {c.strip()}" for L, c in zip("ABCD", shuffled))
        out.append(EvalExample(str(ri),
                               build_math_prompt(f"{row['Question'].strip()}\n\n{opts}", instr),
                               gold, {"problem_type": row.get("Subdomain")}))
    return out


BUILDERS = {"smoke": load_smoke, "gsm8k": load_gsm8k,
            "math500": load_math500, "gpqa": load_gpqa}
GRADING_MODE = {"smoke": "freeform", "gsm8k": "freeform",
                "math500": "freeform", "gpqa": "mcq"}
DEFAULT_INSTRUCTION = {"smoke": FREEFORM_INSTRUCTION, "gsm8k": FREEFORM_INSTRUCTION,
                       "math500": FREEFORM_INSTRUCTION, "gpqa": MCQ_INSTRUCTION}


# ---------------------- Results ---------------------------------------------

def summarize_predictions(predictions):
    n = len(predictions)
    nc = sum(1 for p in predictions if p.correct)
    npar = sum(1 for p in predictions if p.prediction is not None)
    reps = [p.metadata.get("repetition", 0.0) for p in predictions]
    return {
        "num_examples": n,
        "num_correct": nc,
        "accuracy": (nc / n if n else 0.0),
        "num_parsed": npar,
        "parse_rate": (npar / n if n else 0.0),
        "mean_repetition": round(sum(reps) / n, 4) if n else 0.0,
        "degenerate_rate": round(sum(r > 0.5 for r in reps) / n, 4) if n else 0.0,
        "mean_response_chars": round(sum(len(p.response) for p in predictions) / n, 1) if n else 0.0,
    }


def save_results(output_path, *, benchmark, model_name, summary, predictions, extra=None):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"benchmark": benchmark, "model_name": model_name, "summary": summary,
         "extra": extra or {}, "predictions": [asdict(p) for p in predictions]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote results to {path}", flush=True)
'''
    (WORKDIR / "eval_lib.py").write_text(eval_lib_py, encoding="utf-8")
    print("wrote eval_lib.py")
    return (eval_lib_py,)


@app.cell
def _(WORKDIR):
    run_eval_py = r'''from __future__ import annotations

import argparse, asyncio, sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_lib import (BUILDERS, GRADING_MODE, EvalPrediction, answers_match,
                      extract_boxed_answer, generate_responses,
                      math_answers_match, repetition_score, save_results,
                      summarize_predictions)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text())

    bench = cfg["benchmark"]
    mode = cfg.get("grading_mode") or GRADING_MODE[bench]

    print("=" * 80, flush=True)
    print("CS6013 math eval (TA generation path)", flush=True)
    print("=" * 80, flush=True)
    print(f"Experiment: {cfg['exp_name']}", flush=True)
    print(f"Model:      {cfg['model_name']}", flush=True)
    print(f"Benchmark:  {bench} (grading: {mode}) limit={cfg.get('limit')}", flush=True)
    for k in ("max_new_tokens", "temperature", "top_p", "top_k", "min_p",
              "presence_penalty", "repetition_penalty", "enable_thinking"):
        print(f"  {k:<20} {cfg.get(k)}", flush=True)
    print(flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=cfg["vllm_base_url"],
                         api_key=cfg.get("api_key", "EMPTY"),
                         timeout=cfg["request_timeout"])
    served = [m.id for m in (await client.models.list()).data]
    print("Serving:", served, flush=True)
    if cfg["model_name"] not in served:
        raise SystemExit(f"model_name {cfg['model_name']!r} not served. Got {served}")

    examples = BUILDERS[bench](cfg["math_instruction"],
                               dataset_path=cfg.get("dataset_path"),
                               dataset_config=cfg.get("dataset_config"),
                               limit=cfg.get("limit"), seed=cfg.get("seed", 42))
    print(f"Loaded {len(examples)} examples.\n", flush=True)

    responses = await generate_responses(
        client, cfg["model_name"], [e.prompt for e in examples],
        max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"],
        top_p=cfg["top_p"], top_k=cfg["top_k"], min_p=cfg.get("min_p", 0.0),
        presence_penalty=cfg.get("presence_penalty", 1.5),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
        enable_thinking=cfg["enable_thinking"],
        max_concurrency=cfg.get("max_concurrency", 8))

    matcher = answers_match if mode == "mcq" else math_answers_match
    preds = []
    for ex, resp in zip(examples, responses):
        pred = extract_boxed_answer(resp)
        correct, pn, pp, gn, gp = matcher(prediction=pred, gold=ex.gold)
        rep = repetition_score(resp)
        print("=" * 80, flush=True)
        print(f"Example {ex.example_id}", flush=True)
        print(f"Prediction: {pred!r}   Gold: {ex.gold!r}", flush=True)
        print(f"Normalized: {pn!r} vs {gn!r}", flush=True)
        print(f"Correct: {correct}   chars={len(resp)} rep={rep:.3f}", flush=True)
        print("=" * 80, flush=True)
        meta = dict(ex.metadata)
        meta["repetition"] = round(rep, 4)
        meta["response_chars"] = len(resp)
        preds.append(EvalPrediction(ex.example_id, ex.gold, pred, pn, pp, gn, gp,
                                    correct, resp, meta))

    s = summarize_predictions(preds)
    print(flush=True)
    print("=" * 80, flush=True)
    print("RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"Accuracy:   {s['accuracy']:.4f} ({s['num_correct']}/{s['num_examples']})", flush=True)
    print(f"Parse rate: {s['parse_rate']:.4f}", flush=True)
    print(f"Repetition: {s['mean_repetition']:.4f} (degenerate {s['degenerate_rate']:.2%})", flush=True)
    print(f"Mean chars: {s['mean_response_chars']:.0f}", flush=True)
    print("=" * 80, flush=True)

    save_results(cfg["output"], benchmark=bench, model_name=cfg["model_name"],
                 summary=s, predictions=preds, extra=cfg)


if __name__ == "__main__":
    asyncio.run(main())
'''
    (WORKDIR / "run_eval.py").write_text(run_eval_py, encoding="utf-8")
    print("wrote run_eval.py")
    return (run_eval_py,)


@app.cell
def _(mo):
    mo.md("## 7. Generate config.yaml")
    return


@app.cell
def _(
    BENCHMARK,
    ENABLE_THINKING,
    HF_EVAL_REPO,
    LIMIT,
    MAX_CONCURRENCY,
    MAX_NEW_TOKENS,
    MIN_P,
    PRESENCE_PENALTY,
    REPETITION_PENALTY,
    RUN_TAG,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    WORKDIR,
    sys,
):
    import yaml

    sys.path.insert(0, str(WORKDIR))
    from eval_lib import DEFAULT_INSTRUCTION

    _DS = {"smoke": (None, None), "gsm8k": ("openai/gsm8k", "main"),
           "math500": ("HuggingFaceH4/MATH-500", None),
           "gpqa": ("Idavidrein/gpqa", "gpqa_diamond")}
    _path, _dcfg = _DS[BENCHMARK]

    SERVED_NAME = HF_EVAL_REPO

    cfg_dict = {
        "exp_name": RUN_TAG,
        "model_name": SERVED_NAME,
        "vllm_base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "request_timeout": 7200,
        "max_concurrency": MAX_CONCURRENCY,
        "benchmark": BENCHMARK,
        "dataset_path": _path,
        "dataset_config": _dcfg,
        "limit": LIMIT,
        "seed": 42,
        "math_instruction": DEFAULT_INSTRUCTION[BENCHMARK],
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "presence_penalty": PRESENCE_PENALTY,
        "repetition_penalty": REPETITION_PENALTY,
        "enable_thinking": ENABLE_THINKING,
        "output": f"results/{RUN_TAG}.json",
    }
    (WORKDIR / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    print(yaml.safe_dump(cfg_dict, sort_keys=False))
    return DEFAULT_INSTRUCTION, SERVED_NAME, cfg_dict, yaml


@app.cell
def _(mo):
    mo.md(
        """
        ## 8. Launch vLLM

        First start on a fresh model takes **~3 min** (load + compile + CUDA
        graph capture). Cached models are much faster.
        """
    )
    return


@app.cell
def _(DTYPE, MAX_MODEL_LEN, MODEL_DIR, SERVED_NAME, WORKDIR, os, subprocess, sys):
    _env = os.environ.copy()
    _env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"   # molab has no nvcc

    vllm_log_f = open(f"{WORKDIR}/vllm.log", "w")
    vllm_proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", str(MODEL_DIR),
         "--served-model-name", SERVED_NAME,
         "--tensor-parallel-size", "1",
         "--dtype", DTYPE,
         "--max-model-len", str(MAX_MODEL_LEN),
         "--port", "8000"],
        stdout=vllm_log_f, stderr=subprocess.STDOUT,
        env=_env, start_new_session=True,      # survives cell interrupts
    )
    print(f"vLLM starting pid={vllm_proc.pid}")
    print(f"model     {MODEL_DIR}")
    print(f"served as {SERVED_NAME}   dtype={DTYPE}")
    return vllm_log_f, vllm_proc


@app.cell
def _(mo):
    mo.md("## 9. Wait for readiness (streams the log live)")
    return


@app.cell
def _():
    import time
    import urllib.request

    def wait_for_server(url, log_path, timeout_s=1800, interval_s=10):
        start = time.time()
        pos = 0
        while time.time() - start < timeout_s:
            try:
                with open(log_path) as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
                if new:
                    print(new, end="", flush=True)
            except FileNotFoundError:
                pass
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        print(f"\nServer ready after {int(time.time()-start)}s")
                        return True
            except Exception:
                pass
            time.sleep(interval_s)
        raise TimeoutError("vLLM not ready — check vllm.log")

    return time, urllib, wait_for_server


@app.cell
def _(WORKDIR, vllm_proc, wait_for_server):
    assert vllm_proc.poll() is None, "vLLM exited early — check vllm.log"
    wait_for_server("http://localhost:8000/v1/models", f"{WORKDIR}/vllm.log")
    return


@app.cell
def _(mo):
    mo.md("## 10. Run the evaluation")
    return


@app.cell
def _(WORKDIR, subprocess, sys):
    eval_result = subprocess.run(
        [sys.executable, "run_eval.py", "--config", "config.yaml"],
        cwd=str(WORKDIR), capture_output=True, text=True)
    print(eval_result.stdout[-15000:])
    if eval_result.returncode != 0:
        print("STDERR:", eval_result.stderr[-5000:])
    return (eval_result,)


@app.cell
def _(mo):
    mo.md("## 11. Results")
    return


@app.cell
def _(RUN_TAG, WORKDIR):
    import json

    results = json.loads((WORKDIR / "results" / f"{RUN_TAG}.json").read_text())
    print(f"model     {results['model_name']}")
    print(f"benchmark {results['benchmark']}")
    print("-" * 52)
    for _k, _v in results["summary"].items():
        print(f"{_k:<22}{_v}")
    return json, results


@app.cell
def _(mo):
    mo.md(
        """
        ## 12. Push results back to git

        **molab persists almost nothing** — do this in the same session.
        """
    )
    return


@app.cell
def _(
    BENCHMARK,
    GH_PAT,
    GIT_BRANCH,
    HF_EVAL_REPO,
    PUSH_RESULTS,
    RUN_ID,
    RUN_TAG,
    SANDBOX,
    WORKDIR,
    cfg_dict,
    json,
    results,
    sh,
    shutil,
):
    if PUSH_RESULTS and GH_PAT:
        _rd = SANDBOX / "results" / RUN_ID
        _rd.mkdir(parents=True, exist_ok=True)

        # Full predictions can be large; keep them, but also write a compact
        # summary that is easy to diff and read in the repo.
        shutil.copy2(WORKDIR / "results" / f"{RUN_TAG}.json", _rd / f"eval_{RUN_TAG}.json")
        (_rd / f"summary_{RUN_TAG}.json").write_text(json.dumps({
            "run_id": RUN_ID,
            "benchmark": BENCHMARK,
            "model": HF_EVAL_REPO,
            "summary": results["summary"],
            "sampling": {k: cfg_dict[k] for k in
                         ("max_new_tokens", "temperature", "top_p", "top_k",
                          "min_p", "presence_penalty", "repetition_penalty",
                          "enable_thinking", "limit")},
        }, indent=2))

        sh(["git", "config", "user.email", "molab@sandbox.local"], cwd=SANDBOX)
        sh(["git", "config", "user.name", "molab-sandbox"], cwd=SANDBOX)
        sh(["git", "pull", "--rebase", "origin", GIT_BRANCH], cwd=SANDBOX, check=False)
        sh(["git", "add", "results"], cwd=SANDBOX)
        _st = sh(["git", "status", "--porcelain"], cwd=SANDBOX, quiet=True)
        if _st.stdout.strip():
            _acc = results["summary"]["accuracy"]
            sh(["git", "commit", "-m",
                f"results({RUN_ID}): {BENCHMARK} acc={_acc:.4f}"], cwd=SANDBOX)
            sh(["git", "push", "origin", GIT_BRANCH], cwd=SANDBOX)
            print("pushed")
        else:
            print("nothing to commit")
    else:
        print("PUSH_RESULTS off or no PAT — results stay in", WORKDIR / "results")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 13. Stop the server

        Run before switching models: `stop_server()` → edit CONFIG → re-run
        from section 4.
        """
    )
    return


@app.cell
def _(vllm_log_f, vllm_proc):
    def stop_server():
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=60)
        except Exception:
            vllm_proc.kill()
        vllm_log_f.close()
        print("vLLM stopped.")

    # stop_server()
    return (stop_server,)


if __name__ == "__main__":
    app.run()
