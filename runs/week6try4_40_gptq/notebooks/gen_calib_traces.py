import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # CS6013 — Calibration trace generation

        Runs the **bf16 base model** over a curated math prompt set with the
        **eval's exact sampling parameters**, then uploads the traces to a
        Hugging Face dataset repo.

        The traces become the GPTQ calibration corpus: `H = 2·E[xxᵀ]` is estimated
        from whatever tokens we push through the model, and at eval time almost
        every token is one the model generated itself inside `<think>`. Human-written
        dataset solutions are a different distribution.

        **Nothing is filtered on correctness.** The compressed model will get plenty
        of problems wrong too, and filtering on correctness would systematically keep
        the *easier* problems. Correctness, truncation and degeneracy are all
        *recorded* per row so later experiments can slice without regenerating.
        """
    )
    return


@app.cell
def _():
    # ----------------------------------------------------------------------
    # CONFIG
    # ----------------------------------------------------------------------
    MODEL_ID = "Qwen/Qwen3.5-4B"

    # Where the traces get uploaded. Change to your HF username.
    HF_REPO_ID = "grey-cat/cs6013-calib-traces-v1"
    HF_PRIVATE = False

    # Per-source prompt counts. AIME sets are taken whole (30 each).
    # GPQA was dropped: its domains are Biology/Chemistry/Physics, there is no
    # math subset, and it is gated. Bump the numbers below if you want to
    # reclaim those 50 prompts.
    N_MATH500 = 100
    N_COMPMATH = 100
    N_MMLU_PRO_MATH = 50

    # vLLM serving parameters, mirroring evalkit/notebooks/run_eval.py -- the
    # configuration already known to work on molab. 1000 + 32000 is the
    # graders' own max-model-len. Strings because they are passed as CLI args.
    MAX_MODEL_LEN = 33000
    GPU_MEM_UTIL = "0.85"    # evalkit uses 0.35 only because it juggles 2 models
    MAX_NUM_SEQS = "50"
    MAX_CONCURRENCY = 30     # in-flight HTTP requests, same as the graders
    SERVED_MODEL_NAME = "qwen-3.5-4b"
    SEED = 0
    return (
        GPU_MEM_UTIL,
        HF_PRIVATE,
        HF_REPO_ID,
        MAX_CONCURRENCY,
        MAX_MODEL_LEN,
        MAX_NUM_SEQS,
        MODEL_ID,
        N_COMPMATH,
        N_MATH500,
        N_MMLU_PRO_MATH,
        SEED,
        SERVED_MODEL_NAME,
    )


@app.cell
def _():
    # Eval-exact sampling. presence_penalty=1.5 is unusually high and strongly
    # reshapes the token distribution — calibration MUST match it or the traces
    # will not look like eval-time output.
    SAMPLING_KW = dict(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=32000,
    )

    # Copied byte for byte from configs/eval_config.yaml — including the trailing
    # space at the end of lines 1-4. Do NOT let an editor strip them; they change
    # tokenisation.
    MATH_INSTRUCTION = (
        "Answer the following mathematics problem. \n"
        "Please reason step by step, but keep the reasoning concise. \n"
        "After reasoning, output the final answer on its own last line using "
        "exactly this format: \\boxed{X}, \n"
        "where X is the final mathematical answer. Generate \\boxed{X} exactly once. \n"
        "Do not write anything after that line.\n"
    )
    return MATH_INSTRUCTION, SAMPLING_KW


@app.cell
def _():
    # ----------------------------------------------------------------------
    # molab has NO CUDA toolkit (no nvcc), so anything that JIT-compiles CUDA at
    # runtime dies. vLLM's sampler routes through FlashInfer whenever top_k is
    # set, and FlashInfer builds its kernel on first use:
    #
    #   flashinfer/jit/cpp_ext.py get_cuda_path()
    #   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
    #
    # Setting this to "0" uses vLLM's native PyTorch top-k/top-p path instead.
    # Identical sampling, no compiler. (Same fix as evalkit/notebooks/run_eval.py.)
    #
    # This MUST happen before vLLM is imported, hence a separate cell whose
    # output the vLLM cell depends on -- marimo runs cells in dependency order,
    # not file order.
    import os as _os

    _os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Uncomment to silence the (harmless) FlashInfer GDN warmup tracebacks:
    # _os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    ENV_READY = True
    print("VLLM_USE_FLASHINFER_SAMPLER=0  (molab has no nvcc)")
    return (ENV_READY,)


@app.cell
def _(ENV_READY):
    # ----------------------------------------------------------------------
    # Dependencies. Installed ONE AT A TIME: a single pip invocation aborts the
    # whole command if any package fails to build, which would silently take out
    # the packages listed beside it.
    # ----------------------------------------------------------------------
    import subprocess
    import sys

    assert ENV_READY

    def pip_install(*pkgs, optional=False):
        results = {}
        for p in pkgs:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", p],
                capture_output=True, text=True,
            )
            ok = r.returncode == 0
            results[p] = ok
            print(f"  {p:<30} {'ok' if ok else ('skipped (optional)' if optional else 'FAILED')}")
            if not ok and not optional:
                print("   " + r.stderr.strip()[-700:])
        return results

    print("required:")
    pip_install("vllm", "datasets", "huggingface_hub", "transformers", "openai")

    # ------------------------------------------------------------------
    # THE cutlass FIX -- this is what killed the previous run.
    #
    # molab's venv is created with --system-site-packages. vLLM and FlashInfer
    # live in the venv (/tmp/uv-venv/...), but `nvidia_cutlass_dsl` resolves to
    # the OLDER copy in the system tree (/usr/local/lib/python3.13/...).
    # FlashInfer 0.7 declares `nvidia-cutlass-dsl[cu13]>=4.7.0a0`; the system
    # copy predates that, so its Gated-DeltaNet prefill kernel explodes with
    #
    #   AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'CopyR2GOp'
    #
    # At warmup vLLM catches it and prints a wall of tracebacks. On the FIRST
    # REAL REQUEST nothing catches it and EngineCore dies -- which is exactly
    # what happened. Installing a matching cutlass INTO the venv puts it ahead
    # of the system copy on sys.path and the kernel compiles.
    # ------------------------------------------------------------------
    print("\ncutlass DSL (fixes the GDN prefill kernel):")
    pip_install("nvidia-cutlass-dsl[cu13]>=4.7.0")

    # NOT needed for this notebook: vLLM ships its own Gated-DeltaNet and
    # causal-conv1d kernels. These only matter later for the transformers-based
    # compress.py run.
    #   flash-linear-attention : pure Triton, installs anywhere
    #   causal-conv1d          : compiles CUDA C++, needs nvcc. molab has no nvcc,
    #                            so it fails with "bare_metal_version is not
    #                            defined". That is expected and harmless here.
    print("\noptional (transformers path only — failures here do not matter):")
    pip_install("flash-linear-attention", optional=True)
    return pip_install, subprocess, sys


@app.cell
def _(pip_install):
    # Sanity check before spending GPU time.
    import torch

    print(f"  torch            : {torch.__version__}")
    print(f"  cuda available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device           : {torch.cuda.get_device_name(0)}")
        print(f"  total VRAM       : {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    try:
        import vllm
        print(f"  vllm             : {vllm.__version__}")
    except Exception as exc:
        print(f"  vllm             : IMPORT FAILED -> {exc}")

    # The check that matters: does the cutlass now on sys.path have the symbol
    # FlashInfer's GDN prefill kernel needs? If this says MISSING, the run will
    # die on the first request exactly as it did before -- restart the notebook
    # kernel so the freshly installed cutlass is picked up, and re-run.
    try:
        import cutlass
        from cutlass import cute
        _has = hasattr(cute.nvgpu, "CopyR2GOp")
        print(f"  cutlass-dsl      : {getattr(cutlass, '__version__', '?')}  "
              f"at {cutlass.__file__}")
        print(f"  cute.nvgpu.CopyR2GOp : {'present -- GDN kernel OK' if _has else 'MISSING -- RESTART THE KERNEL'}")
    except Exception as exc:
        print(f"  cutlass-dsl      : import failed -> {exc}")
    return torch, vllm


@app.cell
def _():
    import json
    import os
    import random
    import re
    from collections import Counter

    def extract_boxed(text):
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

    def norm_answer(s):
        if s is None:
            return None
        s = str(s).strip().rstrip(".")
        s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
        s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
        s = re.sub(r"\s+", "", s)
        return s or None

    def rep_stats(text, n=25):
        """(fraction of words covered by the most frequent n-gram, distinct-word fraction)."""
        w = text.split()
        if len(w) < 2 * n:
            return 0.0, 1.0
        grams = Counter(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
        _, cnt = grams.most_common(1)[0]
        return min(1.0, cnt * n / len(w)), len(set(w)) / len(w)

    return Counter, extract_boxed, json, norm_answer, os, random, re, rep_stats


@app.cell
def _():
    # ----------------------------------------------------------------------
    # Resilient dataset loading: HF dataset IDs move around, so try candidates
    # in order and REPORT what resolved. Never fail the whole run on one miss.
    # ----------------------------------------------------------------------
    from datasets import load_dataset

    RESOLUTION_LOG = []

    def try_load(candidates, split_candidates=("test", "train"), token=None):
        """candidates: list of (repo_id, config_or_None). Returns (ds, label) or (None, None)."""
        for repo, cfg in candidates:
            for split in split_candidates:
                try:
                    ds = (
                        load_dataset(repo, cfg, split=split, token=token)
                        if cfg
                        else load_dataset(repo, split=split, token=token)
                    )
                    label = f"{repo}{('/' + cfg) if cfg else ''}:{split}"
                    RESOLUTION_LOG.append(("OK  ", label, f"{len(ds)} rows"))
                    return ds, label
                except Exception as exc:
                    RESOLUTION_LOG.append(
                        ("miss", f"{repo}{('/' + cfg) if cfg else ''}:{split}",
                         str(exc)[:110])
                    )
        return None, None

    def first_field(row, *names):
        for n in names:
            v = row.get(n)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    return RESOLUTION_LOG, first_field, load_dataset, try_load


@app.cell
def _(
    N_COMPMATH,
    N_MATH500,
    N_MMLU_PRO_MATH,
    RESOLUTION_LOG,
    SEED,
    first_field,
    random,
    try_load,
):
    # ----------------------------------------------------------------------
    # Build the prompt set
    # ----------------------------------------------------------------------
    rng = random.Random(SEED)
    prompts = []

    def add_rows(rows, source, kind, n=None, level_key=None, stratify_key=None):
        rows = list(rows)
        if stratify_key:
            buckets = {}
            for r in rows:
                buckets.setdefault(str(r.get(stratify_key)), []).append(r)
            for b in buckets.values():
                rng.shuffle(b)
            picked, keys = [], sorted(buckets)
            while len(picked) < (n or len(rows)) and any(buckets[k] for k in keys):
                for k in keys:
                    if buckets[k] and len(picked) < (n or len(rows)):
                        picked.append(buckets[k].pop())
            rows = picked
        else:
            rng.shuffle(rows)
            if n:
                rows = rows[:n]
        for r in rows:
            prompts.append({**r, "source": source, "kind": kind,
                            "level": str(r.get(level_key)) if level_key else None})
        return len(rows)

    def mcq_body(question, options):
        """Render an MCQ inside the problem body so the eval preamble stays identical."""
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opts = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
        return (f"{question}\n\n{opts}\n\n"
                f"Give the letter of the correct option as your final answer.")

    # --- MATH-500 (stratified across levels) ---
    _ds, _lab = try_load([("HuggingFaceH4/MATH-500", None)], ("test",))
    if _ds is not None:
        add_rows(
            [{"problem": first_field(r, "problem"),
              "ref_answer": first_field(r, "answer"),
              "level": str(r.get("level"))} for r in _ds],
            _lab, "free", N_MATH500, level_key="level", stratify_key="level",
        )

    # --- AIME 2024 / 2025 / 2026 (taken whole) ---
    AIME_SETS = {
        "AIME2024": [("Maxwell-Jia/AIME_2024", None), ("HuggingFaceH4/aime_2024", None),
                     ("math-ai/aime24", None), ("AI-MO/aimo-validation-aime", None)],
        "AIME2025": [("yentinglin/aime_2025", None), ("math-ai/aime25", None),
                     ("opencompass/AIME2025", "AIME2025-I"), ("MathArena/aime_2025", None)],
        "AIME2026": [("math-ai/aime26", None), ("MathArena/aime_2026", None),
                     ("yentinglin/aime_2026", None), ("opencompass/AIME2026", None)],
    }
    for _tag, _cands in AIME_SETS.items():
        _d, _l = try_load(_cands, ("test", "train"))
        if _d is not None:
            add_rows(
                [{"problem": first_field(r, "problem", "Problem", "question"),
                  "ref_answer": first_field(r, "answer", "Answer", "solution")} for r in _d],
                f"{_tag}[{_l}]", "free", None,
            )

    # --- CompMath-MCQ (ID unverified — candidates tried in order) ---
    _d, _l = try_load([("CompMath/CompMath-MCQ", None), ("compmath/CompMath-MCQ", None),
                       ("CompMath-MCQ", None), ("nvidia/CompMath-MCQ", None)],
                      ("test", "train"))
    if _d is not None:
        _rows = []
        for r in _d:
            _opts = r.get("options") or r.get("choices") or []
            _q = first_field(r, "question", "problem", "Question")
            if not _q:
                continue
            _rows.append({"problem": mcq_body(_q, list(_opts)) if _opts else _q,
                          "ref_answer": first_field(r, "answer", "Answer", "correct_answer")})
        add_rows(_rows, _l, "mcq", N_COMPMATH)

    # --- MMLU-Pro, math category ---
    _d, _l = try_load([("TIGER-Lab/MMLU-Pro", None)], ("test",))
    if _d is not None:
        _rows = [
            {"problem": mcq_body(first_field(r, "question"), list(r.get("options") or [])),
             "ref_answer": first_field(r, "answer")}
            for r in _d if str(r.get("category", "")).lower() == "math"
        ]
        add_rows(_rows, f"{_l}/math", "mcq", N_MMLU_PRO_MATH)

    print("DATASET RESOLUTION")
    print("=" * 78)
    for _status, _label, _info in RESOLUTION_LOG:
        print(f"  [{_status}] {_label:<48} {_info}")
    print("=" * 78)
    print(f"\nTotal prompts: {len(prompts)}")
    from collections import Counter as _C
    for _s, _n in sorted(_C(p["source"] for p in prompts).items()):
        print(f"  {_s:<52} {_n:>5}")
    print(f"\n  free-response: {sum(1 for p in prompts if p['kind']=='free')}"
          f"   multiple-choice: {sum(1 for p in prompts if p['kind']=='mcq')}")
    return AIME_SETS, add_rows, mcq_body, prompts, rng


@app.cell
def _(MATH_INSTRUCTION, MODEL_ID, prompts):
    # ----------------------------------------------------------------------
    # Render prompts exactly as evaluation/common.py::build_math_prompt does,
    # then through the chat template with thinking enabled.
    # ----------------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    rendered = []
    for _p in prompts:
        _user = f"{MATH_INSTRUCTION}\n\nProblem:\n{_p['problem'].strip()}"
        _p["prompt"] = _user
        try:
            _text = tok.apply_chat_template(
                [{"role": "user", "content": _user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
        except TypeError:  # template does not accept enable_thinking
            _text = tok.apply_chat_template(
                [{"role": "user", "content": _user}],
                tokenize=False, add_generation_prompt=True,
            )
        rendered.append(_text)

    print("--- rendered prompt [0] ---")
    print(rendered[0][:900])
    return AutoTokenizer, rendered, tok


@app.cell
def _(ENV_READY, GPU_MEM_UTIL, MAX_MODEL_LEN, MAX_NUM_SEQS, MODEL_ID,
      SERVED_MODEL_NAME):
    # ----------------------------------------------------------------------
    # SERVE. Identical mechanism to evalkit/notebooks/run_eval.py: vLLM runs as
    # an OpenAI-compatible API server in its OWN PROCESS, not as an in-process
    # `LLM()` object.
    #
    # This is not cosmetic. In-process, an EngineCore crash takes the notebook
    # kernel's Python state with it and all you get is `EngineDeadError`. As a
    # subprocess, the engine's stdout lands in a log file you can read, and a
    # crash leaves the notebook alive.
    #
    # `ENV_READY` is an argument purely to force marimo to run the env-var cell
    # first -- marimo orders cells by data dependency, not by position in the
    # file, and VLLM_USE_FLASHINFER_SAMPLER must be set before the server forks.
    # ----------------------------------------------------------------------
    # Underscore-prefixed imports are cell-local in marimo, so they do not
    # collide with the `os` / `subprocess` / `sys` defined in earlier cells.
    import os as _os
    import socket as _socket
    import subprocess as _sp
    import sys as _sys
    import time
    import urllib.request as _urlreq
    from pathlib import Path as _Path

    assert ENV_READY

    PY = _sys.executable
    SERVER_LOG = _Path("vllm_calib_server.log")
    VLLM = {"proc": None, "port": None}

    def _free_port():
        s = _socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close()
        return p

    def stop_vllm():
        proc = VLLM.get("proc")
        if proc and proc.poll() is None:
            _os.killpg(_os.getpgid(proc.pid), 15)
            try:
                proc.wait(timeout=60)
            except Exception:
                _os.killpg(_os.getpgid(proc.pid), 9)
            print("vLLM stopped")
        VLLM.update(proc=None, port=None)

    def start_vllm():
        if VLLM["proc"] and VLLM["proc"].poll() is None:
            print(f"reusing server on port {VLLM['port']}")
            return VLLM["port"]

        port = _free_port()
        env = {**_os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        cmd = [PY, "-m", "vllm.entrypoints.openai.api_server",
               "--model", MODEL_ID,
               "--served-model-name", SERVED_MODEL_NAME,
               "--tensor-parallel-size", "1",
               "--dtype", "bfloat16",
               "--max-model-len", str(MAX_MODEL_LEN),
               "--port", str(port),
               "--gpu-memory-utilization", GPU_MEM_UTIL,
               "--max-num-seqs", MAX_NUM_SEQS,
               "--trust-remote-code"]

        # Qwen3.5-4B carries a vision tower we never use. Skipping it saves a
        # little memory -- but the flag has moved between vLLM releases and an
        # unknown flag kills the server with an argparse error buried in the
        # log, so only pass it if this build advertises it. (Same guard as
        # evalkit.)
        _help = _sp.run([PY, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                               capture_output=True, text=True)
        if "--language-model-only" in (_help.stdout + _help.stderr):
            cmd.append("--language-model-only")
        else:
            print("note: this vLLM has no --language-model-only; omitting it")

        log = open(SERVER_LOG, "w")
        proc = _sp.Popen(cmd, stdout=log, stderr=_sp.STDOUT,
                                env=env, start_new_session=True)
        VLLM.update(proc=proc, port=port)
        print(f"vLLM PID={proc.pid} port={port}  log={SERVER_LOG.resolve()}")
        print("startup takes ~3 min (weights, compile, CUDA-graph capture) ...")

        deadline = time.time() + 1800
        while time.time() < deadline:
            if proc.poll() is not None:
                print(SERVER_LOG.read_text()[-6000:])
                raise RuntimeError("vLLM exited before becoming ready -- log above")
            try:
                _urlreq.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                print("server ready")
                return port
            except Exception:
                time.sleep(10)
        raise RuntimeError("vLLM did not become ready within 30 min")

    PORT = start_vllm()
    return PORT, SERVER_LOG, start_vllm, stop_vllm, time


@app.cell
def _(MAX_CONCURRENCY, PORT, SAMPLING_KW, SERVED_MODEL_NAME, rendered, time):
    # ----------------------------------------------------------------------
    # GENERATE over HTTP, MAX_CONCURRENCY requests in flight.
    #
    # /v1/completions, not /v1/chat/completions: we already applied the chat
    # template ourselves in the cell above, and the raw-completions endpoint
    # sends those exact tokens. Going through the chat endpoint would apply the
    # template a second time.
    #
    # top_k / min_p / repetition_penalty are not OpenAI-spec fields, so they
    # travel in extra_body -- vLLM reads them from there.
    # ----------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY",
                    timeout=7200, max_retries=2)

    _core = dict(
        temperature=SAMPLING_KW["temperature"],
        top_p=SAMPLING_KW["top_p"],
        presence_penalty=SAMPLING_KW["presence_penalty"],
        max_tokens=SAMPLING_KW["max_tokens"],
    )
    _extra = dict(
        top_k=SAMPLING_KW["top_k"],
        min_p=SAMPLING_KW["min_p"],
        repetition_penalty=SAMPLING_KW["repetition_penalty"],
    )

    _done = {"n": 0}

    def _one(i_text):
        i, text = i_text
        try:
            r = client.completions.create(
                model=SERVED_MODEL_NAME, prompt=text,
                extra_body=_extra, **_core,
            )
            c = r.choices[0]
            out = {
                "index": i,
                "text": c.text,
                "finish_reason": c.finish_reason,
                "n_tokens": (r.usage.completion_tokens if r.usage else None),
                "error": None,
            }
        except Exception as exc:
            out = {"index": i, "text": "", "finish_reason": "error",
                   "n_tokens": 0, "error": repr(exc)[:400]}
        _done["n"] += 1
        if _done["n"] % 10 == 0 or _done["n"] == len(rendered):
            print(f"  {_done['n']}/{len(rendered)} done", flush=True)
        return out

    _t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as _ex:
        raw_outputs = sorted(_ex.map(_one, enumerate(rendered)),
                             key=lambda d: d["index"])
    GEN_SECONDS = time.time() - _t0

    _errs = [o for o in raw_outputs if o["error"]]
    print(f"\ngenerated {len(raw_outputs)} traces in {GEN_SECONDS/60:.1f} min")
    print(f"errors: {len(_errs)}")
    for _e in _errs[:5]:
        print(f"  [{_e['index']}] {_e['error']}")
    return GEN_SECONDS, OpenAI, ThreadPoolExecutor, client, raw_outputs


@app.cell
def _(extract_boxed, norm_answer, prompts, raw_outputs, rep_stats):
    # ----------------------------------------------------------------------
    # Annotate every trace. Record, do not filter.
    # ----------------------------------------------------------------------
    # raw_outputs are plain dicts from the HTTP client, not vLLM RequestOutput
    # objects: {"text", "finish_reason", "n_tokens", "error"}.
    records = []
    for _p, _o in zip(prompts, raw_outputs):
        _txt = _o["text"]
        _pred = extract_boxed(_txt)
        _ref = norm_answer(_p.get("ref_answer"))
        _rf, _df = rep_stats(_txt)
        records.append({
            "prompt": _p["prompt"],
            "problem": _p["problem"],
            "trace": _txt,
            "n_tokens": _o["n_tokens"] or 0,
            "source": _p["source"],
            "kind": _p["kind"],
            "level": _p.get("level"),
            "ref_answer": _p.get("ref_answer"),
            "pred_answer": _pred,
            "truncated": _o["finish_reason"] == "length",
            "failed": _o["error"] is not None,
            "has_boxed": _pred is not None,
            "correct": (norm_answer(_pred) == _ref) if (_pred and _ref) else None,
            "rep_frac": round(_rf, 4),
            "distinct_frac": round(_df, 4),
        })
    print(f"annotated {len(records)} records")
    return (records,)


@app.cell
def _(records):
    # ----------------------------------------------------------------------
    # REPORT. The length histogram is what decides CALIB_SEQLEN.
    # ----------------------------------------------------------------------
    REP_FRAC_MAX = 0.30
    DISTINCT_FRAC_MIN = 0.08

    def is_degenerate(r):
        return r["rep_frac"] > REP_FRAC_MAX or r["distinct_frac"] < DISTINCT_FRAC_MIN

    clean = [r for r in records if not is_degenerate(r)]
    lens = [r["n_tokens"] for r in clean]

    def hist(values, edges, label):
        out = [f"\n  {label}"]
        tot = len(values) or 1
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = sum(1 for v in values if lo <= v < hi)
            out.append(f"    [{lo:>6} , {hi:>6}) {n:>5} {'#' * int(55 * n / tot)}")
        n = sum(1 for v in values if v >= edges[-1])
        out.append(f"    [{edges[-1]:>6} ,    inf) {n:>5} {'#' * int(55 * n / tot)}")
        return "\n".join(out)

    _lines = ["=" * 78, "CORPUS REPORT", "=" * 78,
              f"  generated               : {len(records)}",
              f"  degenerate (dropped)    : {len(records) - len(clean)}",
              f"  kept                    : {len(clean)}",
              f"  total tokens kept       : {sum(lens):,}",
              f"  truncated at 32K        : {sum(r['truncated'] for r in clean)}",
              f"  emitted \\boxed{{}}        : {sum(r['has_boxed'] for r in clean)}"]

    _scored = [r for r in clean if r["correct"] is not None]
    if _scored:
        _nc = sum(r["correct"] for r in _scored)
        _lines.append(f"  correct (where scorable): {_nc}/{len(_scored)} = {_nc/len(_scored)*100:.1f}%")

    _lines.append("\n  TOKENS BY SOURCE (this is what actually weights the Hessian)")
    _by = {}
    for r in clean:
        _d = _by.setdefault(r["source"], {"n": 0, "tok": 0, "kind": r["kind"]})
        _d["n"] += 1
        _d["tok"] += r["n_tokens"]
    _tt = sum(v["tok"] for v in _by.values()) or 1
    for _s, _v in sorted(_by.items(), key=lambda kv: -kv[1]["tok"]):
        _lines.append(f"    {_s[:46]:<46} {_v['kind']:<5} {_v['n']:>4} traces "
                      f"{_v['tok']:>9,} tok  {_v['tok']/_tt*100:5.1f}%")
    _mcq = sum(v["tok"] for v in _by.values() if v["kind"] == "mcq")
    _lines.append(f"    -> multiple-choice share of tokens: {_mcq/_tt*100:.1f}%")

    _lines.append(hist(lens, [0, 512, 1024, 2048, 4096, 8192, 16384, 32000],
                       "TRACE LENGTH (tokens) -- THIS DECIDES CALIB_SEQLEN"))
    for _L in (2048, 4096, 8192, 16384):
        _n = sum(1 for t in lens if t >= _L)
        _lines.append(f"    traces >= {_L:>5}: {_n:>4} ({_n/max(len(lens),1)*100:5.1f}%)  "
                      f"{sum(t for t in lens if t >= _L):>9,} tokens at or above")

    _lines.append(hist([r["rep_frac"] for r in records], [0, 0.1, 0.2, 0.3, 0.5, 0.8],
                       "repetition fraction -- retune REP_FRAC_MAX from this"))
    _lines.append("=" * 78)

    REPORT = "\n".join(_lines)
    print(REPORT)
    return DISTINCT_FRAC_MIN, REPORT, REP_FRAC_MAX, clean, hist, is_degenerate, lens


@app.cell
def _(
    GEN_SECONDS,
    HF_PRIVATE,
    HF_REPO_ID,
    MODEL_ID,
    REPORT,
    REP_FRAC_MAX,
    SAMPLING_KW,
    SEED,
    clean,
    json,
    os,
    records,
):
    # ----------------------------------------------------------------------
    # Upload to a Hugging Face dataset repo.
    # Needs a WRITE token: os.environ["HF_TOKEN"] = "hf_..." before this cell.
    # ----------------------------------------------------------------------
    from huggingface_hub import HfApi, create_repo

    _token = os.environ.get("HF_TOKEN")
    assert _token, "Set os.environ['HF_TOKEN'] to a WRITE token before running this cell."

    os.makedirs("upload", exist_ok=True)

    with open("upload/traces_all.jsonl", "w") as _fh:
        for r in records:
            _fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("upload/traces_clean.jsonl", "w") as _fh:
        for r in clean:
            _fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    META = {
        "model": MODEL_ID,
        "sampling": SAMPLING_KW,
        "seed": SEED,
        "n_generated": len(records),
        "n_clean": len(clean),
        "total_tokens_clean": sum(r["n_tokens"] for r in clean),
        "rep_frac_max": REP_FRAC_MAX,
        "generation_seconds": round(GEN_SECONDS, 1),
    }
    with open("upload/meta.json", "w") as _fh:
        json.dump(META, _fh, indent=2)
    with open("upload/report.txt", "w") as _fh:
        _fh.write(REPORT)

    CARD = f"""---
license: apache-2.0
task_categories: [text-generation]
tags: [calibration, gptq, quantization, math, cs6013]
---

# CS6013 calibration traces — {MODEL_ID}

Self-generated `<think>` traces from the **bf16 base model**, for use as a GPTQ
calibration corpus. No labels were used to select or filter rows.

## Generation
Prompts rendered exactly as the course eval harness does:
`f"{{math_instruction}}\\n\\nProblem:\\n{{problem}}"`, then the chat template with
`add_generation_prompt=True` and thinking enabled.

Sampling matches the graded eval config exactly:
```
{json.dumps(SAMPLING_KW, indent=2)}
```
`presence_penalty = 1.5` is unusually high and materially reshapes the token
distribution; matching it is the point of this corpus.

## Files
| file | contents |
|---|---|
| `traces_all.jsonl` | every generated trace, annotated |
| `traces_clean.jsonl` | degenerate traces removed (`rep_frac > {REP_FRAC_MAX}`) |
| `meta.json` | model, sampling params, seed, counts |
| `report.txt` | full corpus report incl. trace-length histogram |

## Fields
`prompt`, `problem`, `trace`, `n_tokens`, `source`, `kind` (`free`/`mcq`),
`level`, `ref_answer`, `pred_answer`, `truncated`, `has_boxed`, `correct`,
`rep_frac`, `distinct_frac`.

## Filtering policy
Only **degeneracy** (repetition loops) is filtered. Correctness and truncation
are recorded but **not** filtered: the compressed model will get problems wrong
too, and correctness filtering would systematically retain the easier problems.
Truncated traces are the most valuable long-context data in the set.

## Report
```
{REPORT}
```
"""
    with open("upload/README.md", "w") as _fh:
        _fh.write(CARD)

    create_repo(HF_REPO_ID, repo_type="dataset", private=HF_PRIVATE,
                exist_ok=True, token=_token)
    HfApi().upload_folder(folder_path="upload", repo_id=HF_REPO_ID,
                          repo_type="dataset", token=_token)
    print(f"uploaded -> https://huggingface.co/datasets/{HF_REPO_ID}")
    return CARD, HfApi, META, create_repo


if __name__ == "__main__":
    app.run()
