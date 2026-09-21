import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # CS6013 — local eval harness

        Reproduces the graders' pipeline for **one** submission, using their own
        scripts wherever they exist (`measure_checkpoint_bits.py`,
        `ensure_visual_zero.py`, `evaluation/run_eval.py`, `configs/eval_config.yaml`)
        so the numbers here mean the same thing as the numbers on the leaderboard.

        Dropped from `eval.sh`: Slack, the submissions CSV, the batch loop over
        students, and the results-CSV resume logic. Kept: format validation,
        sparse clone, `uv sync` of your `pyproject.toml`, visual zeroing,
        `size_frac`, vLLM serving, and the eval itself.

        **Order:** settings → environment → install → build suites → format check →
        clone → download → size → decompress → serve → eval → baseline → compare.

        Attach a GPU from the notebook-specs button before running anything.
        """
    )
    return


@app.cell
def _(mo):
    github_url = mo.ui.text(
        placeholder="https://github.com/<you>/CS6013/tree/main/<roll>/Week05/Compression_40/Submission02",
        label="GitHub URL (must include /tree/<branch>/... path)",
        full_width=True,
    )
    hf_url = mo.ui.text(
        placeholder="https://huggingface.co/<you>/<roll>-Week05-Compression40-Submission02",
        label="HuggingFace URL",
        full_width=True,
    )
    roll_no = mo.ui.text(placeholder="23B1266", label="Roll no.")
    week_no = mo.ui.text(value="05", label="Week (2 digits)")
    target_no = mo.ui.text(value="40", label="Compression target")

    gh_token = mo.ui.text(
        placeholder="leave empty if your repo is public",
        label="GitHub token (optional)",
        kind="password",
    )
    hf_token = mo.ui.text(
        placeholder="hf_... (needed only for private HF repos)",
        label="HF token (optional)",
        kind="password",
    )
    bundle_path = mo.ui.text(
        value="/root/work/CS6013Fall_ProjectEval.zip",
        label="Eval bundle (.zip or extracted dir)",
        full_width=True,
    )
    base_model_id = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model (for baseline)")

    mo.vstack([
        mo.md("### Settings"),
        github_url,
        hf_url,
        mo.hstack([roll_no, week_no, target_no], justify="start"),
        mo.hstack([gh_token, hf_token], justify="start"),
        bundle_path,
        base_model_id,
    ])
    return (
        base_model_id,
        bundle_path,
        gh_token,
        github_url,
        hf_token,
        hf_url,
        roll_no,
        target_no,
        week_no,
    )


@app.cell
def _():
    import json
    import os
    import re
    import shutil
    import socket
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    WORK = Path("/root/work") if Path("/root").exists() else Path.home() / "work"
    BUNDLE_DIR = WORK / "evalkit"
    SUITE_DIR = WORK / "suites"
    REPO_DIR = WORK / "submission_repo"
    COMP_DIR = WORK / "compressed_model"
    DEC_DIR = WORK / "decompressed_model"
    BASE_DIR = WORK / "base_model"
    OUT_DIR = WORK / "eval_outputs"
    for _d in (WORK, SUITE_DIR, OUT_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # Exactly the graders' serving + sampling parameters. Changing these changes
    # what the accuracy number means, so they are constants, not knobs.
    MAX_NEW_TOKENS = 32000
    MAX_MODEL_LEN = 1000 + MAX_NEW_TOKENS
    MAX_CONCURRENCY = 30
    SERVED_MODEL_NAME = "qwen-3.5-4b"
    GPU_MEM_UTIL = "0.35"
    MAX_NUM_SEQS = "50"
    ORIGINAL_TEXT_GB = 8.0585

    VLLM = {"proc": None, "port": None, "model": None}

    def run_streaming(cmd, cwd=None, env=None, quiet_prefixes=()):
        merged = {**os.environ, **(env or {})}
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            if not any(line.startswith(p) for p in quiet_prefixes):
                print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"failed ({proc.returncode}): {' '.join(map(str, cmd))}")

    def capture(cmd, cwd=None):
        return subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, check=True,
        ).stdout

    def free_port():
        s = socket.socket()
        s.bind(("", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def dir_gb(path):
        path = Path(path)
        if not path.exists():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30

    def disk_free_gb():
        return shutil.disk_usage(WORK).free / 2**30

    return (
        BASE_DIR,
        BUNDLE_DIR,
        COMP_DIR,
        DEC_DIR,
        GPU_MEM_UTIL,
        MAX_CONCURRENCY,
        MAX_MODEL_LEN,
        MAX_NEW_TOKENS,
        MAX_NUM_SEQS,
        ORIGINAL_TEXT_GB,
        OUT_DIR,
        Path,
        REPO_DIR,
        SERVED_MODEL_NAME,
        SUITE_DIR,
        VLLM,
        WORK,
        capture,
        dir_gb,
        disk_free_gb,
        free_port,
        json,
        os,
        re,
        run_streaming,
        shutil,
        socket,
        subprocess,
        sys,
        time,
        urllib,
    )


@app.cell
def _(WORK, disk_free_gb, mo, subprocess, sys):
    def _env():
        lines = [f"- Python `{sys.version.split()[0]}`"]
        try:
            _out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"], text=True).strip()
            lines.append(f"- GPU: **{_out}**")
        except Exception:
            lines.append("- **No GPU detected.** Attach one from the notebook specs button.")
        try:
            kb = int(subprocess.check_output(["grep", "MemTotal", "/proc/meminfo"]).split()[1])
            lines.append(f"- Host RAM: {kb / 1024**2:.0f} GiB")
        except Exception:
            pass
        free = disk_free_gb()
        lines.append(f"- Disk free at `{WORK}`: **{free:.0f} GiB**")
        lines.append("")
        lines.append(
            "Budget: eval venv ~15, compressed ~3, decompressed ~8, base model ~9 "
            "→ **~35 GiB** for the full run including baseline. The notebook frees "
            "the decompressed model before the baseline download; if you are tight, "
            "run the compressed eval, record the numbers, then do the baseline."
        )
        if free < 35:
            lines.append("")
            lines.append(f"- **Only {free:.0f} GiB free.** Expect to clean up between stages.")
        return "\n".join(lines)

    mo.md("### Environment\n" + _env())
    return


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="Install uv + eval environment (~10 min)")
    mo.vstack([
        mo.md(
            "### 1. Eval environment\n"
            "Extracts the bundle and runs `uv sync --extra cuda129` against the "
            "graders' own `pyproject.toml`, so vLLM, transformers and torch are at "
            "exactly the versions they use."
        ),
        setup_btn,
    ])
    return (setup_btn,)


@app.cell
def _(BUNDLE_DIR, Path, bundle_path, mo, run_streaming, setup_btn, shutil, sys):
    mo.stop(not setup_btn.value, mo.md("*Not installed.*"))

    _src = Path(bundle_path.value).expanduser()
    mo.stop(not _src.exists(), mo.md(f"**Bundle not found at `{_src}`.** Upload the zip first."))

    if _src.is_file():
        shutil.rmtree(BUNDLE_DIR, ignore_errors=True)
        BUNDLE_DIR.mkdir(parents=True)
        shutil.unpack_archive(str(_src), str(BUNDLE_DIR))
        _inner = [p for p in BUNDLE_DIR.iterdir() if p.is_dir()]
        KIT = _inner[0] if len(_inner) == 1 else BUNDLE_DIR
    else:
        KIT = _src

    assert (KIT / "eval.sh").exists(), f"eval.sh not found under {KIT}"
    print(f"eval kit: {KIT}")

    run_streaming([sys.executable, "-m", "pip", "install", "-q", "uv", "datasets", "pyyaml"])
    print("\nsyncing the graders' environment (this is the slow part) ...")
    run_streaming(["uv", "sync", "--extra", "cuda129"], cwd=KIT)

    EVAL_PY = KIT / ".venv" / "bin" / "python"
    assert EVAL_PY.exists(), "uv sync did not create .venv"
    print(f"\neval interpreter: {EVAL_PY}")
    return EVAL_PY, KIT


@app.cell
def _(mo):
    suites_btn = mo.ui.run_button(label="Build the 5 benchmark suites")
    mo.vstack([
        mo.md(
            """
### 2. Benchmark suites

A difficulty ladder, scored separately. Quantization damage is not uniform — a
4-bit model usually still nails grade-school arithmetic while collapsing on
problems needing ten chained steps, and one aggregate number hides exactly that.

| suite | n | source | role |
|---|---|---|---|
| `smoke` | 8 | hand-written | ~2 min; is anything catastrophically broken |
| `gsm8k` | 100 | openai/gsm8k | easy word problems; the floor |
| `math500` | 150 | HuggingFaceH4/MATH-500 | broad topics, all experts get traffic |
| `amc` | 83 | AI-MO/aimo-validation-amc | competition; where quantization bites |
| `aime` | 90 | AI-MO aime + MathArena/aime_2025 | hardest; long reasoning, token-limit stress |

`gsm8k` and `math500` are near-certainly in Qwen's training data. That is fine:
these measure *compressed vs baseline on identical inputs*, and contamination
shifts both arms equally. Don't read the absolute numbers as a capability claim.

Requires `eval_suites.py` next to this notebook.
"""
        ),
        suites_btn,
    ])
    return (suites_btn,)


@app.cell
def _(Path, SUITE_DIR, mo, suites_btn):
    mo.stop(not suites_btn.value, mo.md("*Suites not built.*"))

    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("eval_suites", str(Path("eval_suites.py").resolve()))
    mo.stop(_spec is None, mo.md("**`eval_suites.py` not found next to this notebook.**"))
    ES = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(ES)

    from datasets import Dataset, DatasetDict, load_dataset

    SUITES_BUILT = ES.build_suites(SUITE_DIR, load_dataset, Dataset, DatasetDict)
    print("\nverifying each suite round-trips through the graders' loader:")
    _fail = ES.verify_suites(SUITES_BUILT, load_dataset)
    if _fail:
        raise RuntimeError(f"suite verification failed: {_fail}")
    print("\nall suites OK")
    return ES, SUITES_BUILT


@app.cell
def _(github_url, hf_url, mo, re, roll_no, target_no, week_no):
    def _check():
        _week = f"Week{int(week_no.value or 0):02d}"
        _tgt = re.search(r"(\d+)", target_no.value or "")
        _tgt = _tgt.group(1) if _tgt else ""
        _roll = (roll_no.value or "").strip()
        errs = []

        gh = (github_url.value or "").strip().rstrip("/")
        m = re.match(r"^https?://github\.com/([^/]+)/(CS6013)(?:\.git)?/tree/([^/]+)/(.+)$", gh, re.I)
        if not m:
            errs.append(
                "GitHub URL must be `https://github.com/<user>/CS6013/tree/<branch>/"
                f"{_roll}/{_week}/Compression_{_tgt}/SubmissionNN`"
            )
        else:
            if m.group(2) != "CS6013":
                errs.append(f"repo must be exactly `CS6013`, got `{m.group(2)}`")
            sub = m.group(4).strip("/")
            pm = re.match(r"^([^/]+)/(Week\d{2})/(Compression_[^/]+)/(Submission\d{2})/?$", sub)
            if not pm:
                errs.append(
                    f"path must be `<roll>/{_week}/Compression_{_tgt}/SubmissionNN` "
                    f"(note the **underscore**, two-digit week/submission), got `{sub}`"
                )
            else:
                if _roll and pm.group(1).lower() != _roll.lower():
                    errs.append(f"GitHub roll `{pm.group(1)}` != `{_roll}`")
                if pm.group(2) != _week:
                    errs.append(f"GitHub week `{pm.group(2)}` != `{_week}`")
                if _tgt and pm.group(3) != f"Compression_{_tgt}":
                    errs.append(f"GitHub folder `{pm.group(3)}` != `Compression_{_tgt}`")

        hf = (hf_url.value or "").strip().rstrip("/")
        hm = re.match(r"^https?://huggingface\.co/([^/]+)/([^/]+)/?$", hf, re.I)
        if not hm:
            errs.append("HuggingFace URL must be `https://huggingface.co/<user>/<repo>`")
        else:
            nm = re.fullmatch(r"^(.+)-(Week\d{2})-Compression-?(\d+)-(Submission\d{2})$", hm.group(2))
            if not nm:
                errs.append(
                    f"HF repo must be `{_roll}-{_week}-Compression{_tgt}-SubmissionNN` "
                    f"(`Compression40` or `Compression-40`, **never** `Compression_40`), "
                    f"got `{hm.group(2)}`"
                )
            else:
                if _roll and nm.group(1).lower() != _roll.lower():
                    errs.append(f"HF roll `{nm.group(1)}` != `{_roll}`")
                if nm.group(2) != _week:
                    errs.append(f"HF week `{nm.group(2)}` != `{_week}`")
                if _tgt and nm.group(3) != _tgt:
                    errs.append(f"HF target `{nm.group(3)}` != `{_tgt}`")
                gsub = pm.group(4) if (m and pm) else None
                if gsub and gsub != nm.group(4):
                    errs.append(f"GitHub `{gsub}` != HuggingFace `{nm.group(4)}`")

        if not gh or not hf:
            return "### 3. Format check\n\n*Fill in both URLs above.*"
        if errs:
            return "### 3. Format check\n\n**FAILS — this is an automatic zero:**\n\n" + \
                "\n".join(f"- {e}" for e in errs)
        return (
            "### 3. Format check\n\nBoth URLs match the required naming. "
            "The graders abort the whole batch on a mismatch here, so this is worth "
            "getting right before anything else."
        )

    mo.md(_check())
    return


@app.cell
def _(mo):
    clone_btn = mo.ui.run_button(label="Sparse-clone the submission")
    clone_btn
    return (clone_btn,)


@app.cell
def _(REPO_DIR, clone_btn, gh_token, github_url, mo, re, run_streaming, shutil):
    mo.stop(not clone_btn.value, mo.md("*Not cloned.*"))

    _m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/tree/([^/]+)/(.+)$",
                  (github_url.value or "").strip().rstrip("/"))
    mo.stop(_m is None, mo.md("**GitHub URL must include `/tree/<branch>/<path>`.**"))
    _owner, _repo, _branch, _subdir = _m.groups()

    _clone = f"https://github.com/{_owner}/{_repo}.git"
    if gh_token.value:
        _clone = f"https://{_owner}:{gh_token.value}@github.com/{_owner}/{_repo}.git"

    shutil.rmtree(REPO_DIR, ignore_errors=True)
    # Same sparse strategy the graders use: only the submission folder is fetched.
    run_streaming(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                   "--no-checkout", "--branch", _branch, _clone, str(REPO_DIR)])
    run_streaming(["git", "-C", str(REPO_DIR), "sparse-checkout", "set", "--cone", _subdir])
    if not (REPO_DIR / _subdir).exists():
        run_streaming(["git", "-C", str(REPO_DIR), "checkout", _branch])

    SUB_DIR = REPO_DIR / _subdir
    assert SUB_DIR.is_dir(), f"sparse checkout produced no {SUB_DIR}"

    _required_files = ["compress.py", "decompress.py", "pyproject.toml", "README.md",
                       "compression/__init__.py", "decompression/__init__.py"]
    _required_dirs = ["compression", "decompression"]
    _missing = [f for f in _required_files if not (SUB_DIR / f).is_file()]
    _missing += [f"{d}/" for d in _required_dirs if not (SUB_DIR / d).is_dir()]

    print(f"submission: {SUB_DIR}\n")
    if _missing:
        raise RuntimeError(f"LAYOUT INCOMPLETE — graders fail at step 0_format. Missing: {_missing}")
    print("layout OK: " + ", ".join(_required_files))
    return (SUB_DIR,)


@app.cell
def _(mo):
    download_btn = mo.ui.run_button(label="Download compressed checkpoint")
    download_btn
    return (download_btn,)


@app.cell
def _(COMP_DIR, dir_gb, download_btn, hf_token, hf_url, mo, os, re, shutil):
    mo.stop(not download_btn.value, mo.md("*Not downloaded.*"))

    _m2 = re.match(r"^https?://huggingface\.co/([^/]+)/([^/]+)", (hf_url.value or "").strip().rstrip("/"))
    mo.stop(_m2 is None, mo.md("**Unrecognized HuggingFace URL.**"))
    HF_ID = f"{_m2.group(1)}/{_m2.group(2)}"

    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value

    from huggingface_hub import snapshot_download as _snap

    shutil.rmtree(COMP_DIR, ignore_errors=True)
    _snap(HF_ID, local_dir=str(COMP_DIR))
    print(f"{HF_ID} -> {COMP_DIR}  ({dir_gb(COMP_DIR):.3f} GiB)")
    for _f in sorted(COMP_DIR.iterdir()):
        if _f.is_file():
            print(f"  {_f.name:<48} {_f.stat().st_size / 2**20:9.1f} MiB")
    return (HF_ID,)


@app.cell
def _(mo):
    measure_btn = mo.ui.run_button(label="Measure size_frac + zero visual tensors")
    mo.vstack([
        mo.md(
            "### 4. Size and visual zeroing\n"
            "`size_frac = compressed_text_GB / 8.0585`. Only **non-visual** tensors "
            "count, so the vision tower is free but `mtp` is not. Then "
            "`ensure_visual_zero.py` rewrites the checkpoint in place — your "
            "`decompress.py` runs against the zeroed version, exactly as on the "
            "grading node."
        ),
        measure_btn,
    ])
    return (measure_btn,)


@app.cell
def _(COMP_DIR, EVAL_PY, KIT, ORIGINAL_TEXT_GB, capture, json, measure_btn, mo, run_streaming):
    mo.stop(not measure_btn.value, mo.md("*Not measured.*"))

    _raw = capture([EVAL_PY, str(KIT / "measure_checkpoint_bits.py"), "--json", str(COMP_DIR)])
    _m3 = json.loads(_raw)
    SIZE_FRAC = _m3["text_gb"] / ORIGINAL_TEXT_GB

    print(f"total   {_m3['total_gb']:.4f} GiB")
    print(f"visual  {_m3['visual_gb']:.4f} GiB   (excluded from size_frac)")
    print(f"text    {_m3['text_gb']:.4f} GiB   (language_model + mtp)")
    print(f"\nsize_frac = {_m3['text_gb']:.4f} / {ORIGINAL_TEXT_GB} = {SIZE_FRAC:.4f}")
    print("WITHIN the 0.40 target" if SIZE_FRAC <= 0.40 else "*** OVER 0.40 — this submission fails ***")

    print("\nzeroing visual tensors in place ...")
    run_streaming([EVAL_PY, str(KIT / "ensure_visual_zero.py"), str(COMP_DIR)])
    return (SIZE_FRAC,)


@app.cell
def _(mo):
    decompress_btn = mo.ui.run_button(label="uv sync submission + run decompress.py")
    mo.vstack([
        mo.md(
            "### 5. Decompress\n"
            "Creates a venv from **your** `pyproject.toml` and runs `decompress.py` "
            "with that interpreter — the same two steps that fail as `3_venv` and "
            "`4_decompress` on the grading node. `--model_name` is passed as the "
            "literal string `Qwen-3.5-4B`, which is not a resolvable HF id: if your "
            "decompressor tries to download it, it dies here."
        ),
        decompress_btn,
    ])
    return (decompress_btn,)


@app.cell
def _(COMP_DIR, DEC_DIR, SUB_DIR, decompress_btn, dir_gb, mo, run_streaming, shutil):
    mo.stop(not decompress_btn.value, mo.md("*Not decompressed.*"))

    run_streaming(["uv", "sync"], cwd=SUB_DIR)
    _py = SUB_DIR / ".venv" / "bin" / "python"
    if not _py.exists():
        raise RuntimeError("uv sync produced no .venv in the submission dir")

    shutil.rmtree(DEC_DIR, ignore_errors=True)
    DEC_DIR.mkdir(parents=True)
    run_streaming([str(_py), "decompress.py",
                   "--model_name", "Qwen-3.5-4B",
                   "--checkpoint_path", str(COMP_DIR),
                   "--output_path", str(DEC_DIR)], cwd=SUB_DIR)

    _has = (DEC_DIR / "config.json").exists() or any(DEC_DIR.glob("*.safetensors"))
    if not _has:
        raise RuntimeError("decompress.py exited 0 but wrote no checkpoint")
    print(f"\nrestored: {dir_gb(DEC_DIR):.3f} GiB")
    return


@app.cell
def _(mo):
    serve_btn = mo.ui.run_button(label="Start vLLM on the decompressed model")
    serve_btn
    return (serve_btn,)


@app.cell
def _(
    GPU_MEM_UTIL,
    KIT,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    SERVED_MODEL_NAME,
    VLLM,
    free_port,
    os,
    subprocess,
    time,
    urllib,
):
    def start_vllm(model_path, log_path):
        if VLLM["proc"] and VLLM["proc"].poll() is None:
            if VLLM["model"] == str(model_path):
                print(f"reusing server on port {VLLM['port']}")
                return VLLM["port"]
            stop_vllm()

        _py = KIT / ".venv" / "bin" / "python"
        port = free_port()
        env = {**os.environ,
               "PATH": f"{KIT / '.venv' / 'bin'}:{os.environ['PATH']}",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        # Flags copied from eval.sh: --language-model-only skips the vision tower,
        # and the modest memory/seq limits exist because Qwen3.5's GDN/Mamba cache
        # blocks must accommodate max_num_seqs.
        cmd = [str(_py), "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(model_path),
               "--served-model-name", SERVED_MODEL_NAME,
               "--tensor-parallel-size", "1",
               "--dtype", "bfloat16",
               "--max-model-len", str(MAX_MODEL_LEN),
               "--language-model-only",
               "--port", str(port),
               "--gpu-memory-utilization", GPU_MEM_UTIL,
               "--max-num-seqs", MAX_NUM_SEQS]
        log = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        VLLM.update(proc=proc, port=port, model=str(model_path))
        print(f"vLLM PID={proc.pid} port={port}  log={log_path}")

        deadline = time.time() + 1800
        while time.time() < deadline:
            if proc.poll() is not None:
                print(open(log_path).read()[-3000:])
                raise RuntimeError("vLLM exited before becoming ready")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                print("server ready")
                return port
            except Exception:
                time.sleep(10)
        raise RuntimeError("vLLM did not become ready within 30 min")

    def stop_vllm():
        proc = VLLM.get("proc")
        if proc and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), 15)
            try:
                proc.wait(timeout=60)
            except Exception:
                os.killpg(os.getpgid(proc.pid), 9)
            print("vLLM stopped")
        VLLM.update(proc=None, port=None, model=None)

    return start_vllm, stop_vllm


@app.cell
def _(DEC_DIR, mo, serve_btn, start_vllm):
    mo.stop(not serve_btn.value, mo.md("*Server not started.*"))
    PORT = start_vllm(DEC_DIR, "/root/work/vllm_compressed.log")
    return (PORT,)


@app.cell
def _(mo):
    eval_btn = mo.ui.run_button(label="Run all 5 suites (30–120 min)")
    mo.vstack([
        mo.md(
            "### 6. Evaluate\n"
            "Runs the graders' `run_eval.py` per suite against the live server, "
            "with their config: temperature 1.0, top_p 0.95, top_k 20, "
            "presence_penalty 1.5, seed 42, thinking enabled, 32k token budget."
        ),
        eval_btn,
    ])
    return (eval_btn,)


@app.cell
def _(
    ES,
    EVAL_PY,
    KIT,
    MAX_CONCURRENCY,
    MAX_NEW_TOKENS,
    OUT_DIR,
    SUITES_BUILT,
    run_streaming,
    time,
):
    def run_suites(port, tag):
        import yaml

        cfg_base = yaml.safe_load((KIT / "configs" / "eval_config.yaml").read_text())
        rows = []
        for name in ["smoke", "gsm8k", "math500", "amc", "aime"]:
            if name not in SUITES_BUILT:
                continue
            path, n = SUITES_BUILT[name]
            out_json = OUT_DIR / f"{tag}_{name}.json"
            cfg = dict(cfg_base)
            cfg["output"] = str(out_json)
            cfg["vllm_base_url"] = f"http://localhost:{port}/v1"
            tmp = OUT_DIR / f"cfg_{tag}_{name}.yaml"
            tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))

            print(f"\n{'='*70}\n[{tag}] suite '{name}' — {n} problems\n{'='*70}", flush=True)
            t0 = time.time()
            run_streaming(
                [EVAL_PY, "evaluation/run_eval.py",
                 "--config", str(tmp), "--dataset", path, "--port", str(port),
                 "--limit", str(n), "--max-new-tokens", str(MAX_NEW_TOKENS),
                 "--max-concurrency", str(MAX_CONCURRENCY)],
                cwd=KIT,
                quiet_prefixes=("Submitted request", "DEBUG ROW:", "{'problem_idx'"),
            )
            s = ES.score_output(out_json)
            s["suite"] = name
            s["minutes"] = (time.time() - t0) / 60
            rows.append(s)
            print(f"  accuracy={s['accuracy']:.3f}  parse={s['parse_rate']:.3f}  "
                  f"no-boxed={s['truncation_rate']:.3f}  ({s['minutes']:.1f} min)")
        return rows

    return (run_suites,)


@app.cell
def _(PORT, eval_btn, mo, run_suites):
    mo.stop(not eval_btn.value, mo.md("*Waiting.*"))
    COMPRESSED_ROWS = run_suites(PORT, "compressed")
    return (COMPRESSED_ROWS,)


@app.cell
def _(COMPRESSED_ROWS, ES, SIZE_FRAC, mo):
    mo.md(
        f"### Compressed model results\n\n"
        f"`size_frac = {SIZE_FRAC:.4f}`\n\n"
        + ES.format_results_table(COMPRESSED_ROWS)
        + "\n`no-\\boxed` is the fraction of answers with no boxed expression at all — "
        "usually the model ran out of token budget mid-reasoning. If that column climbs "
        "on `aime` while `accuracy` holds elsewhere, the problem is reasoning length, "
        "not arithmetic accuracy, and the two need different fixes."
    )
    return


@app.cell
def _(mo):
    baseline_btn = mo.ui.run_button(label="Free disk, download original, run baseline")
    mo.vstack([
        mo.md(
            "### 7. Baseline\n"
            "Stops the server, deletes the decompressed model to make room, downloads "
            "the original bf16 checkpoint and re-runs the same five suites. The "
            "**delta** is the number worth putting in your report — absolute accuracy "
            "on contaminated public sets says much less."
        ),
        baseline_btn,
    ])
    return (baseline_btn,)


@app.cell
def _(
    BASE_DIR,
    DEC_DIR,
    base_model_id,
    baseline_btn,
    disk_free_gb,
    mo,
    run_suites,
    shutil,
    start_vllm,
    stop_vllm,
):
    mo.stop(not baseline_btn.value, mo.md("*Baseline not run.*"))

    stop_vllm()
    shutil.rmtree(DEC_DIR, ignore_errors=True)
    print(f"freed decompressed model; {disk_free_gb():.0f} GiB free")

    from huggingface_hub import snapshot_download as _snap2

    _snap2(base_model_id.value, local_dir=str(BASE_DIR),
           ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5"])
    _port_b = start_vllm(BASE_DIR, "/root/work/vllm_baseline.log")
    BASELINE_ROWS = run_suites(_port_b, "baseline")
    return (BASELINE_ROWS,)


@app.cell
def _(BASELINE_ROWS, COMPRESSED_ROWS, ES, SIZE_FRAC, mo):
    mo.md(
        f"### Compressed vs baseline\n\n`size_frac = {SIZE_FRAC:.4f}`\n\n"
        + ES.format_comparison_table(COMPRESSED_ROWS, BASELINE_ROWS)
        + "\nRead the ladder, not the average. Damage that appears only on `amc` and "
        "`aime` points at the quantization grid being too coarse for long dependency "
        "chains; damage that reaches `gsm8k` means something is structurally wrong."
    )
    return


@app.cell
def _(mo):
    cleanup_btn = mo.ui.run_button(label="Stop vLLM and free everything")
    cleanup_btn
    return (cleanup_btn,)


@app.cell
def _(BASE_DIR, COMP_DIR, DEC_DIR, cleanup_btn, disk_free_gb, mo, shutil, stop_vllm):
    mo.stop(not cleanup_btn.value, mo.md("*Artifacts kept.*"))
    stop_vllm()
    for _d in (DEC_DIR, BASE_DIR, COMP_DIR):
        shutil.rmtree(_d, ignore_errors=True)
    mo.md(f"Cleaned. {disk_free_gb():.0f} GiB free.")
    return