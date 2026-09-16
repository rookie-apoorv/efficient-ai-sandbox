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
        # CS6013 — Compression pipeline runner

        Clones your submission repo, downloads the base checkpoint, runs
        `compress.py`, verifies with `decompress.py`, and uploads the **compressed**
        checkpoint to Hugging Face.

        Every expensive step is behind a **Run** button, so nothing starts by accident
        when a cell above it re-evaluates.

        **Before you start:** click the notebook-specs button in the header and attach
        a GPU. Suggested order:

        1. Fill in the settings below
        2. Environment check
        3. Install dependencies
        4. Sign in to Hugging Face
        5. Clone the repo
        6. **Smoke test on a small model** ← do not skip this
        7. Download the base model
        8. Compress
        9. Decompress + sanity generate
        10. Upload
        """
    )
    return


@app.cell
def _(mo):
    hf_user = mo.ui.text(placeholder="your-hf-username", label="HF username")
    enrollment = mo.ui.text(placeholder="CS24MTECH11001", label="Enrollment no.")
    week_no = mo.ui.text(value="02", label="Week number")
    target_no = mo.ui.text(value="40", label="Compression target")
    submission_no = mo.ui.text(value="01", label="Submission number")

    repo_url = mo.ui.text(
        placeholder="https://github.com/<you>/CS6013.git", label="GitHub repo URL"
    )
    repo_subdir = mo.ui.text(
        placeholder="CS6013/<roll>/Week02/Compression40/Submission01",
        label="Path to submission inside repo",
    )
    base_model = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model ID")

    mo.vstack(
        [
            mo.md("### Settings"),
            mo.hstack([hf_user, enrollment], justify="start"),
            mo.hstack([week_no, target_no, submission_no], justify="start"),
            repo_url,
            repo_subdir,
            base_model,
        ]
    )
    return (
        base_model,
        enrollment,
        hf_user,
        repo_subdir,
        repo_url,
        submission_no,
        target_no,
        week_no,
    )


@app.cell
def _(enrollment, hf_user, mo, submission_no, target_no, week_no):
    # Repo name format is fixed by the handout; getting it wrong is an automatic zero.
    hf_repo_id = (
        f"{hf_user.value}/{enrollment.value}"
        f"-Week{week_no.value}"
        f"-Compression{target_no.value}"
        f"-Submission{submission_no.value}"
    )
    mo.md(f"Target HF repo: **`{hf_repo_id}`**")
    return (hf_repo_id,)


@app.cell
def _():
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    WORK = Path("/root/work") if Path("/root").exists() else Path.home() / "work"
    BASE_DIR = WORK / "base_model"
    COMP_DIR = WORK / "compressed"
    REST_DIR = WORK / "restored"
    REPO_DIR = WORK / "repo"
    WORK.mkdir(parents=True, exist_ok=True)

    def run_streaming(cmd, cwd=None, env=None):
        """Run a command, streaming stdout live so long jobs show progress."""
        merged = {**os.environ, **(env or {})}
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
        return proc.returncode

    def dir_size_gb(path):
        path = Path(path)
        if not path.exists():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30

    return (
        BASE_DIR,
        COMP_DIR,
        Path,
        REPO_DIR,
        REST_DIR,
        WORK,
        dir_size_gb,
        os,
        run_streaming,
        shutil,
        subprocess,
        sys,
    )


@app.cell
def _(WORK, mo, shutil, subprocess, sys):
    def _env_report():
        lines = [f"- Python `{sys.version.split()[0]}`"]
        try:
            import torch as _t

            lines.append(f"- torch `{_t.__version__}`, CUDA available: `{_t.cuda.is_available()}`")
            if _t.cuda.is_available():
                props = _t.cuda.get_device_properties(0)
                lines.append(f"- GPU: **{props.name}**, {props.total_memory / 2**30:.0f} GiB VRAM")
            else:
                lines.append("- **No GPU attached.** Attach one from the notebook specs button.")
        except ImportError:
            lines.append("- torch not installed yet")

        try:
            mem_kb = int(
                subprocess.check_output(["grep", "MemTotal", "/proc/meminfo"]).split()[1]
            )
            lines.append(f"- Host RAM: {mem_kb / 1024**2:.0f} GiB")
        except Exception:
            pass

        usage = shutil.disk_usage(WORK)
        lines.append(
            f"- Disk at `{WORK}`: {usage.free / 2**30:.0f} GiB free "
            f"of {usage.total / 2**30:.0f} GiB"
        )
        if usage.free / 2**30 < 25:
            lines.append(
                "- **Under 25 GiB free.** You need roughly 8 (base) + 3 (compressed) "
                "+ 8 (restored). The notebook deletes the base model after compression "
                "to make room."
            )
        return "\n".join(lines)

    mo.md("### Environment\n" + _env_report())
    return


@app.cell
def _(mo):
    install_btn = mo.ui.run_button(label="Install dependencies")
    install_btn
    return (install_btn,)


@app.cell
def _(install_btn, mo, run_streaming, sys):
    mo.stop(not install_btn.value, mo.md("*Click above to install.*"))

    # molab preloads most of this; installing is quick and idempotent.
    run_streaming(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.51.0", "safetensors>=0.4.3", "accelerate",
         "huggingface_hub", "datasets"]
    )
    print("\ndependencies ready")
    return


@app.cell
def _(mo):
    hf_token = mo.ui.text(
        placeholder="hf_...", label="HF token (write access)", kind="password"
    )
    login_btn = mo.ui.run_button(label="Sign in to Hugging Face")
    mo.vstack([
        mo.md(
            "### Hugging Face\n"
            "Create a **write** token at huggingface.co/settings/tokens. "
            "Paste it below — it is not written to disk by this notebook."
        ),
        hf_token,
        login_btn,
    ])
    return hf_token, login_btn


@app.cell
def _(hf_token, login_btn, mo, os):
    mo.stop(not login_btn.value, mo.md("*Not signed in yet.*"))
    mo.stop(not hf_token.value, mo.md("**Paste a token first.**"))

    from huggingface_hub import HfApi, login, snapshot_download, upload_folder

    os.environ["HF_TOKEN"] = hf_token.value
    login(token=hf_token.value, add_to_git_credential=False)
    _who = HfApi().whoami()
    mo.md(f"Signed in as **{_who['name']}**")
    return HfApi, snapshot_download, upload_folder


@app.cell
def _(mo):
    clone_btn = mo.ui.run_button(label="Clone / update repo")
    clone_btn
    return (clone_btn,)


@app.cell
def _(REPO_DIR, clone_btn, mo, repo_subdir, repo_url, run_streaming):
    mo.stop(not clone_btn.value, mo.md("*Click above to clone.*"))
    mo.stop(not repo_url.value, mo.md("**Fill in the GitHub repo URL first.**"))

    if REPO_DIR.exists():
        run_streaming(["git", "pull"], cwd=REPO_DIR)
    else:
        run_streaming(["git", "clone", "--depth", "1", repo_url.value, str(REPO_DIR)])

    SUB_DIR = REPO_DIR / repo_subdir.value
    assert (SUB_DIR / "compress.py").exists(), f"compress.py not found in {SUB_DIR}"
    assert (SUB_DIR / "decompress.py").exists(), f"decompress.py not found in {SUB_DIR}"

    _calib = SUB_DIR / "compression" / "calib_data" / "math_calib.jsonl"
    print(f"submission dir: {SUB_DIR}")
    print(f"calibration corpus present: {_calib.exists()}")
    if not _calib.exists():
        print("\n  Build it with:  python -m compression.build_calibration_set")
        print("  (run that from the submission dir, then commit the jsonl)")
    return (SUB_DIR,)


@app.cell
def _(mo):
    build_calib_btn = mo.ui.run_button(label="Build calibration corpus (only if missing)")
    mo.vstack([
        mo.md(
            "### Calibration corpus\n"
            "Only needed if the clone did not include `math_calib.jsonl`. "
            "**Commit the result to your repo afterwards** so the graded run is "
            "reproducible without network access."
        ),
        build_calib_btn,
    ])
    return (build_calib_btn,)


@app.cell
def _(SUB_DIR, build_calib_btn, mo, run_streaming, sys):
    mo.stop(not build_calib_btn.value, mo.md("*Skipped.*"))
    run_streaming(
        [sys.executable, "-m", "compression.build_calibration_set"], cwd=SUB_DIR
    )
    return


@app.cell
def _(mo):
    smoke_btn = mo.ui.run_button(label="Run smoke test (small model, ~10 min)")
    mo.vstack([
        mo.md(
            "### Smoke test\n"
            "Runs the full compress → decompress cycle on **Qwen/Qwen3-0.6B**. "
            "This exercises GPTQ, the planner, the codec, sharding and the restore "
            "path end to end in minutes instead of hours. If anything in the "
            "pipeline is broken, you find out here rather than four hours into the "
            "real run."
        ),
        smoke_btn,
    ])
    return (smoke_btn,)


@app.cell
def _(
    Path,
    SUB_DIR,
    WORK,
    mo,
    run_streaming,
    shutil,
    smoke_btn,
    snapshot_download,
    sys,
):
    mo.stop(not smoke_btn.value, mo.md("*Smoke test not run.*"))

    _smoke_base = WORK / "smoke_base"
    _smoke_comp = WORK / "smoke_comp"
    _smoke_rest = WORK / "smoke_rest"
    for _d in (_smoke_comp, _smoke_rest):
        shutil.rmtree(_d, ignore_errors=True)

    print("downloading Qwen/Qwen3-0.6B ...")
    snapshot_download(
        "Qwen/Qwen3-0.6B", local_dir=str(_smoke_base), ignore_patterns=["*.pth", "*.bin"]
    )

    print("\n--- compress ---")
    run_streaming(
        [sys.executable, "compress.py",
         "--model_name", "Qwen/Qwen3-0.6B",
         "--checkpoint_path", str(_smoke_base),
         "--output_path", str(_smoke_comp)],
        cwd=SUB_DIR,
    )

    print("\n--- decompress ---")
    run_streaming(
        [sys.executable, "decompress.py",
         "--model_name", "Qwen/Qwen3-0.6B",
         "--checkpoint_path", str(_smoke_comp),
         "--output_path", str(_smoke_rest)],
        cwd=SUB_DIR,
    )
    print("\nSMOKE TEST PASSED — the pipeline runs end to end.")
    return


@app.cell
def _(mo):
    download_btn = mo.ui.run_button(label="Download base model")
    download_btn
    return (download_btn,)


@app.cell
def _(BASE_DIR, base_model, dir_size_gb, download_btn, mo, snapshot_download):
    mo.stop(not download_btn.value, mo.md("*Click above to download.*"))

    snapshot_download(
        base_model.value,
        local_dir=str(BASE_DIR),
        ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5"],
    )
    print(f"base checkpoint: {BASE_DIR}  ({dir_size_gb(BASE_DIR):.2f} GiB)")
    for _f in sorted(BASE_DIR.iterdir()):
        print(f"  {_f.name}")
    return


@app.cell
def _(mo):
    compress_btn = mo.ui.run_button(label="Run compress.py  (hours)")
    mo.vstack([
        mo.md(
            "### Compress\n"
            "Read the two reports it prints before it starts writing: the bit-width "
            "plan with the projected ratio, and the **GPTQ coverage** number. If "
            "coverage is under 40%, the experts are fused tensors and most of the "
            "model is falling back to RTN — stop and reconsider before burning hours."
        ),
        compress_btn,
    ])
    return (compress_btn,)


@app.cell
def _(BASE_DIR, COMP_DIR, SUB_DIR, base_model, compress_btn, mo, run_streaming, shutil, sys):
    mo.stop(not compress_btn.value, mo.md("*Not started.*"))

    shutil.rmtree(COMP_DIR, ignore_errors=True)
    run_streaming(
        [sys.executable, "compress.py",
         "--model_name", base_model.value,
         "--checkpoint_path", str(BASE_DIR),
         "--output_path", str(COMP_DIR)],
        cwd=SUB_DIR,
    )
    return


@app.cell
def _(COMP_DIR, Path, dir_size_gb, mo):
    def _verify_compressed():
        import json

        cfg_path = COMP_DIR / "compression_config.json"
        if not cfg_path.exists():
            return "Compression has not produced output yet."

        cfg = json.loads(cfg_path.read_text())
        ratio = cfg["achieved_ratio"]
        target = float(cfg["target_ratio"])

        rows = [
            "### Compression result",
            "",
            f"| | |",
            f"|---|---|",
            f"| method | `{cfg['method']}` |",
            f"| group size | {cfg['group_size']} |",
            f"| GPTQ / RTN / lossless tensors | {cfg['n_gptq_tensors']} / "
            f"{cfg['n_rtn_tensors']} / {cfg['n_raw_tensors']} |",
            f"| original | {cfg['original_total_bytes'] / 2**30:.3f} GiB |",
            f"| compressed | {cfg['compressed_total_bytes'] / 2**30:.3f} GiB |",
            f"| **achieved ratio** | **{ratio:.4f}** |",
            f"| directory on disk | {dir_size_gb(COMP_DIR):.3f} GiB |",
            "",
        ]

        if ratio <= 0.40:
            rows.append(f"Ratio {ratio:.4f} is within the 0.40 target.")
        else:
            rows.append(
                f"**Ratio {ratio:.4f} EXCEEDS 0.40.** Lower `TARGET_RATIO` in "
                "`compression/config.py` and re-run."
            )
        if ratio > target + 1e-6:
            rows.append(f"(Note: above the configured budget of {target}.)")

        # The handout forbids docs, code and logs inside the HF checkpoint.
        banned = [
            f.name
            for f in COMP_DIR.iterdir()
            if f.is_file() and (f.suffix.lower() in {".md", ".py", ".ipynb", ".log"})
        ]
        rows.append("")
        rows.append(
            "Checkpoint is clean (no README/code/logs)."
            if not banned
            else f"**Remove these before upload:** {banned}"
        )
        return "\n".join(rows)

    mo.md(_verify_compressed())
    return


@app.cell
def _(mo):
    free_base_btn = mo.ui.run_button(label="Delete base model to free disk")
    mo.vstack([
        mo.md(
            "### Free disk before decompressing\n"
            "`decompress.py` never reads the base checkpoint — that is the whole point "
            "of the design — so it is safe to delete now, and it makes room for the "
            "restored model."
        ),
        free_base_btn,
    ])
    return (free_base_btn,)


@app.cell
def _(BASE_DIR, WORK, free_base_btn, mo, shutil):
    mo.stop(not free_base_btn.value, mo.md("*Base model kept.*"))
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    _usage = shutil.disk_usage(WORK)
    mo.md(f"Deleted. {_usage.free / 2**30:.0f} GiB free.")
    return


@app.cell
def _(mo):
    decompress_btn = mo.ui.run_button(label="Run decompress.py")
    mo.vstack([
        mo.md(
            "### Decompress\n"
            "This is exactly what the TAs will run against your uploaded checkpoint. "
            "It must succeed with no base-model access and no GPU."
        ),
        decompress_btn,
    ])
    return (decompress_btn,)


@app.cell
def _(COMP_DIR, REST_DIR, SUB_DIR, base_model, decompress_btn, dir_size_gb, mo, run_streaming, shutil, sys):
    mo.stop(not decompress_btn.value, mo.md("*Not started.*"))

    shutil.rmtree(REST_DIR, ignore_errors=True)
    run_streaming(
        [sys.executable, "decompress.py",
         "--model_name", base_model.value,
         "--checkpoint_path", str(COMP_DIR),
         "--output_path", str(REST_DIR)],
        cwd=SUB_DIR,
    )
    print(f"\nrestored: {dir_size_gb(REST_DIR):.3f} GiB")
    return


@app.cell
def _(mo):
    sanity_btn = mo.ui.run_button(label="Sanity generate on a math problem")
    mo.vstack([
        mo.md(
            "### Sanity check\n"
            "Loads the restored checkpoint and generates. You are looking for coherent "
            "reasoning, not a correct answer — repeated tokens or garbage means the "
            "quantization broke something."
        ),
        sanity_btn,
    ])
    return (sanity_btn,)


@app.cell
def _(REST_DIR, mo, sanity_btn):
    mo.stop(not sanity_btn.value, mo.md("*Not run.*"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(str(REST_DIR), trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        str(REST_DIR), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    _model.eval()

    _prompt = _tok.apply_chat_template(
        [{"role": "user",
          "content": "Find the number of ordered pairs of positive integers (a, b) "
                     "such that a + b = 1000 and neither a nor b has a zero digit."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    _ids = _tok(_prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        _out = _model.generate(**_ids, max_new_tokens=512, temperature=1.0, do_sample=True)
    _text = _tok.decode(_out[0][_ids.input_ids.shape[1]:], skip_special_tokens=True)

    del _model
    torch.cuda.empty_cache()
    mo.md(f"### Model output\n\n```\n{_text}\n```")
    return


@app.cell
def _(hf_repo_id, mo):
    upload_btn = mo.ui.run_button(label=f"Upload compressed checkpoint")
    mo.vstack([
        mo.md(
            f"### Upload to `{hf_repo_id}`\n"
            "Uploads the **compressed** directory only — never the restored model. "
            "The repo is created public, as the handout requires. Code, notebooks, "
            "markdown and logs are excluded."
        ),
        upload_btn,
    ])
    return (upload_btn,)


@app.cell
def _(COMP_DIR, HfApi, hf_repo_id, mo, upload_btn, upload_folder):
    mo.stop(not upload_btn.value, mo.md("*Not uploaded.*"))
    mo.stop(
        not (COMP_DIR / "compression_config.json").exists(),
        mo.md("**No compressed checkpoint found — run compress first.**"),
    )

    _api = HfApi()
    _api.create_repo(repo_id=hf_repo_id, repo_type="model", private=False, exist_ok=True)

    upload_folder(
        repo_id=hf_repo_id,
        folder_path=str(COMP_DIR),
        repo_type="model",
        ignore_patterns=["*.md", "*.py", "*.ipynb", "*.log", ".git*", "__pycache__/*"],
        commit_message="Compressed checkpoint",
    )

    _files = _api.list_repo_files(repo_id=hf_repo_id)
    mo.md(
        f"Uploaded to **https://huggingface.co/{hf_repo_id}**\n\n"
        "Files in the repo:\n\n"
        + "\n".join(f"- `{f}`" for f in sorted(_files))
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### After uploading

        1. Open the HF repo and confirm it is **public** and contains no README,
           source code or logs.
        2. Submit the link through the Google Form and sign the honour pledge.
        3. Do not modify the GitHub repo or the HF checkpoint afterwards — the handout
           forbids it once submitted.

        Keep this notebook **outside** the graded submission folder. The handout
        requires that only `compress.py` and `decompress.py` sit beside the
        `compression/` and `decompression/` packages, so park it somewhere like
        `CS6013/<roll>/notebooks/`.
        """
    )
    return


if __name__ == "__main__":
    app.run()