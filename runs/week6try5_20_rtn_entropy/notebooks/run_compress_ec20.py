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
        # CS6013 — week7 EC-20 (RTN): compress + decompress

        Runs `week7_ec_rtn_20/compress.py` (codec E + Fisher allocation + 0.25% exact
        outliers + numpy rANS, RTN) and `decompress.py` on molab and checks the result.
        In codec_lab2 this exact recipe scored KL 0.033 at size 0.195; step 7 re-measures
        KL on the same held-out traces so the numbers are comparable. Order: **inputs → environment → clone code → download base → SMOKE
        run → FULL run → decompress + verify → upload**.

        **Environment policy (read this).** Nothing is installed except, if
        missing, `huggingface_hub` / `psutil`. No flash-attn, no
        flash-linear-attention, no causal-conv1d, no flashinfer. The pipeline runs
        Gated DeltaNet on transformers' pure-PyTorch path and full attention on
        torch's built-in SDPA, both part of molab's stock torch 2.11 +
        transformers 5.14. `compress.py` forces this even if fla is importable,
        and probes the longest sequence before doing any real work.

        Attach the GPU from the notebook-specs button first. Keep this tab open
        during the full run (≈20–40 min); molab shuts down after 90 idle minutes.
        """
    )
    return


@app.cell
def _(mo):
    code_url = mo.ui.text(
        placeholder="https://github.com/<you>/CS6013/tree/main/<path>/week7_ec_rtn_20",
        label="Code: GitHub tree URL of the week7_ec_rtn_20 folder", full_width=True,
    )
    gh_token = mo.ui.text(placeholder="only if the repo is private",
                          label="GitHub token", kind="password")
    hf_token = mo.ui.text(placeholder="WRITE token (for upload)", label="HF token", kind="password")
    base_model_id = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model")
    out_repo = mo.ui.text(
        placeholder="grey-cat/<roll>-Week07-Compression20-Submission01",
        label="HF model repo for the compressed checkpoint", full_width=True,
    )
    art_repo = mo.ui.text(value="grey-cat/cs6013-w7-ec20-artifacts",
                          label="HF dataset repo for plan/logs", full_width=True)
    mo.vstack([mo.md("### 1. Inputs"), code_url, mo.hstack([gh_token, hf_token], justify="start"),
               base_model_id, out_repo, art_repo])
    return art_repo, base_model_id, code_url, gh_token, hf_token, out_repo


@app.cell
def _():
    import json
    import os
    import re
    import shutil
    import struct
    import subprocess
    import sys
    import time
    from pathlib import Path

    WORK = Path("/root/work") if Path("/root").exists() else Path.home() / "work"
    REPO_DIR = WORK / "code_repo"
    BASE_DIR = WORK / "base_model"
    COMP_DIR = WORK / "w20_compressed"
    SMOKE_DIR = WORK / "w20_smoke"
    DEC_DIR = WORK / "w20_restored"
    LOG_DIR = WORK / "logs"
    for _d in (WORK, LOG_DIR):
        _d.mkdir(parents=True, exist_ok=True)
    PY = sys.executable
    GRADER_TEXT_GIB = 8.0585

    def run_streaming(cmd, cwd=None, env=None, log_file=None, check=True):
        """Run a command, stream its output here, and optionally tee it to a file."""
        merged = {**os.environ, **(env or {})}
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                                env=merged, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        fh = open(log_file, "a") if log_file else None
        for line in proc.stdout:
            print(line, end="")
            if fh:
                fh.write(line)
                fh.flush()
        proc.wait()
        if fh:
            fh.close()
        if check and proc.returncode != 0:
            raise RuntimeError(f"failed ({proc.returncode}): {' '.join(map(str, cmd))}")
        return proc.returncode

    def auth_url(url, token):
        return url.replace("https://", f"https://oauth2:{token}@", 1) if token else url

    def dir_gib(path):
        path = Path(path)
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30 if path.exists() else 0.0

    def is_visual_param(name):
        # Mirror of the graders' measure_checkpoint_bits.py
        n = name.replace("\\", "/").lower()
        if "visual." in n:
            return True
        return any(p in {"visual", "vision", "vision_tower", "vision_model"} for p in n.split("."))

    def scored_gib(directory):
        """Grader-style text bytes, from safetensors headers only."""
        total = 0
        for f in sorted(Path(directory).glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__" and not is_visual_param(k):
                    total += v["data_offsets"][1] - v["data_offsets"][0]
        return total / 2**30

    return (BASE_DIR, COMP_DIR, DEC_DIR, GRADER_TEXT_GIB, LOG_DIR, PY, Path, REPO_DIR,
            SMOKE_DIR, WORK, auth_url, dir_gib, json, os, re, run_streaming, scored_gib,
            shutil, struct, subprocess, sys, time)


@app.cell
def _(PY, WORK, mo, run_streaming, shutil, subprocess, sys):
    # ---- 2. Environment: report only, install nothing heavy ----------------
    def _env():
        import importlib
        lines = [f"- Python `{sys.version.split()[0]}` at `{sys.executable}`"]
        for _pkg in ("huggingface_hub", "psutil"):
            try:
                importlib.import_module(_pkg)
            except ImportError:
                run_streaming([PY, "-m", "pip", "install", "-q", _pkg])
        import torch
        import transformers
        lines.append(f"- torch `{torch.__version__}` (cuda `{torch.version.cuda}`), "
                     f"transformers `{transformers.__version__}`")
        if torch.cuda.is_available():
            _p = torch.cuda.get_device_properties(0)
            lines.append(f"- GPU **{_p.name}**, sm{_p.major}{_p.minor}, {_p.total_memory / 2**30:.0f} GiB")
        else:
            lines.append("- **NO GPU attached — attach one before running compress.**")
        for _mod in ("fla", "causal_conv1d", "flash_attn", "flashinfer"):
            try:
                importlib.import_module(_mod)
                lines.append(f"- `{_mod}` importable — **not used** (compress.py pins torch kernels)")
            except Exception:
                lines.append(f"- `{_mod}` not installed — fine, not needed")
        import psutil
        lines.append(f"- host RAM {psutil.virtual_memory().total / 2**30:.0f} GiB, "
                     f"disk free {shutil.disk_usage(WORK).free / 2**30:.0f} GiB")
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except Exception:
            lines.append("- **git missing**")
        return "\n".join(lines)

    mo.md("### 2. Environment\n\n" + _env())
    return


@app.cell
def _(mo):
    clone_btn = mo.ui.run_button(label="3. Clone code")
    clone_btn
    return (clone_btn,)


@app.cell
def _(REPO_DIR, auth_url, clone_btn, code_url, gh_token, mo, re, run_streaming, shutil):
    mo.stop(not clone_btn.value, mo.md("*Not cloned.*"))
    _m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/tree/([^/]+)/(.+)$",
                  code_url.value.strip().rstrip("/"))
    assert _m, "Code URL must look like https://github.com/<user>/<repo>/tree/<branch>/<path>"
    _owner, _repo, _branch, _sub = _m.groups()
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    run_streaming(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                   "--no-checkout", "--branch", _branch,
                   auth_url(f"https://github.com/{_owner}/{_repo}.git", gh_token.value), REPO_DIR])
    run_streaming(["git", "-C", REPO_DIR, "sparse-checkout", "set", "--cone", _sub])
    run_streaming(["git", "-C", REPO_DIR, "checkout", _branch])
    CODE_DIR = REPO_DIR / _sub
    _need = ["compress.py", "decompress.py", "pyproject.toml", "README.md",
             "compression/__init__.py", "compression/fisher.py", "compression/codec.py",
             "compression/rans.py", "decompression/rans.py",
             "decompression/__init__.py", "compression/calib_data/traces_all.jsonl"]
    _missing = [f for f in _need if not (CODE_DIR / f).is_file()]
    assert not _missing, f"missing in {CODE_DIR}: {_missing}"
    print(f"code at {CODE_DIR} -- layout OK")
    return (CODE_DIR,)


@app.cell
def _(mo):
    dl_btn = mo.ui.run_button(label="4. Download base model")
    dl_btn
    return (dl_btn,)


@app.cell
def _(BASE_DIR, GRADER_TEXT_GIB, base_model_id, dir_gib, dl_btn, hf_token, mo, os, scored_gib):
    mo.stop(not dl_btn.value, mo.md("*Base model not downloaded.*"))
    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value
    from huggingface_hub import snapshot_download as _snap

    _snap(base_model_id.value, local_dir=str(BASE_DIR))
    BASE_TEXT_GIB = scored_gib(BASE_DIR)
    print(f"{base_model_id.value} -> {BASE_DIR} ({dir_gib(BASE_DIR):.2f} GiB on disk)")
    print(f"scored text = {BASE_TEXT_GIB:.4f} GiB (graders use {GRADER_TEXT_GIB})")
    return (BASE_TEXT_GIB,)


@app.cell
def _(mo):
    smoke_btn = mo.ui.run_button(label="5. SMOKE run (4 short Fisher samples, ~10–15 min)")
    mo.vstack([
        mo.md("### 5. Smoke run\nOnly the Fisher pass is shrunk (4 samples x 512 tokens); "
              "planning, rANS encoding and decompression run at full size, so this checks "
              "memory, timing, format and the size budget. Its output is **not** a submission."),
        smoke_btn,
    ])
    return (smoke_btn,)


@app.cell
def _(BASE_DIR, BASE_TEXT_GIB, CODE_DIR, LOG_DIR, PY, SMOKE_DIR, WORK, base_model_id,
      mo, run_streaming, scored_gib, shutil, smoke_btn):
    mo.stop(not smoke_btn.value, mo.md("*Smoke run not started.*"))
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    shutil.rmtree(WORK / f"{SMOKE_DIR.name}_artifacts", ignore_errors=True)
    run_streaming([PY, "compress.py", "--model_name", base_model_id.value,
                   "--checkpoint_path", BASE_DIR, "--output_path", SMOKE_DIR],
                  cwd=CODE_DIR, env={"CS6013_SMOKE": "1", "PYTHONUNBUFFERED": "1"},
                  log_file=LOG_DIR / "smoke_compress.log")
    _dec = WORK / "w20_smoke_restored"
    shutil.rmtree(_dec, ignore_errors=True)
    run_streaming([PY, "decompress.py", "--model_name", base_model_id.value,
                   "--checkpoint_path", SMOKE_DIR, "--output_path", _dec], cwd=CODE_DIR)
    print(f"\nSMOKE size_frac (grader logic) = {scored_gib(SMOKE_DIR) / BASE_TEXT_GIB:.4f}")
    shutil.rmtree(_dec, ignore_errors=True)
    SMOKE_OK = True
    return (SMOKE_OK,)


@app.cell
def _(mo):
    full_btn = mo.ui.run_button(label="6. FULL compress (64 Fisher samples, ~20–40 min)")
    mo.vstack([mo.md("### 6. Full run\nRun only after the smoke run passes. Watch the "
                     "`plan round` lines: the exact entropy must come in under the budget."), full_btn])
    return (full_btn,)


@app.cell
def _(BASE_DIR, CODE_DIR, COMP_DIR, LOG_DIR, PY, SMOKE_OK, WORK, base_model_id, full_btn,
      mo, run_streaming, shutil, time):
    mo.stop(not full_btn.value, mo.md("*Full run not started.*"))
    assert SMOKE_OK
    shutil.rmtree(COMP_DIR, ignore_errors=True)
    _t0 = time.time()
    run_streaming([PY, "compress.py", "--model_name", base_model_id.value,
                   "--checkpoint_path", BASE_DIR, "--output_path", COMP_DIR],
                  cwd=CODE_DIR, env={"CS6013_SMOKE": "0", "PYTHONUNBUFFERED": "1"},
                  log_file=LOG_DIR / "full_compress.log")
    ART_DIR = WORK / f"{COMP_DIR.name}_artifacts"
    print(f"\nfull compress finished in {(time.time() - _t0) / 60:.1f} min")
    return (ART_DIR,)


@app.cell
def _(mo):
    dec_btn = mo.ui.run_button(label="7. Decompress + verify")
    dec_btn
    return (dec_btn,)


@app.cell
def _(ART_DIR, BASE_DIR, BASE_TEXT_GIB, CODE_DIR, COMP_DIR, DEC_DIR, GRADER_TEXT_GIB,
      LOG_DIR, PY, base_model_id, dec_btn, json, mo, run_streaming, scored_gib, shutil, time):
    mo.stop(not dec_btn.value, mo.md("*Not decompressed.*"))
    _frac_meas = scored_gib(COMP_DIR) / BASE_TEXT_GIB
    _frac_grader = scored_gib(COMP_DIR) / GRADER_TEXT_GIB
    print(f"size_frac: {_frac_meas:.4f} (measured denominator) / {_frac_grader:.4f} (8.0585)")
    assert max(_frac_meas, _frac_grader) <= 0.20, "OVER 0.20 -- do not upload"
    _banned = [f.name for f in COMP_DIR.iterdir() if f.suffix.lower() in {".md", ".py", ".ipynb", ".log"}]
    assert not _banned, f"banned files in checkpoint dir: {_banned}"

    shutil.rmtree(DEC_DIR, ignore_errors=True)
    _t_dec = time.time()
    run_streaming([PY, "decompress.py", "--model_name", base_model_id.value,
                   "--checkpoint_path", COMP_DIR, "--output_path", DEC_DIR],
                  cwd=CODE_DIR, log_file=LOG_DIR / "decompress.log")
    print(f"decompress took {(time.time() - _t_dec) / 60:.1f} min")

    # Verification runs in a subprocess so its GPU memory is released afterwards.
    _verify = r'''
import json, sys, glob, torch
from safetensors import safe_open
sys.path.insert(0, sys.argv[4])
base, dec, trace_file = sys.argv[1], sys.argv[2], sys.argv[3]
def keys(d):
    out = {}
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                sl = h.get_slice(k); out[k] = (tuple(sl.get_shape()), sl.get_dtype())
    return out
kb, kd = keys(base), keys(dec)
assert set(kb) == set(kd), f"key mismatch: {sorted(set(kb) ^ set(kd))[:5]}"
bad = [k for k in kb if kb[k] != kd[k]]
assert not bad, f"shape/dtype mismatch: {bad[:5]}"
print(f"[verify] {len(kb)} tensors: same keys, shapes, dtypes as base")

from transformers import AutoModelForCausalLM, AutoTokenizer
from compression.fisher import force_torch_kernels
from compression.calibration import holdout_ids, load_traces
tok = AutoTokenizer.from_pretrained(dec)
def load(p):
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    force_torch_kernels(m); return m
traces = load_traces(trace_file)
_, held = holdout_ids(tok, traces, 4096, 24000)
ids = [x.cuda() for x, _ in held]
starts = [h for _, h in held]
mb = load(base)
with torch.no_grad():
    ref = [mb(input_ids=x, use_cache=False).logits[0, h - 1:-1].float().log_softmax(-1).cpu()
           for x, h in zip(ids, starts)]
del mb; torch.cuda.empty_cache()
md = load(dec)
kl, agree, nb, nq, n = 0.0, 0.0, 0.0, 0.0, 0
with torch.no_grad():
    for x, h, lb in zip(ids, starts, ref):
        lq = md(input_ids=x, use_cache=False).logits[0, h - 1:-1].float().log_softmax(-1).cpu()
        tgt = x[0, h:].cpu()
        kl += (lb.exp() * (lb - lq)).sum().item(); agree += (lb.argmax(-1) == lq.argmax(-1)).sum().item()
        nb -= lb.gather(1, tgt[:, None]).sum().item(); nq -= lq.gather(1, tgt[:, None]).sum().item()
        n += tgt.numel()
print(f"[verify] held-out traces ({n} tokens, same set as codec_lab): KL(base||restored) = "
      f"{kl / n:.4f} nats/token, top-1 = {agree / n * 100:.2f}%, NLL {nb / n:.3f} -> {nq / n:.3f}")
print("[verify] codec_lab2 reference for this recipe (RTN, size 0.195): KL 0.0333, top-1 94.9%")
msgs = [{"role": "user", "content": "What is 17 * 23? Put the answer in \\boxed{}."}]
p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
x = tok(p, return_tensors="pt").input_ids.cuda()
out = md.generate(x, max_new_tokens=60, do_sample=False)
print("[verify] greedy sample:", repr(tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)))
'''
    _vf = LOG_DIR / "verify_restored.py"
    _vf.write_text(_verify)
    run_streaming([PY, _vf, BASE_DIR, DEC_DIR, CODE_DIR / "compression/calib_data/traces_all.jsonl",
                   CODE_DIR], cwd=CODE_DIR, log_file=LOG_DIR / "verify.log")
    _plan = json.loads((ART_DIR / "plan.json").read_text())
    _bpw = [t["bpw"] for t in _plan["tensors"]]
    print(f"\nplan: {len(_bpw)} coded tensors, bits/weight min {min(_bpw):.2f} "
          f"median {sorted(_bpw)[len(_bpw) // 2]:.2f} max {max(_bpw):.2f}")
    VERIFIED = True
    return (VERIFIED,)


@app.cell
def _(mo):
    up_btn = mo.ui.run_button(label="8. Upload checkpoint + artifacts to HF")
    up_btn
    return (up_btn,)


@app.cell
def _(ART_DIR, COMP_DIR, LOG_DIR, VERIFIED, art_repo, hf_token, mo, out_repo, shutil, up_btn):
    mo.stop(not up_btn.value, mo.md("*Not uploaded.*"))
    assert VERIFIED
    assert hf_token.value, "HF write token required"
    from huggingface_hub import HfApi, create_repo

    _api = HfApi(token=hf_token.value)
    create_repo(out_repo.value, repo_type="model", exist_ok=True, token=hf_token.value)
    _api.upload_folder(folder_path=str(COMP_DIR), repo_id=out_repo.value, repo_type="model",
                       commit_message="week7 EC-20 RTN: codec E + Fisher allocation + rANS")
    # Plan, energies, probe, size report and logs go to a SEPARATE dataset repo:
    # the checkpoint repo must contain weights + config/tokenizer only.
    for _f in LOG_DIR.glob("*.log"):
        shutil.copy2(_f, ART_DIR / _f.name)
    create_repo(art_repo.value, repo_type="dataset", exist_ok=True, token=hf_token.value)
    _api.upload_folder(folder_path=str(ART_DIR), repo_id=art_repo.value, repo_type="dataset",
                       commit_message="week7 EC-20 run artifacts")
    print(f"checkpoint -> https://huggingface.co/{out_repo.value}")
    print(f"artifacts  -> https://huggingface.co/datasets/{art_repo.value}")
    return


if __name__ == "__main__":
    app.run()
