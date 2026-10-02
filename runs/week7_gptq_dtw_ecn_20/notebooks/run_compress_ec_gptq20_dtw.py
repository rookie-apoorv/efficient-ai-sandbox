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
        # CS6013 — week8 EC-GPTQ-20-DTW (decision-token-weighted Fisher): compress + decompress

        Runs `week8_ec_gptq_20_dtw/compress.py` and `decompress.py` on molab, checks the result
        and uploads it.

        Method: codec E grid (uniform step x group RMS, no clipping) + Fisher-weighted bits
        per tensor + 0.25% Fisher outliers kept exact + **GPTQ on that grid** + numpy rANS.
        Calibration and Fisher use the base model's own AIME 2022-26 traces; the Fisher loss
        weights decision positions (Wait / But / </think>, sentence starts) x5 (`DECISION_WEIGHT`).

        Order: **inputs → environment → clone code → download base → build calibration
        file → SMOKE run → FULL run → decompress + verify → upload**.

        **Environment policy.** Nothing heavy is installed (only `huggingface_hub`,
        `psutil`, `pyarrow` if missing). No flash-attn / fla / causal-conv1d / flashinfer:
        Gated DeltaNet runs on transformers' pure-PyTorch path, attention on torch SDPA.

        Attach the GPU first. Keep this tab open during the full run (expect 1–3 h);
        molab shuts down after 90 idle minutes.
        """
    )
    return


@app.cell
def _(mo):
    code_url = mo.ui.text(
        placeholder="https://github.com/<you>/CS6013/tree/main/<path>/week8_ec_gptq_20_dtw",
        label="Code: GitHub tree URL of the week8_ec_gptq_20_dtw folder", full_width=True,
    )
    gh_token = mo.ui.text(placeholder="only if the repo is private",
                          label="GitHub token", kind="password")
    hf_token = mo.ui.text(placeholder="WRITE token (reads the private traces repo, uploads)",
                          label="HF token", kind="password")
    traces_repo = mo.ui.text(placeholder="grey-cat/cs6013-qwen35-4b-base-traces",
                             label="HF dataset repo with the base traces (nb1)", full_width=True)
    base_model_id = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model")
    out_repo = mo.ui.text(
        placeholder="grey-cat/<roll>-Week08-Compression20-Submission02",
        label="HF model repo for the compressed checkpoint", full_width=True,
    )
    art_repo = mo.ui.text(value="grey-cat/cs6013-w8-ecgptq20dtw-artifacts",
                          label="HF dataset repo for plan/logs", full_width=True)
    mo.vstack([mo.md("### 1. Inputs"), code_url, mo.hstack([gh_token, hf_token], justify="start"),
               base_model_id, traces_repo, out_repo, art_repo])
    return art_repo, base_model_id, code_url, gh_token, hf_token, out_repo, traces_repo


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
    COMP_DIR = WORK / "w8_20d_compressed"
    SMOKE_DIR = WORK / "w8_20d_smoke"
    DEC_DIR = WORK / "w8_20d_restored"
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
        for _pkg in ("huggingface_hub", "psutil", "pyarrow"):
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
             "compression/gptq_ec.py", "compression/build_calib.py", "compression/rans.py",
             "decompression/rans.py", "decompression/__init__.py"]
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
    rebuild = mo.ui.checkbox(value=False, label="rebuild even if the repo already has aime_calib.npz")
    calib_btn = mo.ui.run_button(label="5. Build calibration file (AIME traces)")
    mo.vstack([mo.md("### 5. Calibration file\nPulls `raw/A/aime_20XX.parquet` + `derived/A.parquet` "
                     "from the traces repo and writes `compression/calib_data/aime_calib.npz` "
                     "(GPTQ set: correct finished traces, <=2 per problem, one 8K runaway head "
                     "for never-solved problems, ~2M tokens; Fisher set: 256 x 4K). "
                     "**Commit this file to GitHub afterwards** -- compress.py reads it and "
                     "never touches the network. If the cloned repo already has it, it is reused."),
               rebuild, calib_btn])
    return calib_btn, rebuild


@app.cell
def _(BASE_DIR, BASE_TEXT_GIB, CODE_DIR, LOG_DIR, PY, calib_btn, hf_token, json, mo,
      rebuild, run_streaming, traces_repo):
    mo.stop(not calib_btn.value, mo.md("*Calibration file not checked.*"))
    assert BASE_TEXT_GIB > 0
    CALIB_PATH = CODE_DIR / "compression/calib_data/aime_calib.npz"
    if CALIB_PATH.exists() and not rebuild.value:
        print(f"using the committed {CALIB_PATH.name}")
    else:
        assert traces_repo.value.strip(), "fill in the traces repo"
        _cmd = [PY, "-m", "compression.build_calib", "--traces_repo", traces_repo.value.strip(),
                "--tokenizer", BASE_DIR, "--out", CALIB_PATH]
        if hf_token.value:
            _cmd += ["--token", hf_token.value]
        run_streaming(_cmd, cwd=CODE_DIR, log_file=LOG_DIR / "build_calib.log")
    _js = CALIB_PATH.with_suffix(".json")
    if _js.exists():
        print(json.dumps(json.loads(_js.read_text()), indent=2))
    return (CALIB_PATH,)


@app.cell
def _(mo):
    smoke_btn = mo.ui.run_button(label="6. SMOKE run (~15–30 min)")
    mo.vstack([
        mo.md("### 6. Smoke run\nFisher shrunk to 4 x 512 tokens and GPTQ calibration to 8K "
              "tokens; planning, GPTQ over all 32 layers, rANS encoding and decompression run "
              "at full size, so this checks memory, format and the size budget. Its output is "
              "**not** a submission."),
        smoke_btn,
    ])
    return (smoke_btn,)


@app.cell
def _(BASE_DIR, BASE_TEXT_GIB, CALIB_PATH, CODE_DIR, LOG_DIR, PY, SMOKE_DIR, WORK,
      base_model_id, mo, run_streaming, scored_gib, shutil, smoke_btn):
    mo.stop(not smoke_btn.value, mo.md("*Smoke run not started.*"))
    assert CALIB_PATH.exists()
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    shutil.rmtree(WORK / f"{SMOKE_DIR.name}_artifacts", ignore_errors=True)
    run_streaming([PY, "compress.py", "--model_name", base_model_id.value,
                   "--checkpoint_path", BASE_DIR, "--output_path", SMOKE_DIR],
                  cwd=CODE_DIR, env={"CS6013_SMOKE": "1", "PYTHONUNBUFFERED": "1"},
                  log_file=LOG_DIR / "smoke_compress.log")
    _dec = WORK / "w8_20d_smoke_restored"
    shutil.rmtree(_dec, ignore_errors=True)
    run_streaming([PY, "decompress.py", "--model_name", base_model_id.value,
                   "--checkpoint_path", SMOKE_DIR, "--output_path", _dec], cwd=CODE_DIR)
    print(f"\nSMOKE size_frac (grader logic) = {scored_gib(SMOKE_DIR) / BASE_TEXT_GIB:.4f}")
    shutil.rmtree(_dec, ignore_errors=True)
    SMOKE_OK = True
    return (SMOKE_OK,)


@app.cell
def _(mo):
    full_btn = mo.ui.run_button(label="7. FULL compress (256 Fisher samples, ~2M-token GPTQ, 1–3 h)")
    mo.vstack([mo.md("### 7. Full run\nRun only after the smoke run passes. Watch the "
                     "`plan round` lines (exact entropy under budget) and the per-layer "
                     "`GPTQ vs plan` drift; re-planning keeps the total on target."), full_btn])
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
    dec_btn = mo.ui.run_button(label="8. Decompress + verify")
    dec_btn
    return (dec_btn,)


@app.cell
def _(ART_DIR, BASE_DIR, BASE_TEXT_GIB, CODE_DIR, COMP_DIR, DEC_DIR, GRADER_TEXT_GIB,
      LOG_DIR, PY, base_model_id, dec_btn, hf_token, json, mo, os, run_streaming, scored_gib,
      shutil, time, traces_repo):
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
import glob, json, os, sys, torch
import pandas as pd
from safetensors import safe_open
sys.path.insert(0, sys.argv[4])
base, dec, traces_repo = sys.argv[1], sys.argv[2], sys.argv[3]
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

# held-out check: base-model traces from module B (HMMT/BRUMO/CMIMC/SMT -- NOT used for
# calibration), seed 42, finished, first 4096 tokens; KL scored on the completion part.
from huggingface_hub import HfApi, hf_hub_download
tok = os.environ.get("HF_TOKEN")
files = [f for f in HfApi(token=tok).list_repo_files(traces_repo, repo_type="dataset")
         if f.startswith("raw/B/") and f.endswith(".parquet")]
df = pd.concat([pd.read_parquet(hf_hub_download(traces_repo, f, repo_type="dataset", token=tok))
                for f in sorted(files)], ignore_index=True)
df = df[(df.seed == 42) & (df.finish_reason == "stop")].sample(frac=1.0, random_state=0)
evals, ntok = [], 0
for r in df.itertuples():
    p, c = list(r.prompt_token_ids), list(r.completion_token_ids)
    ids = (p + c)[:4096]
    if len(ids) - len(p) < 256:
        continue
    evals.append((torch.tensor(ids)[None], len(p)))
    ntok += len(ids) - len(p)
    if ntok >= 24000:
        break

from transformers import AutoModelForCausalLM, AutoTokenizer
from compression.fisher import decision_mask, decision_token_sets, force_torch_kernels
dsets = decision_token_sets(AutoTokenizer.from_pretrained(base))
def load(p):
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    force_torch_kernels(m); return m
mb = load(base)
with torch.no_grad():
    ref = [mb(input_ids=x.cuda(), use_cache=False).logits[0, h - 1:-1].float().log_softmax(-1).cpu()
           for x, h in evals]
del mb; torch.cuda.empty_cache()
md = load(dec)
kl, agree, nb, nq, n = 0.0, 0.0, 0.0, 0.0, 0
kl_d, n_d = 0.0, 0
with torch.no_grad():
    for (x, h), lb in zip(evals, ref):
        lq = md(input_ids=x.cuda(), use_cache=False).logits[0, h - 1:-1].float().log_softmax(-1).cpu()
        tgt = x[0, h:]
        klt = (lb.exp() * (lb - lq)).sum(-1)
        dm = decision_mask(x, h, dsets)
        kl += klt.sum().item(); agree += (lb.argmax(-1) == lq.argmax(-1)).sum().item()
        kl_d += klt[dm].sum().item(); n_d += int(dm.sum())
        nb -= lb.gather(1, tgt[:, None]).sum().item(); nq -= lq.gather(1, tgt[:, None]).sum().item()
        n += tgt.numel()
print(f"[verify] held-out module-B traces ({len(evals)} traces, {n} tokens): KL(base||restored) = "
      f"{kl / n:.5f} nats/token, top-1 = {agree / n * 100:.2f}%, NLL {nb / n:.4f} -> {nq / n:.4f}")
print(f"[verify] KL at decision positions ({n_d / n * 100:.1f}% of tokens): {kl_d / max(n_d, 1):.5f} | "
      f"elsewhere: {(kl - kl_d) / max(n - n_d, 1):.5f} nats/token")
print("[verify] for scale (codec lab, AIME traces, RTN): int4/g128 KL 0.041, week7 codec-E 20% KL 0.033")
tk = AutoTokenizer.from_pretrained(dec)
msgs = [{"role": "user", "content": "What is 17 * 23? Put the answer in \\boxed{}."}]
pr = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
x = tk(pr, return_tensors="pt").input_ids.cuda()
out = md.generate(x, max_new_tokens=60, do_sample=False)
print("[verify] greedy sample:", repr(tk.decode(out[0, x.shape[1]:], skip_special_tokens=True)))
'''
    _vf = LOG_DIR / "verify_restored.py"
    _vf.write_text(_verify)
    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value
    run_streaming([PY, _vf, BASE_DIR, DEC_DIR, traces_repo.value.strip(), CODE_DIR],
                  cwd=CODE_DIR, log_file=LOG_DIR / "verify.log")
    _plan = json.loads((ART_DIR / "plan.json").read_text())
    _bpw = [t["bpw"] for t in _plan["tensors"]]
    print(f"\nplan: {len(_bpw)} coded tensors, bits/weight min {min(_bpw):.2f} "
          f"median {sorted(_bpw)[len(_bpw) // 2]:.2f} max {max(_bpw):.2f}")
    VERIFIED = True
    return (VERIFIED,)


@app.cell
def _(mo):
    up_btn = mo.ui.run_button(label="9. Upload checkpoint + artifacts to HF")
    up_btn
    return (up_btn,)


@app.cell
def _(ART_DIR, CALIB_PATH, COMP_DIR, LOG_DIR, VERIFIED, art_repo, hf_token, mo, out_repo,
      shutil, up_btn):
    mo.stop(not up_btn.value, mo.md("*Not uploaded.*"))
    assert VERIFIED
    assert hf_token.value, "HF write token required"
    from huggingface_hub import HfApi, create_repo

    _api = HfApi(token=hf_token.value)
    create_repo(out_repo.value, repo_type="model", exist_ok=True, token=hf_token.value)
    _api.upload_folder(folder_path=str(COMP_DIR), repo_id=out_repo.value, repo_type="model",
                       commit_message="week8 EC-GPTQ-20-DTW: codec E + Fisher allocation + GPTQ + rANS")
    # Plan, size report, calibration file and logs go to a SEPARATE dataset repo:
    # the checkpoint repo must contain weights + config/tokenizer only.
    for _f in LOG_DIR.glob("*.log"):
        shutil.copy2(_f, ART_DIR / _f.name)
    for _f in (CALIB_PATH, CALIB_PATH.with_suffix(".json")):   # commit these to GitHub too
        if _f.exists():
            shutil.copy2(_f, ART_DIR / _f.name)
    create_repo(art_repo.value, repo_type="dataset", exist_ok=True, token=hf_token.value)
    _api.upload_folder(folder_path=str(ART_DIR), repo_id=art_repo.value, repo_type="dataset",
                       commit_message="week8 EC-GPTQ-20-DTW run artifacts")
    print(f"checkpoint -> https://huggingface.co/{out_repo.value}")
    print(f"artifacts  -> https://huggingface.co/datasets/{art_repo.value}")
    return


if __name__ == "__main__":
    app.run()
