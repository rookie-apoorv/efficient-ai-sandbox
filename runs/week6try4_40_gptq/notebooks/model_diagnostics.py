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
        # Qwen3.5-4B — base-model diagnostics

        Everything we need to know about the model **before** deciding how to
        compress it. Three tiers, cheapest first:

        | tier | what | cost |
        |---|---|---|
        | **1** | structure, inventory, byte accounting, budget arithmetic | seconds, header-only |
        | **2** | per-tensor weight statistics + round-trip error **under our actual codec** | minutes, GPU |
        | **3** | per-channel activation energy `E[x²]` = `diag(H)`, from our own traces | ~20 min, GPU |

        The single most important cell is the **byte assertion** in Tier 1: scored
        text bytes must equal `8,652,701,696` exactly. That is the denominator the
        graders divide by. If it does not match to the byte, every `size_frac` we
        compute afterwards is fiction.

        The single most useful cell is the last one, which asks: **does weighting
        error by activation energy actually reorder the planner's choices?** If the
        two rankings agree, the current data-free planner is fine and we should
        spend effort elsewhere. If they disagree, that is the fix worth making.
        """
    )
    return


@app.cell
def _():
    # ----------------------------------------------------------------------
    # CONFIG
    # ----------------------------------------------------------------------
    MODEL_ID = "Qwen/Qwen3.5-4B"
    TRACES_REPO = "grey-cat/cs6013-calib-traces-v1"
    TRACES_FILE = "traces_clean.jsonl"

    # The graders' denominator, from CS6013Fall_ProjectEval/README.md step 3:
    #     size_frac = compressed_text_GB / 8.0585
    # 8.0585 GiB expressed in bytes. Tier 1 asserts against this.
    ORIGINAL_TEXT_BYTES = 8_652_701_696

    # Shipping codec settings (week6try2 / week7 config.py), for reference rows.
    SHIP_BITS = 4
    SHIP_GROUP = 64

    # Tier 2 exploration grid.
    GRID_BITS = (3, 4, 8)
    GRID_GROUPS = (32, 64, 128)
    T2_MAX_ROWS = 512      # row subsample per tensor; raise for accuracy, costs time
    T2_MSE = True          # match config.MSE_CLIPPING

    # Tier 3 activation pass.
    N_ACT_TRACES = 16      # traces pushed through the model
    ACT_SEQLEN = 4096      # truncate each to this many tokens

    OUT_DIR = "diagnostics_out"
    return (
        ACT_SEQLEN,
        GRID_BITS,
        GRID_GROUPS,
        MODEL_ID,
        N_ACT_TRACES,
        ORIGINAL_TEXT_BYTES,
        OUT_DIR,
        SHIP_BITS,
        SHIP_GROUP,
        T2_MAX_ROWS,
        T2_MSE,
        TRACES_FILE,
        TRACES_REPO,
    )


@app.cell
def _():
    # ----------------------------------------------------------------------
    # Dependencies, one at a time so a single failure cannot take out its
    # neighbours (a single `pip install a b c` aborts the whole command).
    #
    # No vLLM here: Tier 3 uses transformers directly, so none of the
    # FlashInfer / cutlass trouble from the trace-generation notebook applies.
    # ----------------------------------------------------------------------
    import subprocess
    import sys

    def pip_install(*pkgs, optional=False):
        for p in pkgs:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", p],
                capture_output=True, text=True,
            )
            ok = r.returncode == 0
            print(f"  {p:<28} {'ok' if ok else ('skipped' if optional else 'FAILED')}")
            if not ok and not optional:
                print("   " + r.stderr.strip()[-600:])

    print("required:")
    pip_install("torch", "transformers", "safetensors", "huggingface_hub", "pandas")

    # Pure-Triton DeltaNet kernels. Without these the 24 Gated-DeltaNet layers
    # fall back to a slow PyTorch path -- correct, just slower. causal-conv1d is
    # deliberately NOT installed: it compiles CUDA C++ and molab has no nvcc.
    print("\noptional (speeds up Tier 3 only):")
    pip_install("flash-linear-attention", optional=True)
    return pip_install, subprocess, sys


@app.cell
def _(pip_install):
    import json
    import math
    import os
    import re
    import time
    from collections import defaultdict

    import pandas as pd
    import torch

    assert pip_install is not None

    print(f"torch          : {torch.__version__}")
    print(f"cuda available : {torch.cuda.is_available()}")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"device         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    import transformers
    print(f"transformers   : {transformers.__version__}")
    return DEV, defaultdict, json, math, os, pd, re, time, torch, transformers


@app.cell
def _(MODEL_ID, os, time):
    # ----------------------------------------------------------------------
    # Fetch the checkpoint once. Tier 1 reads only the safetensors headers, but
    # Tiers 2 and 3 need the actual weights, so pull the whole repo.
    # Set HF_TOKEN first if the repo is gated.
    # ----------------------------------------------------------------------
    from huggingface_hub import snapshot_download

    _t0 = time.time()
    BASE_DIR = snapshot_download(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    print(f"checkpoint at {BASE_DIR}  ({time.time()-_t0:.0f}s)")
    print("\nfiles:")
    for _f in sorted(os.listdir(BASE_DIR)):
        _p = os.path.join(BASE_DIR, _f)
        if os.path.isfile(_p):
            print(f"  {_f:<48} {os.path.getsize(_p)/2**20:10.1f} MiB")
    return BASE_DIR, snapshot_download


@app.cell
def _(json, re):
    # ----------------------------------------------------------------------
    # Helpers. Nothing here touches tensor data.
    # ----------------------------------------------------------------------

    # safetensors on-disk layout: 8-byte little-endian header length, then that
    # many bytes of JSON mapping name -> {dtype, shape, data_offsets}. Reading
    # only this gives us the complete inventory for free.
    _DTYPE_BYTES = {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
    }

    def read_st_header(path):
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        hdr.pop("__metadata__", None)
        return hdr

    def dtype_bytes(dt):
        if dt not in _DTYPE_BYTES:
            raise KeyError(f"unknown safetensors dtype {dt!r}")
        return _DTYPE_BYTES[dt]

    # ------------------------------------------------------------------
    # EXACT mirror of the graders' is_visual_param() in
    # CS6013Fall_ProjectEval/measure_checkpoint_bits.py. Both the substring
    # marker and the dotted-component test, in that order. Any divergence here
    # silently mis-measures the submission, so this is copied, not paraphrased.
    # ------------------------------------------------------------------
    _VIS_PARTS = {"visual", "vision", "vision_tower", "vision_model"}

    def is_visual_param(name):
        n = name.replace("\\", "/").lower()
        if "visual." in n:
            return True
        return any(p in _VIS_PARTS for p in n.split("."))

    def counts_toward_size(name):
        """True if this tensor lands in the scored TEXT numerator/denominator."""
        return not is_visual_param(name)

    # ------------------------------------------------------------------
    # Name -> (layer index, role). Checkpoint keys and module paths differ on
    # this model (module path is model.layers.*, checkpoint keys carry an extra
    # language_model component), so match on the `.layers.N.` infix rather than
    # on any fixed prefix.
    # ------------------------------------------------------------------
    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
    _PARAM_RE = re.compile(r"\.(weight|bias|weight_scale|scale|A_log|dt_bias)$")

    def split_layer(name):
        m = _LAYER_RE.search(name)
        if m:
            return int(m.group(1)), m.group(2)
        return None, name

    def role_of(name):
        """Coarse family key used for the rollup, e.g. 'mlp.gate_proj'."""
        if is_visual_param(name):
            return "VISION (dropped, unscored)"
        if "mtp" in name.split("."):
            _, suf = split_layer(name)
            return "mtp." + re.sub(r"^.*?mtp\.", "", suf).rsplit(".", 1)[0]
        if "embed_tokens" in name:
            return "embed_tokens"
        if "lm_head" in name:
            return "lm_head"
        idx, suf = split_layer(name)
        if idx is None:
            return "ROOT: " + name
        return re.sub(r"\.(weight|bias)$", "", suf)

    def param_kind(name):
        m = _PARAM_RE.search(name)
        return m.group(1) if m else "(none)"

    return (
        counts_toward_size,
        dtype_bytes,
        is_visual_param,
        param_kind,
        read_st_header,
        role_of,
        split_layer,
    )


@app.cell
def _(
    BASE_DIR,
    counts_toward_size,
    dtype_bytes,
    os,
    param_kind,
    pd,
    read_st_header,
    role_of,
    split_layer,
):
    # ======================================================================
    # TIER 1a — complete tensor inventory, header-only
    # ======================================================================
    _rows = []
    for _fn in sorted(os.listdir(BASE_DIR)):
        if not _fn.endswith(".safetensors"):
            continue
        for _name, _info in read_st_header(os.path.join(BASE_DIR, _fn)).items():
            _shape = _info["shape"]
            _numel = 1
            for _d in _shape:
                _numel *= _d
            _li, _ = split_layer(_name)
            _rows.append({
                "name": _name,
                "file": _fn,
                "dtype": _info["dtype"],
                "shape": tuple(_shape),
                "ndim": len(_shape),
                "numel": _numel,
                "bytes": _numel * dtype_bytes(_info["dtype"]),
                "layer": _li,
                "role": role_of(_name),
                "kind": param_kind(_name),
                "scored": counts_toward_size(_name),
            })

    T = pd.DataFrame(_rows).sort_values(["file", "name"]).reset_index(drop=True)
    print(f"tensors in checkpoint : {len(T)}")
    print(f"total params          : {T.numel.sum()/1e9:.4f} B")
    print(f"total bytes           : {T.bytes.sum()/2**30:.4f} GiB")
    print("\nby dtype:")
    print(T.groupby("dtype").agg(n=("name", "size"),
                                 params=("numel", "sum"),
                                 GiB=("bytes", lambda s: s.sum() / 2**30)).round(4))
    return (T,)


@app.cell
def _(ORIGINAL_TEXT_BYTES, T):
    # ======================================================================
    # TIER 1b — scored vs unscored, and THE assertion
    # ======================================================================
    _text = int(T[T.scored].bytes.sum())
    _vis = int(T[~T.scored].bytes.sum())

    print(f"scored TEXT bytes   : {_text:,}  ({_text/2**30:.6f} GiB)")
    print(f"unscored VISION     : {_vis:,}  ({_vis/2**30:.6f} GiB)")
    print(f"harness constant    : {ORIGINAL_TEXT_BYTES:,}  "
          f"({ORIGINAL_TEXT_BYTES/2**30:.6f} GiB)")
    _delta = _text - ORIGINAL_TEXT_BYTES
    print(f"difference          : {_delta:,} bytes")

    if _delta == 0:
        print("\n  MATCH. Our denominator is the graders' denominator, byte for byte.")
        print("  Every size_frac computed below is the number they will compute.")
    else:
        print("\n  *** MISMATCH ***")
        print("  Either is_visual_param() has drifted, or this checkpoint revision")
        print("  differs from the one the harness constant was derived from.")
        print("  STOP and resolve this before trusting any budget number.")

    TEXT_BYTES = _text
    VISION_BYTES = _vis
    BYTES_MATCH = (_delta == 0)
    return BYTES_MATCH, TEXT_BYTES, VISION_BYTES


@app.cell
def _(T, TEXT_BYTES):
    # ======================================================================
    # TIER 1c — family rollup: where do the bytes actually live?
    # ======================================================================
    _g = (T.groupby("role")
            .agg(n=("name", "size"),
                 params=("numel", "sum"),
                 bytes=("bytes", "sum"),
                 scored=("scored", "first"))
            .sort_values("bytes", ascending=False))
    _g["params_B"] = (_g["params"] / 1e9).round(4)
    _g["MiB"] = (_g["bytes"] / 2**20).round(1)
    _g["pct_of_text"] = (_g["bytes"] / TEXT_BYTES * 100).round(2)
    FAMILIES = _g[["n", "params_B", "MiB", "pct_of_text", "scored"]]

    print("Where the bytes live (pct is of the SCORED text tower):\n")
    print(FAMILIES.to_string())
    print("\nRead this table for: which few families dominate the budget, and")
    print("therefore which ones are worth arguing about at all.")
    return (FAMILIES,)


@app.cell
def _(BASE_DIR, T, json, os, pd):
    # ======================================================================
    # TIER 1d — per-layer structure. Which layers are linear vs full attention?
    # ======================================================================
    _cfg = json.load(open(os.path.join(BASE_DIR, "config.json")))
    TCFG = _cfg.get("text_config", _cfg)

    print("architecture:")
    for _k in ("architectures", "model_type", "num_hidden_layers", "hidden_size",
               "intermediate_size", "vocab_size", "tie_word_embeddings",
               "head_dim", "partial_rotary_factor", "full_attention_interval",
               "num_attention_heads", "num_key_value_heads",
               "linear_num_value_heads", "linear_num_key_heads",
               "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"):
        _v = _cfg.get(_k, TCFG.get(_k, "-"))
        if _v != "-":
            print(f"  {_k:<26} {_v}")

    LAYER_TYPES = TCFG.get("layer_types") or _cfg.get("layer_types")
    if LAYER_TYPES:
        _ln = [(i, t) for i, t in enumerate(LAYER_TYPES)]
        _full = [i for i, t in _ln if "full" in t]
        print(f"\n  layer_types: {len(LAYER_TYPES)} layers, "
              f"{len(_full)} full-attention at indices {_full}")
        print(f"               {len(LAYER_TYPES)-len(_full)} linear-attention (Gated DeltaNet)")
    else:
        print("\n  layer_types absent from config -- inferring from tensor names below")

    # Per-layer tensor census, so we can see the two block shapes side by side.
    _lay = T[T.layer.notna()].copy()
    _lay["layer"] = _lay["layer"].astype(int)
    _per = (_lay.groupby("layer")
                .agg(n_tensors=("name", "size"), params=("numel", "sum"),
                     MiB=("bytes", lambda s: s.sum() / 2**20)))
    _per["type"] = [LAYER_TYPES[i] if LAYER_TYPES and i < len(LAYER_TYPES) else "?"
                    for i in _per.index]
    _per["params_M"] = (_per["params"] / 1e6).round(2)
    PER_LAYER = _per[["type", "n_tensors", "params_M", "MiB"]].round(2)
    print("\nper-layer census:\n")
    print(PER_LAYER.to_string())

    print("\nrole sets by layer type (what tensors a block of each kind contains):")
    for _t in sorted(set(PER_LAYER["type"])):
        _idxs = [i for i in PER_LAYER.index if PER_LAYER.loc[i, "type"] == _t]
        if not _idxs:
            continue
        _roles = sorted(set(_lay[_lay.layer == _idxs[0]].role))
        print(f"\n  {_t}  (e.g. layer {_idxs[0]}, {len(_roles)} roles)")
        for _r in _roles:
            _row = _lay[(_lay.layer == _idxs[0]) & (_lay.role == _r)].iloc[0]
            print(f"     {_r:<34} {str(_row['shape']):<20} {_row['numel']/1e6:8.2f} M")
    return LAYER_TYPES, PER_LAYER, TCFG


@app.cell
def _(SHIP_GROUP, T, TEXT_BYTES):
    # ======================================================================
    # TIER 1e — quantizability triage under the CURRENT rules
    #
    #   config.MIN_NUMEL = 1 << 20, 2-D only, last dim usable by the codec.
    #
    # Everything that fails these lands in `fixed_bytes` and is stored at full
    # bf16 -- which is where the surprises usually are.
    # ======================================================================
    MIN_NUMEL = 1 << 20

    # NB: bracket indexing, not attribute access. In a row-wise apply the row is
    # a Series, and `r.shape` / `r.ndim` would silently resolve to the Series'
    # OWN attributes instead of our columns.
    def _bucket(r):
        if not r["scored"]:
            return "unscored (vision, dropped -> zeros)"
        if r["ndim"] != 2:
            return "not 2-D (norms, biases, conv1d, A_log ...)"
        if r["numel"] < MIN_NUMEL:
            return f"2-D but < {MIN_NUMEL/1e6:.1f}M elements"
        if r["shape"][-1] % SHIP_GROUP:
            return f"2-D, large, but cols % {SHIP_GROUP} != 0"
        return "QUANTIZABLE candidate"

    _t = T.copy()
    _t["bucket"] = _t.apply(_bucket, axis=1)
    TRIAGE = (_t.groupby("bucket")
                .agg(n=("name", "size"), params=("numel", "sum"),
                     bytes=("bytes", "sum"))
                .sort_values("bytes", ascending=False))
    TRIAGE["params_B"] = (TRIAGE["params"] / 1e9).round(4)
    TRIAGE["MiB"] = (TRIAGE["bytes"] / 2**20).round(1)
    TRIAGE["pct_of_text"] = (TRIAGE["bytes"] / TEXT_BYTES * 100).round(2)

    print("Quantizability triage:\n")
    print(TRIAGE[["n", "params_B", "MiB", "pct_of_text"]].to_string())

    CANDIDATES = _t[_t.bucket == "QUANTIZABLE candidate"].copy()
    FIXED = _t[(_t.scored) & (_t.bucket != "QUANTIZABLE candidate")]
    print(f"\ncandidates        : {len(CANDIDATES):>5}  "
          f"{CANDIDATES.numel.sum()/1e9:.4f} B params  "
          f"{CANDIDATES.bytes.sum()/2**30:.4f} GiB")
    print(f"scored but FIXED  : {len(FIXED):>5}  "
          f"{FIXED.numel.sum()/1e9:.4f} B params  "
          f"{FIXED.bytes.sum()/2**30:.4f} GiB  <-- stored at bf16, full price")

    print("\nLargest scored-but-fixed tensors (these are pure budget leaks):")
    print(FIXED.nlargest(12, "bytes")[["name", "shape", "numel", "bytes"]].to_string(index=False))
    return CANDIDATES, FIXED, MIN_NUMEL, TRIAGE


@app.cell
def _(CANDIDATES, FIXED, GRID_BITS, GRID_GROUPS, ORIGINAL_TEXT_BYTES, math, pd):
    # ======================================================================
    # TIER 1f — the budget table. What is affordable, before running anything.
    #
    # Cost per weight = bits + (16 + 8)/group
    #                   ^codes  ^fp16 scale + uint8 zero-point, per group per row
    #
    # `packed` is the honest cost for bits in {4, 8} (the shipping packer emits
    # whole bytes, two 4-bit codes per byte). For 3-bit the packer does not
    # exist yet, so that row is the idealised bit cost -- treat it as a lower
    # bound, not a promise.
    # ======================================================================
    def plan_bytes(df, bits, group, perm=False):
        total = 0
        for _r in df.itertuples():
            rows = _r.numel // _r.shape[-1]
            cols = _r.shape[-1]
            ng = math.ceil(cols / group)
            if bits == 4:
                code = rows * ((cols + 1) // 2)
            elif bits == 8:
                code = rows * cols
            else:
                code = math.ceil(rows * cols * bits / 8)
            total += code + rows * ng * 3          # fp16 scale + uint8 zp
            if perm:
                total += cols * 4                  # act_order permutation, int32
        return total

    _fixed = int(FIXED.bytes.sum())
    _grid = []
    for _b in GRID_BITS:
        for _g in GRID_GROUPS:
            _q = plan_bytes(CANDIDATES, _b, _g)
            _qp = plan_bytes(CANDIDATES, _b, _g, perm=True)
            _grid.append({
                "bits": _b, "group": _g,
                "bits_per_wt": round(_b + 24 / _g, 4),
                "quant_GiB": round(_q / 2**30, 4),
                "fixed_GiB": round(_fixed / 2**30, 4),
                "size_frac": round((_q + _fixed) / ORIGINAL_TEXT_BYTES, 4),
                "size_frac_with_perm": round((_qp + _fixed) / ORIGINAL_TEXT_BYTES, 4),
            })
    BUDGET = pd.DataFrame(_grid)

    print("Uniform-codec budget table (all candidates at the same setting):\n")
    print(BUDGET.to_string(index=False))

    print("\nHeadroom against each target, using the shipping 4-bit/64 baseline:")
    _base = BUDGET[(BUDGET.bits == 4) & (BUDGET.group == 64)].iloc[0]["size_frac"]
    _hi = BUDGET[(BUDGET.bits == 8) & (BUDGET.group == 64)].iloc[0]["size_frac"]
    for _tgt in (0.40, 0.20, 0.10):
        _spare = (_tgt - _base) * ORIGINAL_TEXT_BYTES
        _frac_to8 = (_tgt - _base) / (_hi - _base) if _hi > _base else float("nan")
        print(f"  target {_tgt:.2f}:  int4/g64 costs {_base:.4f}  ->  "
              f"{_spare/2**30:+.3f} GiB spare  "
              f"= {_frac_to8*100:5.1f}% of candidate bytes promotable to int8")
    print("\nA negative 'spare' means uniform int4/g64 already misses that target,")
    print("so the 20% and 10% runs need a fundamentally cheaper codec, not a planner.")
    return BUDGET, plan_bytes


@app.cell
def _(
    BASE_DIR,
    CANDIDATES,
    DEV,
    GRID_BITS,
    GRID_GROUPS,
    T2_MAX_ROWS,
    T2_MSE,
    os,
    pd,
    time,
    torch,
):
    # ======================================================================
    # TIER 2 — weight statistics and round-trip error UNDER OUR ACTUAL CODEC
    #
    # This is the part that replaces the planner's `estimate_error`. Two things
    # the current estimator gets wrong and this does not:
    #
    #   1. It calls find_qparams(mse=False) -- plain min/max -- while the real
    #      encoder runs with MSE_CLIPPING=True. Clipping helps 4-bit far more
    #      than 8-bit (16 levels vs 256), so the data-free proxy systematically
    #      OVERSTATES how bad int4 is and skews the promotion ranking.
    #   2. It only ever measures the shipping group size.
    #
    # find_qparams below is week6try2's, generalised to arbitrary bit widths
    # and vectorised over all groups at once (the original loops groups, which
    # is far too slow to sweep a grid).
    # ======================================================================
    from safetensors import safe_open

    def find_qparams_v(block, bits, mse=T2_MSE, grid=100, max_shrink=0.8, norm=2.4):
        """block: [N, g] float32. Returns (scale, zp), each [N]."""
        maxq = 2 ** bits - 1
        x_min = block.min(dim=1).values.clamp(max=0.0)
        x_max = block.max(dim=1).values.clamp(min=0.0)
        deg = (x_min == 0) & (x_max == 0)
        x_min = torch.where(deg, torch.full_like(x_min, -1.0), x_min)
        x_max = torch.where(deg, torch.full_like(x_max, 1.0), x_max)

        def build(lo, hi):
            s = ((hi - lo) / maxq).to(torch.float16).float()
            s = torch.where(s > 0, s, torch.ones_like(s))
            z = torch.round(-lo / s).clamp(0, maxq)
            return s, z

        scale, zp = build(x_min, x_max)
        if mse:
            best = torch.full_like(x_min, float("inf"))
            for step in range(int(max_shrink * grid)):
                sh = 1.0 - step / grid
                cs, cz = build(sh * x_min, sh * x_max)
                codes = torch.clamp(torch.round(block / cs[:, None]) + cz[:, None], 0, maxq)
                recon = (codes - cz[:, None]) * cs[:, None]
                err = ((recon - block).abs() ** norm).sum(dim=1)
                better = err < best
                best = torch.where(better, err, best)
                scale = torch.where(better, cs, scale)
                zp = torch.where(better, cz, zp)
        return scale, zp

    def rtn_roundtrip(W, bits, group, mse=T2_MSE):
        """Fake-quantize [rows, cols] with our codec. Returns dequantized W."""
        rows, cols = W.shape
        pad = (-cols) % group
        Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
        ng = Wp.shape[1] // group
        blk = Wp.reshape(rows * ng, group)
        s, z = find_qparams_v(blk, bits, mse=mse)
        maxq = 2 ** bits - 1
        codes = torch.clamp(torch.round(blk / s[:, None]) + z[:, None], 0, maxq)
        deq = ((codes - z[:, None]) * s[:, None]).reshape(rows, ng * group)
        return deq[:, :cols]

    _stats = []
    _t0 = time.time()
    _files = sorted(set(CANDIDATES.file))

    for _fn in _files:
        _sub = CANDIDATES[CANDIDATES.file == _fn]
        with safe_open(os.path.join(BASE_DIR, _fn), framework="pt", device="cpu") as _h:
            for _r in _sub.itertuples():
                W = _h.get_tensor(_r.name).float()
                rows, cols = W.shape
                if rows > T2_MAX_ROWS:
                    W = W[torch.linspace(0, rows - 1, T2_MAX_ROWS).long()]
                W = W.to(DEV)
                nrm = W.norm().item()
                absW = W.abs()
                std = W.std().item()
                rec = {
                    "name": _r.name, "role": _r.role, "layer": _r.layer,
                    "shape": _r.shape, "numel": _r.numel,
                    "std": std,
                    "max_abs": absW.max().item(),
                    "kurtosis_proxy": absW.max().item() / max(std, 1e-12),
                    "frac_beyond_4sigma": (absW > 4 * std).float().mean().item(),
                    "frac_near_zero": (absW < 0.01 * std).float().mean().item(),
                }
                # per-group dynamic-range spread at the shipping group size:
                # how much the required step size varies group to group. High
                # spread => group-wise scaling is doing a lot of work, and
                # tightening the group will pay.
                _g0 = 64
                _pad = (-cols) % _g0
                _Wp = torch.nn.functional.pad(W, (0, _pad)) if _pad else W
                _blk = _Wp.reshape(W.shape[0] * (_Wp.shape[1] // _g0), _g0)
                _rng = (_blk.max(dim=1).values - _blk.min(dim=1).values)
                rec["grp_range_p50"] = _rng.median().item()
                rec["grp_range_p99_over_p50"] = (
                    torch.quantile(_rng.float(), 0.99).item() / max(_rng.median().item(), 1e-12)
                )
                for _b in GRID_BITS:
                    for _g in GRID_GROUPS:
                        _dq = rtn_roundtrip(W, _b, _g)
                        rec[f"relerr_b{_b}_g{_g}"] = ((_dq - W).norm() / max(nrm, 1e-12)).item()
                _stats.append(rec)
                del W
        print(f"  {_fn}: {len(_sub)} tensors  ({time.time()-_t0:.0f}s)", flush=True)

    W2 = pd.DataFrame(_stats)
    print(f"\nTier 2 complete: {len(W2)} tensors in {(time.time()-_t0)/60:.1f} min")
    return W2, find_qparams_v, rtn_roundtrip, safe_open


@app.cell
def _(GRID_BITS, GRID_GROUPS, W2):
    # ======================================================================
    # TIER 2 report
    # ======================================================================
    print("Mean round-trip relative error by role and codec setting")
    print("(this is what the planner SHOULD be ranking on, data-free version)\n")
    _cols = [f"relerr_b{b}_g{g}" for b in GRID_BITS for g in GRID_GROUPS]
    print(W2.groupby("role")[_cols].mean().round(5).to_string())

    print("\n\nHardest tensors at the shipping setting (int4 / group 64):")
    _s = W2.sort_values("relerr_b4_g64", ascending=False)
    print(_s[["name", "role", "numel", "relerr_b4_g64", "relerr_b8_g64",
              "kurtosis_proxy", "grp_range_p99_over_p50"]].head(20).to_string(index=False))

    print("\nEasiest tensors (candidates to push BELOW 4 bits at the 20%/10% targets):")
    print(_s[["name", "role", "numel", "relerr_b4_g64", "relerr_b3_g64"]]
          .tail(15).to_string(index=False))

    print("\n\nMarginal value of the two moves available to the planner, per role:")
    print("  tighten  = relerr(b4,g64) -> relerr(b4,g32)   costs 0.375 bits/weight")
    print("  promote  = relerr(b4,g64) -> relerr(b8,g64)   costs 4.0   bits/weight\n")
    _m = W2.groupby("role").agg(
        e4_64=("relerr_b4_g64", "mean"),
        e4_32=("relerr_b4_g32", "mean"),
        e8_64=("relerr_b8_g64", "mean"),
    )
    _m["gain_per_bit_tighten"] = ((_m.e4_64 - _m.e4_32) / 0.375).round(5)
    _m["gain_per_bit_promote"] = ((_m.e4_64 - _m.e8_64) / 4.0).round(5)
    _m["tighten_wins"] = _m.gain_per_bit_tighten > _m.gain_per_bit_promote
    print(_m.round(5).to_string())
    print("\nWherever tighten_wins is True, spending budget on a smaller group is")
    print("strictly more efficient than promoting to int8 -- and the current")
    print("planner cannot make that move at all.")
    return


@app.cell
def _(
    ACT_SEQLEN,
    BASE_DIR,
    DEV,
    MODEL_ID,
    N_ACT_TRACES,
    TRACES_FILE,
    TRACES_REPO,
    json,
    os,
    torch,
):
    # ======================================================================
    # TIER 3a — load the model and our own calibration traces
    #
    # transformers, not vLLM: we need module-level hooks, and none of the
    # FlashInfer/cutlass trouble from the generation notebook applies here.
    # ======================================================================
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tp = hf_hub_download(TRACES_REPO, TRACES_FILE, repo_type="dataset",
                          token=os.environ.get("HF_TOKEN"))
    TRACES = [json.loads(l) for l in open(_tp)]
    print(f"loaded {len(TRACES)} traces from {TRACES_REPO}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEV)
    model.eval()
    model.config.use_cache = False
    print(f"model loaded; {sum(p.numel() for p in model.parameters())/1e9:.3f} B "
          "params instantiated (text tower only -- vision is never materialised)")

    # Reconstruct what the model actually saw at generation time: the chat
    # template applied to the prompt, followed by its own trace. The Hessian
    # must be estimated on THIS distribution, not on the bare problem text.
    def build_act_batch(rec):
        try:
            head = tok.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            head = tok.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                tokenize=False, add_generation_prompt=True)
        ids = tok(head + rec["trace"], return_tensors="pt").input_ids
        return ids[:, :ACT_SEQLEN]

    _longest = sorted(TRACES, key=lambda r: -r.get("n_tokens", 0))[:N_ACT_TRACES]
    ACT_BATCHES = [build_act_batch(r) for r in _longest]
    print(f"prepared {len(ACT_BATCHES)} activation batches, "
          f"{sum(b.shape[1] for b in ACT_BATCHES):,} tokens total "
          f"(longest traces first -- they exercise the DeltaNet recurrence hardest)")
    return (
        ACT_BATCHES,
        AutoModelForCausalLM,
        AutoTokenizer,
        TRACES,
        build_act_batch,
        hf_hub_download,
        model,
        tok,
    )


@app.cell
def _(ACT_BATCHES, DEV, defaultdict, model, time, torch):
    # ======================================================================
    # TIER 3b — per-input-channel activation energy
    #
    # For a linear y = W x, GPTQ's objective is ||Wx - Ŵx||^2, whose Hessian is
    # H = 2 E[x xᵀ]. Its DIAGONAL, E[x_j^2], is the per-input-channel energy:
    # exactly how much an error in column j of W actually costs at the output.
    #
    # This is what the planner is missing. It is also already computed and then
    # thrown away inside the existing GPTQ pass.
    #
    # We also fingerprint each linear's input so we can detect SIBLINGS --
    # modules fed the identical tensor, which today build bit-identical
    # Hessians separately.
    # ======================================================================
    _acc = defaultdict(lambda: None)
    _cnt = defaultdict(int)
    _fp = {}
    _handles = []

    def _mk(name):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs.get("input")
            if not torch.is_tensor(x):
                return None
            xf = x.detach().reshape(-1, x.shape[-1]).float()
            s = (xf * xf).sum(dim=0)
            if _acc[name] is None:
                _acc[name] = s
            else:
                _acc[name] += s
            _cnt[name] += xf.shape[0]
            if name not in _fp:
                # cheap content fingerprint of the first batch's input
                _fp[name] = (tuple(x.shape),
                             round(float(xf[0, :16].sum()), 6),
                             round(float(xf.sum()), 3))
            return None
        return hook

    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Linear):
            _handles.append(_m.register_forward_pre_hook(_mk(_n), with_kwargs=True))
    print(f"hooked {len(_handles)} nn.Linear modules")

    _t0 = time.time()
    with torch.no_grad():
        for _i, _b in enumerate(ACT_BATCHES):
            model(_b.to(DEV), use_cache=False)
            print(f"  trace {_i+1}/{len(ACT_BATCHES)}  "
                  f"{_b.shape[1]:>6} tok  ({time.time()-_t0:.0f}s)", flush=True)
    for _h in _handles:
        _h.remove()

    HESS_DIAG = {k: (v / max(_cnt[k], 1)).cpu() for k, v in _acc.items() if v is not None}
    INPUT_FP = dict(_fp)
    print(f"\ncollected E[x^2] for {len(HESS_DIAG)} linears "
          f"in {(time.time()-_t0)/60:.1f} min")
    return HESS_DIAG, INPUT_FP


@app.cell
def _(HESS_DIAG, INPUT_FP, defaultdict, pd, torch):
    # ======================================================================
    # TIER 3c — what the activations look like, and who shares an input
    # ======================================================================
    _rows = []
    for _n, _h in HESS_DIAG.items():
        _hs = _h.sort(descending=True).values
        _tot = _hs.sum().clamp_min(1e-30)
        _rows.append({
            "module": _n,
            "in_features": _h.numel(),
            "mean_x2": _h.mean().item(),
            "max_x2": _h.max().item(),
            "max_over_mean": (_h.max() / _h.mean().clamp_min(1e-30)).item(),
            "top1pct_energy": (_hs[: max(1, _h.numel() // 100)].sum() / _tot).item(),
            "top10_energy": (_hs[:10].sum() / _tot).item(),
        })
    ACT = pd.DataFrame(_rows)

    print("Activation-energy concentration per linear input")
    print("(top1pct_energy near 1.0 = a handful of channels carry everything,")
    print(" which is the classic transformer outlier-channel signature and the")
    print(" thing that makes plain weight-MSE ranking wrong)\n")
    print(ACT.sort_values("top1pct_energy", ascending=False)
             .head(20).round(4).to_string(index=False))
    print("\nLeast concentrated:")
    print(ACT.sort_values("top1pct_energy").head(10).round(4).to_string(index=False))

    # ---- siblings: modules that received a bit-identical input tensor -------
    _groups = defaultdict(list)
    for _n, _f in INPUT_FP.items():
        _groups[_f].append(_n)
    SIBLINGS = {k: v for k, v in _groups.items() if len(v) > 1}
    _dup = sum(len(v) - 1 for v in SIBLINGS.values())
    print(f"\n\nSibling groups (identical input => identical Hessian): "
          f"{len(SIBLINGS)} groups, {_dup} redundant Hessian builds")
    for _k, _v in list(SIBLINGS.items())[:6]:
        print("   " + "  |  ".join(n.split("layers.")[-1] for n in _v))
    print("\nEvery redundant build is wasted x^T x work at zero accuracy benefit.")
    return ACT, SIBLINGS


@app.cell
def _(
    DEV,
    HESS_DIAG,
    SHIP_BITS,
    T2_MAX_ROWS,
    model,
    pd,
    rtn_roundtrip,
    torch,
):
    # ======================================================================
    # TIER 3d — THE QUESTION
    #
    # Does weighting the error by activation energy actually change WHICH
    # tensors the planner promotes? If the two rankings agree, the data-free
    # planner is fine and Hessian-aware planning is not worth the restructure.
    # If they disagree, that is the fix worth making.
    #
    # For each linear we compute, at the shipping codec:
    #     plain :  || dW ||^2                       (what the planner uses today)
    #     hess  :  sum_j  E[x_j^2] * || dW[:, j] ||^2   (what GPTQ actually minimises)
    # then rank by error reduction per extra byte -- the planner's own criterion.
    # ======================================================================
    _mods = dict(model.named_modules())
    _rows = []
    for _n, _h in HESS_DIAG.items():
        _m = _mods.get(_n)
        if _m is None or not hasattr(_m, "weight") or _m.weight.dim() != 2:
            continue
        W = _m.weight.detach().float()
        rows, cols = W.shape
        if _h.numel() != cols:
            continue                      # shape mismatch, skip rather than guess
        if rows > T2_MAX_ROWS:
            W = W[torch.linspace(0, rows - 1, T2_MAX_ROWS).long()]
        W = W.to(DEV)
        h = _h.to(DEV)

        d_lo = rtn_roundtrip(W, SHIP_BITS, 64) - W
        d_hi = rtn_roundtrip(W, 8, 64) - W
        col_lo = (d_lo * d_lo).sum(dim=0)
        col_hi = (d_hi * d_hi).sum(dim=0)
        scale = rows / W.shape[0]         # undo the row subsample

        _rows.append({
            "module": _n,
            "numel": rows * cols,
            "plain_lo": col_lo.sum().item() * scale,
            "plain_hi": col_hi.sum().item() * scale,
            "hess_lo": (col_lo * h).sum().item() * scale,
            "hess_hi": (col_hi * h).sum().item() * scale,
            "extra_bytes": rows * cols - rows * ((cols + 1) // 2),
        })
        del W, h

    R = pd.DataFrame(_rows)
    R["score_plain"] = (R.plain_lo - R.plain_hi) / R.extra_bytes
    R["score_hess"] = (R.hess_lo - R.hess_hi) / R.extra_bytes
    R["rank_plain"] = R.score_plain.rank(ascending=False)
    R["rank_hess"] = R.score_hess.rank(ascending=False)
    R["rank_shift"] = (R.rank_plain - R.rank_hess)

    _rho = R.rank_plain.corr(R.rank_hess, method="spearman")
    print(f"tensors compared            : {len(R)}")
    print(f"Spearman rank correlation   : {_rho:.4f}")
    print()
    if _rho > 0.95:
        print("  The two rankings essentially agree. Activation weighting would")
        print("  change almost nothing -- the data-free planner is adequate and")
        print("  effort is better spent on group size or on the codec itself.")
    elif _rho > 0.8:
        print("  Broad agreement with meaningful local reordering. Hessian-aware")
        print("  planning is worth doing, but expect a modest gain, not a jump.")
    else:
        print("  The rankings genuinely disagree. The data-free planner is")
        print("  promoting the wrong tensors, and making it Hessian-aware is the")
        print("  highest-value change available.")

    # Which tensors would the two planners disagree about, at the top of the list?
    _k = min(40, len(R))
    _top_p = set(R.nsmallest(_k, "rank_plain").module)
    _top_h = set(R.nsmallest(_k, "rank_hess").module)
    print(f"\nTop-{_k} promotion sets overlap: {len(_top_p & _top_h)}/{_k}")
    print(f"  promoted by PLAIN only : {len(_top_p - _top_h)}")
    print(f"  promoted by HESS  only : {len(_top_h - _top_p)}")

    print(f"\nBiggest rank movers (positive = activation weighting promotes it "
          f"EARLIER than the current planner would):")
    print(R.reindex(R.rank_shift.abs().sort_values(ascending=False).index)
           [["module", "rank_plain", "rank_hess", "rank_shift"]]
           .head(20).to_string(index=False))
    RANK_RHO = _rho
    return R, RANK_RHO


@app.cell
def _(
    ACT,
    BUDGET,
    BYTES_MATCH,
    FAMILIES,
    OUT_DIR,
    PER_LAYER,
    R,
    RANK_RHO,
    T,
    TRIAGE,
    W2,
    json,
    os,
):
    # ======================================================================
    # Save everything. These files are the inputs to the planner redesign.
    # ======================================================================
    os.makedirs(OUT_DIR, exist_ok=True)
    T.to_csv(f"{OUT_DIR}/tier1_tensors.csv", index=False)
    FAMILIES.to_csv(f"{OUT_DIR}/tier1_families.csv")
    PER_LAYER.to_csv(f"{OUT_DIR}/tier1_per_layer.csv")
    TRIAGE.to_csv(f"{OUT_DIR}/tier1_triage.csv")
    BUDGET.to_csv(f"{OUT_DIR}/tier1_budget.csv", index=False)
    W2.to_csv(f"{OUT_DIR}/tier2_weight_stats.csv", index=False)
    ACT.to_csv(f"{OUT_DIR}/tier3_activations.csv", index=False)
    R.to_csv(f"{OUT_DIR}/tier3_ranking_compare.csv", index=False)
    with open(f"{OUT_DIR}/summary.json", "w") as _fh:
        json.dump({
            "bytes_match_harness": bool(BYTES_MATCH),
            "n_tensors": int(len(T)),
            "rank_spearman_plain_vs_hess": float(RANK_RHO),
        }, _fh, indent=2)

    print("written:")
    for _f in sorted(os.listdir(OUT_DIR)):
        print(f"  {OUT_DIR}/{_f}")
    print("\nDownload these from the molab file browser and send them over; the")
    print("planner design follows from tier1_budget, tier2_weight_stats and")
    print("tier3_ranking_compare.")
    return


@app.cell
def _(BUDGET, BYTES_MATCH, MODEL_ID, OUT_DIR, RANK_RHO, T, TEXT_BYTES,
      TRACES_REPO, json, os):
    # ----------------------------------------------------------------------
    # Upload diagnostics_out/ to a Hugging Face DATASET repo.
    #
    # Needs a WRITE token. Set it in a cell above, or here before running:
    #     os.environ["HF_TOKEN"] = "hf_..."
    #
    # upload_folder is one call and is idempotent -- re-running it after a
    # re-run of the notebook overwrites the files in place rather than
    # duplicating them, so you can iterate without cleaning up.
    # ----------------------------------------------------------------------
    from huggingface_hub import HfApi, create_repo

    DIAG_REPO_ID = "grey-cat/cs6013-qwen35-4b-diagnostics"   # <- your username
    DIAG_PRIVATE = False

    _token = os.environ.get("HF_TOKEN")
    assert _token, "Set os.environ['HF_TOKEN'] to a WRITE token before this cell."

    # ---- a card, so the repo explains itself without this notebook ----------
    _b4 = BUDGET[(BUDGET.bits == 4) & (BUDGET.group == 64)].iloc[0]
    _b8 = BUDGET[(BUDGET.bits == 8) & (BUDGET.group == 64)].iloc[0]

    _card = f"""---
license: apache-2.0
tags: [diagnostics, quantization, gptq, cs6013]
---

# Base-model diagnostics — {MODEL_ID}

Structural, weight-level and activation-level diagnostics of the **bf16 base
model**, used to design the bit-width planner for a GPTQ compression run.

Generated by `week7_gptq_40/notebooks/model_diagnostics.py`.
Calibration traces: [`{TRACES_REPO}`](https://huggingface.co/datasets/{TRACES_REPO}).

## Headline numbers

| quantity | value |
|---|---|
| tensors in checkpoint | {len(T):,} |
| scored TEXT bytes | {TEXT_BYTES:,} ({TEXT_BYTES / 2**30:.4f} GiB) |
| matches harness constant `8,652,701,696` | **{'YES' if BYTES_MATCH else 'NO — results unusable'}** |
| uniform int4 / group 64 | `size_frac = {_b4.size_frac:.4f}` |
| uniform int8 / group 64 | `size_frac = {_b8.size_frac:.4f}` |
| Spearman ρ, plain-`‖ΔW‖²` vs Hessian-weighted ranking | **{RANK_RHO:.4f}** |

That last row is the one that matters: it measures whether weighting
quantization error by activation energy `E[x²]` reorders which tensors a
bit-width planner would promote. ρ near 1 means a data-free planner is
adequate; ρ well below 1 means it is promoting the wrong tensors.

## Files

| file | contents |
|---|---|
| `tier1_tensors.csv` | every tensor: name, shape, dtype, numel, bytes, layer, role, scored |
| `tier1_families.csv` | bytes rolled up by role — where the budget actually goes |
| `tier1_per_layer.csv` | per-layer census, with linear- vs full-attention type |
| `tier1_triage.csv` | quantizable / too-small / not-2-D / unscored buckets |
| `tier1_budget.csv` | `size_frac` for the bits × group grid |
| `tier2_weight_stats.csv` | per-tensor weight stats + round-trip relative error under the **shipping codec** (asymmetric int, integer zero-point, MSE-clipped range search) |
| `tier3_activations.csv` | per-linear activation-energy concentration |
| `tier3_ranking_compare.csv` | plain vs Hessian-weighted promotion ranking, per tensor |
| `summary.json` | machine-readable headline numbers |

## Caveats

- Tier 2 subsamples rows per tensor for speed; relative errors are estimates,
  though the codec itself is bit-identical to the one that ships.
- 3-bit rows in `tier1_budget.csv` are **idealised** bit costs. The packer only
  emits 4- and 8-bit today, so treat 3-bit as a lower bound, not a promise.
- Tier 3 uses the longest traces in the calibration set, which stress the
  Gated-DeltaNet recurrence hardest. It is not a uniform sample of the corpus.
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w") as _fh:
        _fh.write(_card)

    # ---- push ---------------------------------------------------------------
    create_repo(DIAG_REPO_ID, repo_type="dataset", private=DIAG_PRIVATE,
                exist_ok=True, token=_token)
    _api = HfApi()
    _api.upload_folder(
        folder_path=OUT_DIR,
        repo_id=DIAG_REPO_ID,
        repo_type="dataset",
        commit_message="diagnostics: tier 1 structure, tier 2 weights, tier 3 activations",
        token=_token,
    )

    DIAG_URL = f"https://huggingface.co/datasets/{DIAG_REPO_ID}"
    print("uploaded:")
    for _f in sorted(os.listdir(OUT_DIR)):
        print(f"  {_f:<32} {os.path.getsize(os.path.join(OUT_DIR, _f))/1024:8.1f} KiB")
    print(f"\n{DIAG_URL}")
    print(f"\ntotal payload: "
          f"{sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))/1024:.0f} KiB")
    return DIAG_PRIVATE, DIAG_REPO_ID, DIAG_URL


if __name__ == "__main__":
    app.run()
