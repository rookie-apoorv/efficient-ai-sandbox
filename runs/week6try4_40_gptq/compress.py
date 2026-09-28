"""Compression entry point (week7_gptq_40).

    python compress.py \
        --model_name <model_name> \
        --checkpoint_path <path to base checkpoint dir> \
        --output_path <path for the compressed checkpoint dir>

These three arguments are the complete command line; every tuning constant is
in ``compression/config.py``. ``CS6013_SMOKE=1`` in the environment runs a fast
structural test (see config.py) -- never submit its output.

Steps
-----
0. environment report (versions, GPU, which optional kernels exist -- none used)
1. inventory: stream every tensor off disk; size, drop list, int4/int8 byte
   cost and weight error of each candidate (shipping MSE-clipped codec)
2. load the bf16 model; pin the pure-PyTorch kernels; build the calibration set
   (policy E); probe one GDN and one full-attention layer at the longest length
3. energy pre-pass: E[x^2] of every linear's input on a subset of the corpus
4. plan: energy-weighted int4/int8 knapsack under TARGET_RATIO
5. GPTQ, layer by layer, over the full corpus
6. write: GPTQ payloads + RTN for the embedding + raw small tensors + zeros meta
   for dropped components; then re-measure size_frac from the written headers

Side outputs (energy, plan, calibration stats, probe, size report) go to
``<output_path>_artifacts/`` -- NOT inside the checkpoint directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import torch

from compression import FORMAT_VERSION, config
from compression.io_utils import (
    ShardedSafetensorsWriter,
    copy_auxiliary_files,
    iter_tensors,
    resolve_checkpoint_dir,
)
from compression.planner import format_plan_report, plan_bit_widths, role_of
from compression.quantize import (
    QuantResult,
    counts_toward_size,
    estimate_error,
    is_quantizable,
    quantize_tensor_rtn,
    quantized_nbytes,
    should_drop,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _is_gptq_target(name: str) -> bool:
    """Decoder-stack linears are reached by GPTQ (and carry an act-order perm)."""
    return ".layers." in name and not name.endswith("embed_tokens.weight")


# --------------------------------------------------------------------------
# 1. inventory
# --------------------------------------------------------------------------
def inventory(base_dir: Path, device: str):
    candidates, fixed_bytes, size_bytes, total_bytes = [], 0, 0, 0
    dropped, uncounted, perm_bytes = [], {}, 0
    t0 = time.time()

    for seen, (name, tensor) in enumerate(iter_tensors(base_dir), 1):
        nbytes = tensor.numel() * tensor.element_size()
        total_bytes += nbytes
        scored = counts_toward_size(name)
        if scored:
            size_bytes += nbytes

        if should_drop(name, config.DROP_COMPONENTS):
            dropped.append((name, nbytes, scored))
            continue

        if not is_quantizable(tensor, config.GROUP_SIZE, config.MIN_NUMEL):
            if scored:
                fixed_bytes += nbytes
            continue

        rows = tensor.numel() // tensor.shape[-1]
        cols = tensor.shape[-1]
        if not scored:
            uncounted[name] = config.BASE_BITS
            continue

        if config.METHOD == "gptq" and config.ACT_ORDER and _is_gptq_target(name):
            perm_bytes += cols * 4  # int32 act-order permutation, stored per tensor

        candidates.append(
            {
                "name": name,
                "numel": tensor.numel(),
                "rows": rows,
                "cols": cols,
                "bytes": {
                    b: quantized_nbytes(rows, cols, b, config.GROUP_SIZE)
                    for b in (config.BASE_BITS, config.HIGH_BITS)
                },
                "err": {
                    b: estimate_error(
                        tensor, b, config.GROUP_SIZE,
                        max_rows=config.ESTIMATE_MAX_ROWS,
                        mse=config.ESTIMATE_MSE, device=device,
                    )
                    for b in (config.BASE_BITS, config.HIGH_BITS)
                },
            }
        )
        if seen % 200 == 0:
            _log(f"  scanned {seen} tensors ({time.time() - t0:.0f}s)")

    _log(f"[inventory] {len(candidates)} planner candidates in {time.time() - t0:.0f}s")
    if dropped:
        by_root = {}
        for nm, nb, sc in dropped:
            root = nm.split(".")[0] + ("" if sc else "  (unscored)")
            if "mtp" in nm.split("."):
                root = "mtp  (scored -> freed budget)"
            by_root[root] = by_root.get(root, 0) + nb
        _log(f"[inventory] dropping {len(dropped)} tensors -> restored as zeros:")
        for root, nb in sorted(by_root.items(), key=lambda kv: -kv[1]):
            _log(f"    {root:<40} {nb / 2**20:9.1f} MiB")
    return {
        "candidates": candidates,
        "fixed_bytes": fixed_bytes + perm_bytes,
        "perm_bytes": perm_bytes,
        "size_bytes": size_bytes,
        "total_bytes": total_bytes,
        "uncounted": uncounted,
        "n_dropped": len(dropped),
    }


# --------------------------------------------------------------------------
# 2-5. model, calibration, probe, energy, plan, GPTQ
# --------------------------------------------------------------------------
def _fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_model(base_dir: Path, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compression.preflight import force_torch_kernels

    _log(f"[model] loading bf16 model (attn={config.ATTN_IMPLEMENTATION}) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=config.ATTN_IMPLEMENTATION,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    if config.FORCE_TORCH_KERNELS:
        _log(f"[model] pinned pure-PyTorch kernels: {force_torch_kernels(model)}")
    model.to(device)
    _log(
        f"[model] {type(model).__name__} on {device} in {time.time() - t0:.0f}s, "
        f"attn_implementation={model.config._attn_implementation}"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    return model, tokenizer


def run_calibrated_stages(base_dir: Path, inv: dict, art_dir: Path):
    from compression.calibration import build_calibration, select_for_energy
    from compression.energy import collect_energy, energy_for_candidates
    from compression.modelscan import coverage_report, format_coverage
    from compression.pipeline import gptq_quantize_model
    from compression.preflight import probe_long_sequence, sdpa_backends

    device = torch.device(config.resolve_device())
    cache_device = config.resolve_cache_device()
    model, tokenizer = load_model(base_dir, device)

    ckpt_keys = [c["name"] for c in inv["candidates"]]
    _log(format_coverage(coverage_report(model, config.GROUP_SIZE, config.MIN_NUMEL,
                                         ckpt_keys=ckpt_keys)))

    # ---- calibration ------------------------------------------------------
    samples, stats = build_calibration(
        tokenizer,
        config.CALIB_FILE,
        expected_sha256=config.CALIB_SHA256,
        runaway_max_tokens=config.RUNAWAY_MAX_TOKENS,
        max_seq=config.CALIB_MAX_SEQ,
        limit_samples=config.CALIB_LIMIT_SAMPLES,
        seed=config.CALIB_SEED,
    )
    _log(stats.format())
    (art_dir / "calib_stats.json").write_text(json.dumps(stats.__dict__, indent=2))

    # ---- preflight probe --------------------------------------------------
    if config.PREFLIGHT_PROBE and device.type == "cuda":
        _log(f"[preflight] torch SDPA kernels at head_dim 256: {sdpa_backends()}")
        _log(f"[preflight] probing one GDN + one full-attention layer at "
             f"{stats.max_len:,} tokens ...")
        try:
            probe = probe_long_sequence(model, stats.max_len, device)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Out of memory running a single layer at {stats.max_len} tokens. "
                "Set CALIB_MAX_SEQ in compression/config.py (e.g. 16384) and re-run."
            ) from exc
        _log(f"[preflight] {probe}")
        # Rough ETA: every layer is replayed twice over the corpus.
        types = getattr(model.config, "layer_types", []) or []
        n_full = sum(t == "full_attention" for t in types)
        n_gdn = len(types) - n_full
        per_tok = (n_gdn * probe.get("gdn_sec", 0) + n_full * probe.get("full_attn_sec", 0)) / probe["seq_len"]
        _log(f"[preflight] rough GPTQ forward-replay ETA: "
             f"{2 * per_tok * stats.n_tokens / 60:.0f} min (+ GPTQ solves)")
        (art_dir / "probe.json").write_text(json.dumps(probe, indent=2))

    # ---- energy -----------------------------------------------------------
    weights = None
    if config.PLANNER == "energy":
        e_samples, e_tok = select_for_energy(samples, config.ENERGY_MAX_TOKENS, config.CALIB_SEED)
        cfg_sha = hashlib.sha256((base_dir / "config.json").read_bytes()).hexdigest() \
            if (base_dir / "config.json").exists() else "none"
        fp = _fingerprint(config.CALIB_SHA256, cfg_sha, e_tok, len(e_samples),
                          config.CALIB_MAX_SEQ, config.RUNAWAY_MAX_TOKENS, config.SMOKE)
        cache = art_dir / "energy.json"
        energy = None
        if cache.exists():
            blob = json.loads(cache.read_text())
            if blob.get("fingerprint") == fp:
                energy = blob["energy"]
                _log(f"[energy] reusing cached energies from {cache}")
        if energy is None:
            _log(f"[energy] pre-pass over {len(e_samples)} samples / {e_tok:,} tokens ...")
            energy = collect_energy(model, e_samples, ckpt_keys, device)
            cache.write_text(json.dumps({"fingerprint": fp, "energy": energy}, indent=2))
        weights = energy_for_candidates(inv["candidates"], energy)

    # ---- plan -------------------------------------------------------------
    bits_by_name, projected = make_plan(inv, weights, art_dir)

    # ---- GPTQ -------------------------------------------------------------
    _log(f"[gptq] quantizing over {stats.n_samples} samples / {stats.n_tokens:,} tokens ...")
    payloads = gptq_quantize_model(
        model,
        samples,
        bits_by_name=bits_by_name,
        group_size=config.GROUP_SIZE,
        min_numel=config.MIN_NUMEL,
        device=device,
        cache_device=cache_device,
        hessian_budget_bytes=int(config.HESSIAN_BUDGET_GB * 2**30),
        percdamp=config.PERCDAMP,
        act_order=config.ACT_ORDER,
        blocksize=config.BLOCK_SIZE,
        mse=config.MSE_CLIPPING,
    )
    if not payloads:
        raise RuntimeError("GPTQ produced zero quantized tensors; refusing to write an RTN model.")
    _log(f"[gptq] produced {len(payloads)} quantized tensors")

    del model, tokenizer, samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bits_by_name, payloads


def make_plan(inv: dict, weights, art_dir: Path):
    cands = inv["candidates"]
    bits_by_name, projected = plan_bit_widths(
        cands, inv["fixed_bytes"], inv["size_bytes"], config.TARGET_RATIO,
        config.BASE_BITS, config.HIGH_BITS, weights=weights,
    )
    planner = config.PLANNER if weights is not None else "plain"
    _log(format_plan_report(cands, bits_by_name, inv["fixed_bytes"], inv["size_bytes"],
                            projected, planner=planner, high_bits=config.HIGH_BITS))
    if projected > config.TARGET_RATIO * inv["size_bytes"]:
        _log("[plan] WARNING: uniform int4 already exceeds the target ratio.")
    plan_rows = [
        {
            "name": c["name"], "role": role_of(c["name"]), "bits": bits_by_name[c["name"]],
            "score": c["score"], "energy": (weights or {}).get(c["name"]),
            "err4": c["err"][config.BASE_BITS], "err8": c["err"][config.HIGH_BITS],
            "bytes4": c["bytes"][config.BASE_BITS], "bytes8": c["bytes"][config.HIGH_BITS],
        }
        for c in sorted(cands, key=lambda c: -c["score"])
    ]
    (art_dir / "plan.json").write_text(json.dumps(
        {"planner": planner, "projected_bytes": projected,
         "size_bytes": inv["size_bytes"], "tensors": plan_rows}, indent=2))
    bits_by_name.update(inv["uncounted"])
    return bits_by_name, projected


# --------------------------------------------------------------------------
# 6. write + measure
# --------------------------------------------------------------------------
def write_compressed(base_dir, out_dir, model_name, bits_by_name, payloads, size_bytes):
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(out_dir, config.SHARD_PREFIX, config.MAX_SHARD_BYTES)
    meta_all, n_gptq, n_rtn, n_raw, n_zero = {}, 0, 0, 0, 0
    rtn_device = config.resolve_device()
    t0 = time.time()

    for name, tensor in iter_tensors(base_dir):
        if should_drop(name, config.DROP_COMPONENTS):
            meta_all[name] = {"mode": "zeros", "shape": list(tensor.shape),
                              "dtype": str(tensor.dtype).replace("torch.", "")}
            n_zero += 1
            continue

        result: QuantResult | None = payloads.get(name)
        if result is not None:
            n_gptq += 1
        elif name in bits_by_name:
            result = quantize_tensor_rtn(tensor, bits_by_name[name], config.GROUP_SIZE,
                                         mse=config.MSE_CLIPPING, device=rtn_device)
            result.meta["method"] = "rtn"
            n_rtn += 1
            _log(f"  RTN {name} @ int{bits_by_name[name]}")
        else:
            writer.add(name, tensor)
            meta_all[name] = {"mode": "raw", "shape": list(tensor.shape),
                              "dtype": str(tensor.dtype).replace("torch.", "")}
            n_raw += 1
            continue

        writer.add(f"{name}::codes", result.codes)
        writer.add(f"{name}::scales", result.scales)
        writer.add(f"{name}::zps", result.zps)
        if result.perm is not None:
            writer.add(f"{name}::perm", result.perm)
        meta_all[name] = result.meta

    writer.finalize()
    _log(f"[write] {n_gptq} GPTQ / {n_rtn} RTN / {n_raw} raw / {n_zero} zeros "
         f"in {time.time() - t0:.0f}s")

    measured = measure_scored_bytes(out_dir)
    compressed = {
        "format": FORMAT_VERSION,
        "model_name": model_name,
        "method": config.METHOD,
        "planner": config.PLANNER,
        "group_size": config.GROUP_SIZE,
        "min_numel": config.MIN_NUMEL,
        "target_ratio": config.TARGET_RATIO,
        "act_order": config.ACT_ORDER,
        "mse_clipping": config.MSE_CLIPPING,
        "percdamp": config.PERCDAMP,
        "calibration": "policy-E self-generated traces, sha256 " + config.CALIB_SHA256[:12],
        "runaway_max_tokens": config.RUNAWAY_MAX_TOKENS,
        "smoke": config.SMOKE,
        "n_gptq_tensors": n_gptq,
        "n_rtn_tensors": n_rtn,
        "n_raw_tensors": n_raw,
        "n_dropped_tensors": n_zero,
        "drop_components": list(config.DROP_COMPONENTS),
        "original_text_bytes": size_bytes,
        "compressed_total_bytes": writer.total_bytes,
        "compressed_text_bytes": measured,
        "size_frac": measured / max(size_bytes, 1),
        "shard_prefix": config.SHARD_PREFIX,
        "tensors": meta_all,
    }
    (out_dir / "compression_config.json").write_text(json.dumps(compressed, indent=2))
    return compressed


def measure_scored_bytes(directory: Path) -> int:
    """Sum tensor bytes from safetensors HEADERS, counting only what graders score."""
    total = 0
    for f in sorted(Path(directory).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if counts_toward_size(key):
                a, b = info["data_offsets"]
                total += b - a
    return total


# --------------------------------------------------------------------------
def convert_from_hf_checkpoint(model_name: str, checkpoint_path: str, output_path: str) -> dict:
    from compression.preflight import environment_report

    base_dir = resolve_checkpoint_dir(checkpoint_path, model_name)
    out_dir = Path(output_path).expanduser()
    art_dir = out_dir.parent / f"{out_dir.name}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    _log(environment_report())
    _log(f"[compress] base checkpoint : {base_dir}")
    _log(f"[compress] output          : {out_dir}")
    _log(f"[compress] artifacts       : {art_dir}")
    _log(f"[compress] method {config.METHOD}, planner {config.PLANNER}, group "
         f"{config.GROUP_SIZE}, target {config.TARGET_RATIO}"
         + ("   *** SMOKE MODE -- DO NOT SUBMIT ***" if config.SMOKE else ""))

    _log("[compress] step 1: inventory + weight-error estimates ...")
    inv = inventory(base_dir, config.resolve_device())

    if config.METHOD == "gptq":
        bits_by_name, payloads = run_calibrated_stages(base_dir, inv, art_dir)
    else:
        bits_by_name, _ = make_plan(inv, None, art_dir)
        payloads = {}

    _log("[compress] step 6: encoding and writing shards ...")
    comp = write_compressed(base_dir, out_dir, model_name, bits_by_name, payloads, inv["size_bytes"])
    copied = copy_auxiliary_files(base_dir, out_dir)
    _log(f"[compress] copied config/tokenizer files: {', '.join(copied)}")

    grader_frac = comp["compressed_text_bytes"] / 2**30 / config.GRADER_TEXT_GIB
    report = {
        "compressed_text_bytes": comp["compressed_text_bytes"],
        "original_text_bytes": inv["size_bytes"],
        "size_frac_measured_denominator": comp["size_frac"],
        "size_frac_grader_constant": grader_frac,
        "compressed_total_bytes": comp["compressed_total_bytes"],
        "minutes": (time.time() - t_all) / 60,
        "smoke": config.SMOKE,
    }
    (art_dir / "size_report.json").write_text(json.dumps(report, indent=2))
    ok = max(comp["size_frac"], grader_frac) <= 0.40
    _log(
        f"\n[compress] done in {report['minutes']:.1f} min. "
        f"{comp['compressed_total_bytes'] / 2**30:.3f} GiB on disk.\n"
        f"[compress] SIZE_FRAC = {comp['size_frac']:.4f} (measured denominator) / "
        f"{grader_frac:.4f} (grader's 8.0585 GiB) "
        + ("-- within 0.40" if ok else "-- *** OVER 0.40, THIS WILL FAIL ***")
    )
    return comp


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a Hugging Face checkpoint.")
    parser.add_argument("--model_name", "--model-name", dest="model_name", required=True)
    parser.add_argument("--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", required=True)
    parser.add_argument("--output_path", "--output-path", dest="output_path", required=True)
    args = parser.parse_args()
    convert_from_hf_checkpoint(args.model_name, args.checkpoint_path, args.output_path)


if __name__ == "__main__":
    main()
