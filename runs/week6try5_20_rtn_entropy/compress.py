"""Compression entry point (week7_ec_rtn_20: codec E + Fisher allocation + rANS).

    python compress.py --model_name <name> --checkpoint_path <base dir> --output_path <out dir>

Every tuning constant lives in ``compression/config.py``. ``CS6013_SMOKE=1``
shrinks only the Fisher pass (4 short samples) -- the checkpoint is still
complete, so it tests format and decompression end to end. Never submit it.

Steps
-----
1. inventory from safetensors headers: candidates (2-D, >= MIN_NUMEL), bf16
   tensors, dropped tensors (vision, MTP)
2. load the bf16 model (pure-PyTorch kernels), Fisher pass on trace samples
3. rate/Fisher-loss tables per tensor, Lagrangian choice of k per tensor
4. exact size check of the plan; re-plan if it overshoots TARGET_RATIO
5. encode: codec E + escape-coded outliers + rANS; every stream is decoded
   again and compared before it is written
6. write shards, copy config/tokenizer, re-measure size_frac from the headers
Side outputs go to ``<output_path>_artifacts/`` (outside the checkpoint).
"""

from __future__ import annotations

import argparse
import gc
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from compression import FORMAT_VERSION, config
from compression.io_utils import (ShardedSafetensorsWriter, copy_auxiliary_files,
                                  iter_tensors, list_weight_files, resolve_checkpoint_dir)

_VISION = {"visual", "vision", "vision_tower", "vision_model"}


def _log(msg):
    print(msg, flush=True)


def is_scored(name):
    n = name.replace("\\", "/").lower()
    return not ("visual." in n or any(p in _VISION for p in n.split(".")))


def is_dropped(name):
    return bool(set(name.split(".")) & set(config.DROP_COMPONENTS))


# ---------------------------------------------------------------------------
def inventory(base_dir):
    cands, raw, dropped = {}, {}, {}
    scored_bytes = fixed_bytes = 0
    for f in list_weight_files(base_dir):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                shape = tuple(sl.get_shape())
                dt = sl.get_dtype()
                nb = int(np.prod(shape)) * {"BF16": 2, "F16": 2, "F32": 4}.get(dt, 1)
                if is_scored(k):
                    scored_bytes += nb
                if is_dropped(k):
                    dropped[k] = shape
                elif (len(shape) == 2 and dt in ("BF16", "F16", "F32")
                      and int(np.prod(shape)) >= config.MIN_NUMEL and is_scored(k)):
                    cands[k] = shape
                else:
                    raw[k] = shape
                    if is_scored(k):
                        fixed_bytes += nb
    return cands, raw, dropped, scored_bytes, fixed_bytes


def measure_scored_bytes(directory):
    total = 0
    for f in sorted(Path(directory).glob("*.safetensors")):
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in hdr.items():
            if k != "__metadata__" and is_scored(k):
                total += v["data_offsets"][1] - v["data_offsets"][0]
    return total


# ---------------------------------------------------------------------------
def plan(params, fisher, groups, device, budget_bits, log=_log):
    from compression import codec

    kgrid = torch.logspace(np.log10(config.K_MIN), np.log10(config.K_MAX), config.K_STEPS).tolist()
    names = list(params)
    tabs = codec.build_tables(names, params, fisher, groups, kgrid, device,
                              config.OUTLIER_FRAC, max_rows=config.TABLE_MAX_ROWS, log=log)
    budget = budget_bits
    for rnd in range(config.REPLAN_ROUNDS):
        kidx, pred = codec.allocate(tabs, budget)
        kmap = {n: kgrid[kidx[n]] for n in names}
        # exact entropy of the full tensors (no rANS yet)
        exact = sum(codec.exact_bits(params[n], fisher[n], groups[n], kmap[n],
                                     config.OUTLIER_FRAC, device) for n in names)
        log(f"  plan round {rnd + 1}: predicted {pred / 8 / 2**30:.4f} GiB, "
            f"exact entropy {exact / 8 / 2**30:.4f} GiB, budget {budget_bits / 8 / 2**30:.4f} GiB")
        if exact <= budget_bits:
            return kmap, tabs, kgrid, exact
        budget -= (exact - budget_bits) * 1.05
    raise RuntimeError("could not meet the size budget after re-planning")


def convert_from_hf_checkpoint(model_name, checkpoint_path, output_path):
    from compression import codec, rans
    from compression.calibration import fisher_samples, holdout_ids, load_traces
    from compression.fisher import collect_fisher, load_model, map_params
    from decompression import rans as rans_dec

    t_all = time.time()
    base_dir = resolve_checkpoint_dir(checkpoint_path, model_name)
    out_dir = Path(output_path).expanduser()
    art_dir = out_dir.parent / f"{out_dir.name}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import transformers
    _log(f"[env] torch {torch.__version__}, transformers {transformers.__version__}, device "
         f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}"
         + ("   *** SMOKE MODE -- DO NOT SUBMIT ***" if config.SMOKE else ""))

    # 1. inventory ------------------------------------------------------------
    cands, raw, dropped, scored_bytes, fixed_bytes = inventory(base_dir)
    denom = config.GRADER_TEXT_GIB * 2**30
    n_w = sum(int(np.prod(s)) for s in cands.values())
    _log(f"[inventory] {len(cands)} coded tensors ({n_w / 1e9:.3f} B weights), {len(raw)} bf16, "
         f"{len(dropped)} dropped | scored text {scored_bytes / 2**30:.4f} GiB "
         f"(graders' constant {config.GRADER_TEXT_GIB}) | bf16 kept {fixed_bytes / 2**20:.1f} MiB")

    # 2. model + Fisher ---------------------------------------------------------
    model, tok, n_gdn = load_model(base_dir, device)
    params, missing = map_params(model, cands)
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoint tensors not found in the model, e.g. {missing[:3]}")
    _log(f"[model] {type(model).__name__} on {device}, {n_gdn} GDN layers on the torch path")
    traces = load_traces(config.CALIB_FILE, config.CALIB_SHA256)
    held, _ = holdout_ids(tok, traces, config.HOLDOUT_MAX_LEN, config.HOLDOUT_TOKENS)
    samples = fisher_samples(tok, traces, config.FISHER_SAMPLES, config.FISHER_MAX_TOKENS,
                             held, config.SEED)
    _log(f"[fisher] {len(samples)} samples (<= {config.FISHER_MAX_TOKENS} tokens), "
         f"{len(held)} held-out traces excluded")
    fisher = collect_fisher(model, params, samples, device, log=_log)

    # 3-4. plan -------------------------------------------------------------------
    groups = {n: codec.pick_group(s[1], config.GROUP_SIZE) for n, s in cands.items()}
    overhead = sum(rans.lanes_for(int(np.prod(s))) * 8 + 1024 for s in cands.values())
    budget_bits = (config.TARGET_RATIO * denom - fixed_bytes - overhead) * 8
    _log(f"[plan] target size_frac {config.TARGET_RATIO} -> {budget_bits / n_w:.4f} bits/weight")
    kmap, tabs, kgrid, exact_bits = plan(params, fisher, groups, device, budget_bits)

    # 5-6. encode + write ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(out_dir, config.SHARD_PREFIX, config.MAX_SHARD_BYTES)
    metas, rows_out = {}, []
    t0 = time.time()
    tot_f = 0.0
    for name, tensor in iter_tensors(base_dir):
        if name in dropped:
            metas[name] = {"mode": "zeros", "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype).replace("torch.", "")}
            continue
        if name not in cands:
            writer.add(name, tensor)
            metas[name] = {"mode": "raw", "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype).replace("torch.", "")}
            continue
        G, k = groups[name], kmap[name]
        sym, nsym, qmin, steps, outl, st = codec.quantize_tensor(
            params[name], fisher[name], G, k, config.OUTLIER_FRAC, device)
        enc = rans.encode(sym, nsym)
        dec = rans_dec.decode(enc["words"], enc["state"], enc["count"], enc["freq"], sym.size)
        if not np.array_equal(dec, sym):
            raise RuntimeError(f"rANS round trip failed for {name}")
        writer.add(f"{name}::words", torch.from_numpy(enc["words"].view(np.int16)))
        writer.add(f"{name}::state", torch.from_numpy(enc["state"].view(np.int32)))
        writer.add(f"{name}::count", torch.from_numpy(enc["count"].view(
            np.int16 if enc["count"].dtype == np.uint16 else np.int32)))
        writer.add(f"{name}::freq", torch.from_numpy(enc["freq"].view(np.int32)))
        writer.add(f"{name}::steps", steps)
        if outl.numel():
            writer.add(f"{name}::outliers", outl)
        r, c = tensor.shape
        bits = enc["bytes"] * 8 + steps.numel() * 16 + outl.numel() * 16
        metas[name] = {"mode": "ec", "shape": list(tensor.shape),
                       "dtype": str(tensor.dtype).replace("torch.", ""),
                       "rows": r, "cols": c, "group_size": G, "k": k, "q_min": qmin,
                       "escape": nsym - 1, "outliers": int(outl.numel()),
                       "stream_dtypes": {"words": "uint16", "state": "uint32", "freq": "uint32",
                                         "count": str(enc["count"].dtype)}}
        tot_f += st["fisher_loss"]
        rows_out.append({"tensor": name, "k": k, "group": G, "bpw": bits / (r * c),
                         "rel_mse": st["sse"] / st["wss"], "fisher_loss": st["fisher_loss"],
                         "outliers": st["outliers"]})
        del sym, dec, enc
        if len(rows_out) % 20 == 0:
            _log(f"  encoded {len(rows_out)}/{len(cands)} ({time.time() - t0:.0f}s)")
    writer.finalize()
    del model, fisher, params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scored = measure_scored_bytes(out_dir)
    comp = {
        "format": FORMAT_VERSION, "model_name": model_name, "method": "codecE-fisher-rtn",
        "target_ratio": config.TARGET_RATIO, "outlier_frac": config.OUTLIER_FRAC,
        "group_size": config.GROUP_SIZE, "fisher_samples": len(samples), "smoke": config.SMOKE,
        "drop_components": list(config.DROP_COMPONENTS), "compressed_text_bytes": scored,
        "original_text_bytes": scored_bytes, "tensors": metas,
    }
    (out_dir / "compression_config.json").write_text(json.dumps(comp, indent=2))
    copied = copy_auxiliary_files(base_dir, out_dir)
    _log(f"[compress] copied: {', '.join(copied)}")

    frac_g = scored / denom
    frac_m = scored / scored_bytes
    (art_dir / "plan.json").write_text(json.dumps({"tensors": rows_out, "kgrid": kgrid}, indent=2))
    (art_dir / "size_report.json").write_text(json.dumps({
        "compressed_text_bytes": scored, "size_frac_grader": frac_g, "size_frac_measured": frac_m,
        "planned_entropy_bytes": exact_bits / 8, "fixed_bytes": fixed_bytes,
        "fisher_loss_total": tot_f, "minutes": (time.time() - t_all) / 60, "smoke": config.SMOKE,
    }, indent=2))
    ok = max(frac_g, frac_m) <= config.HARD_LIMIT
    _log(f"\n[compress] done in {(time.time() - t_all) / 60:.1f} min. SIZE_FRAC = {frac_g:.4f} "
         f"(graders' 8.0585 GiB) / {frac_m:.4f} (measured denominator) "
         + ("-- within 0.20" if ok else "-- *** OVER 0.20, DO NOT SUBMIT ***"))
    _log(f"[compress] Fisher-predicted loss increase (sum): {tot_f:.4g}")
    if not ok:
        raise SystemExit(2)
    return comp


def main():
    p = argparse.ArgumentParser(description="Compress a Hugging Face checkpoint.")
    p.add_argument("--model_name", "--model-name", dest="model_name", required=True)
    p.add_argument("--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", required=True)
    p.add_argument("--output_path", "--output-path", dest="output_path", required=True)
    a = p.parse_args()
    convert_from_hf_checkpoint(a.model_name, a.checkpoint_path, a.output_path)


if __name__ == "__main__":
    main()
