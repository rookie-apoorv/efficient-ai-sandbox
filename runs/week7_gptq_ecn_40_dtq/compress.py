"""Compression entry point (week8_ec_gptq_40_dtw: decision-token-weighted Fisher).

    python compress.py --model_name <name> --checkpoint_path <base dir> --output_path <out dir>

Method: codec E grid + Fisher bit allocation + Fisher outliers + GPTQ on the
grid + numpy rANS. All constants are in ``compression/config.py``; the
calibration traces are bundled in ``compression/calib_data/aime_calib.npz``.
``CS6013_SMOKE=1`` runs a fast structural test (never submit its output).

Steps
-----
1. inventory from safetensors headers (coded / bf16 / dropped tensors)
2. load bf16 model (pure-PyTorch kernels) and the calibration file
3. Fisher pass -> outlier masks -> rate/loss tables -> per-tensor k under the
   budget, checked against the exact full-tensor entropy
4. tensors outside the decoder stack (tied embedding): RTN on the grid
5. GPTQ layer by layer over the calibration traces; after each layer the bits
   actually used are compared with the plan and the remaining layers are
   re-planned if they drift (rate control)
6. rANS-encode every code stream (each one is decoded again and compared),
   write shards, copy config/tokenizer, re-measure size_frac from the headers
Side outputs go to ``<output_path>_artifacts/``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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


def load_calib(path):
    z = np.load(path)
    ids, off, ln, pl = z["ids"], z["offsets"], z["lengths"], z["prompt_lens"]
    seqs = [torch.from_numpy(ids[o:o + n].astype(np.int64))[None] for o, n in zip(off, ln)]
    gptq = [s for s, g in zip(seqs, z["is_gptq"]) if g]
    fisher = [(s, int(p)) for s, p, f in zip(seqs, pl, z["is_fisher"]) if f]
    return gptq, fisher, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cap_tokens(samples, max_tokens):
    out, tok = [], 0
    for s in samples:
        if tok >= max_tokens:
            break
        out.append(s)
        tok += s.shape[1]
    return out, tok


# ---------------------------------------------------------------------------
def convert_from_hf_checkpoint(model_name, checkpoint_path, output_path):
    from compression import codec, rans
    from compression.fisher import collect_fisher, load_model, map_params
    from compression.gptq_ec import Hessian, gptq_on_grid
    from compression.modelscan import (chunk_by_memory, find_decoder_layers, linear_targets,
                                       make_key_resolver)
    from compression.replay import capture_layer0_inputs, run_layer
    from decompression import rans as rans_dec

    t_all = time.time()
    base_dir = resolve_checkpoint_dir(checkpoint_path, model_name)
    out_dir = Path(output_path).expanduser()
    art_dir = out_dir.parent / f"{out_dir.name}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_device = config.CACHE_DEVICE if torch.cuda.is_available() else "cpu"
    import transformers
    _log(f"[env] torch {torch.__version__}, transformers {transformers.__version__}, "
         f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | method "
         f"{config.METHOD}, target {config.TARGET_RATIO}"
         + ("   *** SMOKE MODE -- DO NOT SUBMIT ***" if config.SMOKE else ""))

    # 1. inventory -------------------------------------------------------------
    cands, raw, dropped, scored_bytes, fixed_bytes = inventory(base_dir)
    denom = config.GRADER_TEXT_GIB * 2**30
    n_w = sum(int(np.prod(s)) for s in cands.values())
    _log(f"[inventory] {len(cands)} coded tensors ({n_w / 1e9:.3f} B weights), {len(raw)} bf16 "
         f"({fixed_bytes / 2**20:.1f} MiB), {len(dropped)} dropped | scored text "
         f"{scored_bytes / 2**30:.4f} GiB")

    # 2. model + calibration ---------------------------------------------------------
    model, tok, n_gdn = load_model(base_dir, device)
    params, missing = map_params(model, cands)
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoint tensors not in the model, e.g. {missing[:3]}")
    gptq_all, fisher_all, calib_sha = load_calib(config.CALIB_FILE)
    fisher_set = fisher_all[: config.FISHER_SAMPLES]
    fisher_set = [(s[:, : config.FISHER_MAX_TOKENS], p) for s, p in fisher_set
                  if min(s.shape[1], config.FISHER_MAX_TOKENS) - p >= 16]
    gptq_set, gptq_tok = cap_tokens(gptq_all, config.GPTQ_MAX_TOKENS)
    _log(f"[calib] {config.CALIB_FILE.name} sha {calib_sha[:12]} | GPTQ {len(gptq_set)} seqs / "
         f"{gptq_tok:,} tokens (max {max(s.shape[1] for s in gptq_set):,}) | Fisher "
         f"{len(fisher_set)} seqs <= {config.FISHER_MAX_TOKENS} | {n_gdn} GDN layers on torch path")

    # 3. Fisher -> masks -> plan ---------------------------------------------------------
    names = list(cands)
    groups = {n: codec.pick_group(cands[n][1], config.GROUP_SIZE) for n in names}
    from compression.fisher import decision_mask, decision_token_sets
    dsets = decision_token_sets(tok)
    _frac = sum(int(decision_mask(s, p, dsets).sum()) for s, p in fisher_set) / max(
        sum(s.shape[1] - p for s, p in fisher_set), 1)
    _log(f"[fisher] decision-token weighting x{config.DECISION_WEIGHT}: {len(dsets[0])} decision-word "
         f"ids, {len(dsets[1])} boundary ids, {_frac * 100:.1f}% of Fisher positions weighted")

    def _pos_weight(ids, start):
        return 1.0 + (config.DECISION_WEIGHT - 1.0) * decision_mask(ids, start, dsets).float()

    fisher = collect_fisher(model, params, fisher_set, device, log=_log, pos_weight=_pos_weight)
    masks_idx = {n: codec.outlier_index(params[n], fisher[n], groups[n], config.OUTLIER_FRAC, device)
                 for n in names}
    masks = {n: codec.mask_from_index(masks_idx[n], cands[n], "cpu") for n in names}
    kgrid = codec.kgrid_from(config.K_MIN, config.K_MAX, config.K_STEPS)
    tabs = codec.build_tables(names, params, fisher, groups, masks, kgrid, device,
                              max_rows=config.TABLE_MAX_ROWS, log=_log)
    del fisher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overhead = sum(rans.lanes_for(int(np.prod(s))) * 8 + 1024 for s in cands.values())
    budget_bits = (config.TARGET_RATIO * denom - fixed_bytes - overhead) * 8
    _log(f"[plan] target {config.TARGET_RATIO} -> {budget_bits / n_w:.4f} bits/weight on average")

    def exact_rtn_bits(n, k):
        m = masks[n].to(device)
        q, _ = codec.rtn(params[n], groups[n], k, m, device)
        sym, nsym, _ = codec.to_symbols(q, m)
        return codec.stored_bits(sym, nsym, groups[n], int(m.sum()))

    budget = budget_bits
    for rnd in range(4):
        kidx, pred = codec.allocate(tabs, budget)
        planned = {n: exact_rtn_bits(n, kgrid[kidx[n]]) for n in names}
        exact = sum(planned.values())
        _log(f"  plan round {rnd + 1}: table {pred / 8 / 2**30:.4f} GiB, exact RTN "
             f"{exact / 8 / 2**30:.4f} GiB, budget {budget_bits / 8 / 2**30:.4f} GiB")
        if exact <= budget_bits:
            break
        budget -= (exact - budget_bits) * 1.05
    else:
        raise RuntimeError("could not meet the size budget after re-planning")
    kmap = {n: kgrid[kidx[n]] for n in names}
    # per-tensor correction from table estimate to exact bits, used when re-planning
    corr = {n: planned[n] / max(tabs[n]["bits"][kidx[n]].item(), 1.0) for n in names}

    # 4-5. quantize ------------------------------------------------------------------
    results = {}   # name -> dict(q, steps, outl, G, k, bits)

    def finish(n, q, w_hat, k):
        G = groups[n]
        m = masks[n].to(q.device)
        sym, nsym, qmin = codec.to_symbols(q, m)
        bits = codec.stored_bits(sym, nsym, G, int(m.sum()))
        steps = codec.group_steps(params[n].detach().float(), G, k).to(torch.float16).cpu()
        outl = w_hat.reshape(-1)[masks_idx[n].to(w_hat.device)].to(torch.float16).cpu()
        lo, hi = int(q.min()), int(q.max())
        qd = torch.int16 if -32768 <= lo and hi <= 32767 else torch.int32
        results[n] = {"q": q.to(qd).cpu(), "steps": steps, "outl": outl, "G": G, "k": k,
                      "bits": bits, "rel_mse": ((w_hat - params[n].float()).pow(2).sum()
                                                / params[n].float().pow(2).sum()).item()}
        return bits

    layers_attr, layers = find_decoder_layers(model)
    resolve = make_key_resolver(names, n_layers=len(layers))
    layer_targets = []
    for li, layer in enumerate(layers):
        tg = {}
        for mn, mod in linear_targets(layer, 64, config.MIN_NUMEL).items():
            key = resolve(layers_attr, li, mn)
            if key in kmap:
                tg[mn] = (mod, key)
        layer_targets.append(tg)
    gptq_names = {key for tg in layer_targets for _, key in tg.values()}
    if config.METHOD != "gptq":
        gptq_names = set()
    other = [n for n in names if n not in gptq_names]
    _log(f"[quant] {len(gptq_names)} tensors by GPTQ, {len(other)} by RTN ({other[:2]}...)")

    with torch.no_grad():
        for n in other:
            q, w_hat = codec.rtn(params[n], groups[n], kmap[n], masks[n].to(device), device)
            finish(n, q, w_hat, kmap[n])
            params[n].data.copy_(w_hat.to(params[n].dtype))
            del q, w_hat

    used_actual = sum(results[n]["bits"] for n in other)
    used_planned = sum(planned[n] for n in other)
    n_replans = 0
    if gptq_names:
        t0 = time.time()
        hidden, kws = capture_layer0_inputs(model, layers, gptq_set, device, cache_device)
        _log(f"[gptq] cached {len(hidden)} samples / {gptq_tok:,} tokens on {cache_device} "
             f"({time.time() - t0:.0f}s)")
        g_act = g_plan = 0.0
        for li, layer in enumerate(layers):
            tl = time.time()
            tg = layer_targets[li]
            for chunk in chunk_by_memory({mn: m for mn, (m, _) in tg.items()},
                                         int(config.HESSIAN_BUDGET_GB * 2**30)):
                hs, handles = {}, []
                for mn in chunk:
                    mod = tg[mn][0]
                    hs[mn] = Hessian(mod.in_features, device)
                    handles.append(mod.register_forward_hook(
                        lambda m, inp, out, _h=hs[mn]: _h.add(inp[0].detach())))
                run_layer(layer, hidden, kws, device, update=False)
                for h in handles:
                    h.remove()
                for mn in chunk:
                    mod, key = tg[mn]
                    G, k = groups[key], kmap[key]
                    steps = codec.group_steps(mod.weight.detach().float(), G, k)
                    q, w_hat = gptq_on_grid(mod.weight, hs[mn].H, steps, G,
                                            masks[key].to(device), config.PERCDAMP,
                                            config.BLOCK_SIZE, config.ACT_ORDER)
                    b = finish(key, q, w_hat, k)
                    mod.weight.data.copy_(w_hat.to(mod.weight.dtype))
                    g_act += b
                    g_plan += planned[key]
                    hs[mn] = None
                    del q, w_hat
                del hs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            run_layer(layer, hidden, kws, device, update=True)

            # rate control: re-plan the layers still to come
            remaining = [key for tg2 in layer_targets[li + 1:] for _, key in tg2.values()]
            used_actual = sum(r["bits"] for r in results.values())
            drift = g_act / max(g_plan, 1.0) - 1.0
            if remaining and abs(drift) > config.RATE_TOLERANCE:
                infl = 1.0 + drift
                sub = {n: {"bits": tabs[n]["bits"] * corr[n], "loss": tabs[n]["loss"]}
                       for n in remaining}
                kidx2, _ = codec.allocate(sub, budget_bits - used_actual, inflate=infl)
                for n in remaining:
                    kmap[n] = kgrid[kidx2[n]]
                    planned[n] = sub[n]["bits"][kidx2[n]].item()
                n_replans += 1
            el = time.time() - t_all
            _log(f"[gptq] layer {li + 1:>2}/{len(layers)} ({time.time() - tl:4.0f}s) | bits used "
                 f"{used_actual / 8 / 2**30:.4f} GiB | GPTQ vs plan {drift * 100:+.2f}%"
                 + (" -> re-planned" if remaining and abs(drift) > config.RATE_TOLERANCE else "")
                 + f" | elapsed {el / 60:.1f} min")
        del hidden, kws

    del model, params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. encode + write -------------------------------------------------------------
    _log("[write] rANS-encoding and writing shards ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(out_dir, config.SHARD_PREFIX, config.MAX_SHARD_BYTES)
    metas, rows_out = {}, []
    t0 = time.time()
    for name, tensor in iter_tensors(base_dir):
        dt = str(tensor.dtype).replace("torch.", "")
        if name in dropped:
            metas[name] = {"mode": "zeros", "shape": list(tensor.shape), "dtype": dt}
            continue
        if name not in results:
            writer.add(name, tensor)
            metas[name] = {"mode": "raw", "shape": list(tensor.shape), "dtype": dt}
            continue
        R = results.pop(name)
        m = masks[name]
        sym, nsym, qmin = codec.to_symbols(R["q"], m)
        sym = sym.numpy()
        enc = rans.encode(sym, nsym)
        dec = rans_dec.decode(enc["words"], enc["state"], enc["count"], enc["freq"], sym.size)
        if not np.array_equal(dec, sym):
            raise RuntimeError(f"rANS round trip failed for {name}")
        writer.add(f"{name}::words", torch.from_numpy(enc["words"].view(np.int16)))
        writer.add(f"{name}::state", torch.from_numpy(enc["state"].view(np.int32)))
        cnt_signed = np.int16 if enc["count"].dtype == np.uint16 else np.int32
        writer.add(f"{name}::count", torch.from_numpy(enc["count"].view(cnt_signed)))
        writer.add(f"{name}::freq", torch.from_numpy(enc["freq"].view(np.int32)))
        writer.add(f"{name}::steps", R["steps"])
        if R["outl"].numel():
            writer.add(f"{name}::outliers", R["outl"])
        r, c = tensor.shape
        metas[name] = {"mode": "ec", "shape": list(tensor.shape), "dtype": dt, "rows": r,
                       "cols": c, "group_size": R["G"], "k": R["k"], "q_min": qmin,
                       "escape": nsym - 1, "outliers": int(R["outl"].numel()),
                       "method": "gptq" if name in gptq_names else "rtn",
                       "stream_dtypes": {"words": "uint16", "state": "uint32", "freq": "uint32",
                                         "count": str(enc["count"].dtype)}}
        bits = enc["bytes"] * 8 + R["steps"].numel() * 16 + R["outl"].numel() * 16
        rows_out.append({"tensor": name, "k": R["k"], "group": R["G"], "bpw": bits / (r * c),
                         "rel_mse": R["rel_mse"], "method": metas[name]["method"]})
        del sym, dec, enc, R
        if len(rows_out) % 25 == 0:
            _log(f"  encoded {len(rows_out)}/{len(cands)} ({time.time() - t0:.0f}s)")
    writer.finalize()

    scored = measure_scored_bytes(out_dir)
    comp = {"format": FORMAT_VERSION, "model_name": model_name,
            "method": f"codecE-fisher-dtw{config.DECISION_WEIGHT:g}-{config.METHOD}", "target_ratio": config.TARGET_RATIO,
            "outlier_frac": config.OUTLIER_FRAC, "group_size": config.GROUP_SIZE,
            "calib_sha256": calib_sha, "smoke": config.SMOKE,
            "drop_components": list(config.DROP_COMPONENTS), "compressed_text_bytes": scored,
            "original_text_bytes": scored_bytes, "tensors": metas}
    (out_dir / "compression_config.json").write_text(json.dumps(comp, indent=2))
    copied = copy_auxiliary_files(base_dir, out_dir)
    _log(f"[write] copied: {', '.join(copied)}")

    frac_g, frac_m = scored / denom, scored / scored_bytes
    (art_dir / "plan.json").write_text(json.dumps({"tensors": rows_out, "kgrid": kgrid}, indent=2))
    (art_dir / "size_report.json").write_text(json.dumps({
        "compressed_text_bytes": scored, "size_frac_grader": frac_g, "size_frac_measured": frac_m,
        "fixed_bytes": fixed_bytes, "replans": n_replans, "gptq_tokens": gptq_tok,
        "fisher_samples": len(fisher_set), "minutes": (time.time() - t_all) / 60,
        "smoke": config.SMOKE}, indent=2))
    ok = max(frac_g, frac_m) <= config.HARD_LIMIT
    bpw = [r["bpw"] for r in rows_out]
    _log(f"\n[compress] done in {(time.time() - t_all) / 60:.1f} min | bits/weight min "
         f"{min(bpw):.2f} median {sorted(bpw)[len(bpw) // 2]:.2f} max {max(bpw):.2f} | "
         f"re-plans {n_replans}")
    _log(f"[compress] SIZE_FRAC = {frac_g:.4f} (graders' 8.0585 GiB) / {frac_m:.4f} (measured) "
         + (f"-- within {config.HARD_LIMIT}" if ok else f"-- *** OVER {config.HARD_LIMIT} ***"))
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
