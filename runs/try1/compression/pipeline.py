"""Compression pipeline: base checkpoint -> quantized checkpoint directory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .checkpoint_io import (
    ShardWriter,
    copy_aux_files,
    dir_size_bytes,
    iter_tensors,
    resolve_checkpoint,
    tensor_manifest,
    weights_nbytes,
)
from .policy import QuantPolicy, tensor_family
from .quantize import quantization_error, quantize_tensor

QUANT_CONFIG = "quant_config.json"
FORMAT_VERSION = 1

# Suffixes appended to a quantized tensor's original name.
SUF_PACKED, SUF_SCALE, SUF_ZERO = ".qpacked", ".qscale", ".qzero"


def compress_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
    policy: QuantPolicy,
    *,
    max_shard_bytes: int | None = None,
    measure_error: bool = True,
    report_path: str | None = None,
) -> dict:
    """Quantize a checkpoint according to ``policy`` and write it to ``output_path``.

    Returns a report dict (also written to ``compression_report.json``).
    """
    started = time.time()
    src = resolve_checkpoint(checkpoint_path or model_name)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    manifest = tensor_manifest(src)
    base_bytes = weights_nbytes(src)
    print(
        f"[compress] source: {src}\n"
        f"[compress] {len(manifest)} tensors, "
        f"{sum(t['numel'] for t in manifest.values()):,} params, "
        f"{base_bytes:,} bytes ({base_bytes / 1024**3:.3f} GiB)"
    )
    print(f"[compress] profile: {policy.name} (bits={policy.bits}, group={policy.group_size})")

    tensors_meta: dict[str, dict] = {}
    dense_meta: dict[str, str] = {}
    errors: list[tuple[float, str]] = []
    family_stats: dict[str, dict[str, float]] = {}

    writer_kwargs = {}
    if max_shard_bytes:
        writer_kwargs["max_shard_bytes"] = max_shard_bytes

    with ShardWriter(out, **writer_kwargs) as writer:
        for name, tensor in tqdm(
            iter_tensors(src), total=len(manifest), desc="quantizing", unit="t"
        ):
            family = tensor_family(name)
            stats = family_stats.setdefault(
                family, {"params": 0, "orig_bytes": 0, "new_bytes": 0, "quantized": 0}
            )
            numel = tensor.numel()
            orig_bytes = numel * tensor.element_size()
            stats["params"] += numel
            stats["orig_bytes"] += orig_bytes

            plan = policy.plan(name, tuple(tensor.shape))
            if plan is None:
                writer.add(name, tensor)
                dense_meta[name] = str(tensor.dtype).replace("torch.", "")
                stats["new_bytes"] += orig_bytes
                continue

            qt = quantize_tensor(tensor, bits=plan["bits"], group_size=plan["group_size"])
            writer.add(name + SUF_PACKED, qt.packed)
            writer.add(name + SUF_SCALE, qt.scale)
            writer.add(name + SUF_ZERO, qt.zero)

            tensors_meta[name] = {
                "bits": qt.bits,
                "group_size": qt.group_size,
                "shape": list(qt.shape),
                "orig_dtype": qt.orig_dtype,
            }
            stats["new_bytes"] += qt.nbytes()
            stats["quantized"] += numel

            if measure_error:
                err = quantization_error(tensor, qt)
                errors.append((err["rel_fro"], name))

            del tensor, qt

    quant_config = {
        "format_version": FORMAT_VERSION,
        "base_model": model_name,
        "profile": policy.name,
        "default_bits": policy.bits,
        "default_group_size": policy.group_size,
        "suffixes": {"packed": SUF_PACKED, "scale": SUF_SCALE, "zero": SUF_ZERO},
        "tensors": tensors_meta,
        "dense_dtypes": dense_meta,
    }
    (out / QUANT_CONFIG).write_text(json.dumps(quant_config, indent=2))

    copied = copy_aux_files(src, out)

    # Size is measured BEFORE the report is written, and the report is written
    # OUTSIDE the checkpoint directory. The spec forbids shipping experiment
    # outputs in the HuggingFace checkpoint, and anything left in the directory
    # counts against the size budget.
    weights_bytes = dir_size_bytes(out, "*.safetensors")
    all_bytes = dir_size_bytes(out)
    n_params = sum(t["numel"] for t in manifest.values())
    quantized_params = sum(s["quantized"] for s in family_stats.values())

    report = {
        "base_model": model_name,
        "profile": policy.name,
        "elapsed_sec": round(time.time() - started, 1),
        "num_tensors": len(manifest),
        "num_params": n_params,
        "num_params_quantized": quantized_params,
        "frac_params_quantized": quantized_params / n_params if n_params else 0.0,
        "base_weights_bytes": base_bytes,
        "compressed_weights_bytes": weights_bytes,
        "compressed_all_files_bytes": all_bytes,
        "ratio_weights_only": weights_bytes / base_bytes if base_bytes else 0.0,
        "ratio_all_files": all_bytes / base_bytes if base_bytes else 0.0,
        "effective_bits_per_param": (weights_bytes * 8 / n_params) if n_params else 0.0,
        "aux_files_copied": copied,
        "per_family": {
            k: {
                **v,
                "ratio": (v["new_bytes"] / v["orig_bytes"]) if v["orig_bytes"] else 0.0,
            }
            for k, v in sorted(family_stats.items())
        },
        "worst_tensors_by_rel_fro": [
            {"name": n, "rel_fro": round(e, 5)}
            for e, n in sorted(errors, reverse=True)[:15]
        ],
    }

    report_file = (
        Path(report_path)
        if report_path
        else out.parent / f"{out.name}_compression_report.json"
    )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2))

    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    print("\n" + "=" * 74)
    print(f"COMPRESSION REPORT  ({report['profile']})")
    print("=" * 74)
    print(
        f"  params            {report['num_params']:,} "
        f"({report['frac_params_quantized'] * 100:.1f}% quantized)"
    )
    print(f"  base weights      {report['base_weights_bytes']:,} bytes")
    print(f"  compressed        {report['compressed_weights_bytes']:,} bytes")
    print(f"  + aux files       {report['compressed_all_files_bytes']:,} bytes")
    print(f"  effective bits    {report['effective_bits_per_param']:.3f} bits/param")
    print()
    print(f"  RATIO (weights)   {report['ratio_weights_only'] * 100:.2f}%")
    print(f"  RATIO (all files) {report['ratio_all_files'] * 100:.2f}%")
    print()

    r = report["ratio_all_files"] * 100
    hits = [t for t in (40, 20, 10) if r <= t]
    if hits:
        print(f"  -> clears target(s): {', '.join(f'{t}%' for t in hits)} (best: {min(hits)}%)")
    else:
        print(f"  -> clears NO target ({r:.2f}% > 40%). Lower --bits or use a mixed profile.")

    print("\n  per-family:")
    print(f"    {'family':<14}{'params':>16}{'orig MiB':>12}{'new MiB':>11}{'ratio':>9}")
    for family, s in report["per_family"].items():
        print(
            f"    {family:<14}{s['params']:>16,}"
            f"{s['orig_bytes'] / 1024**2:>12.1f}"
            f"{s['new_bytes'] / 1024**2:>11.1f}"
            f"{s['ratio'] * 100:>8.1f}%"
        )

    worst = report["worst_tensors_by_rel_fro"]
    if worst:
        print("\n  highest relative reconstruction error:")
        for item in worst[:8]:
            print(f"    {item['rel_fro']:.5f}  {item['name']}")
    print("=" * 74 + "\n")
