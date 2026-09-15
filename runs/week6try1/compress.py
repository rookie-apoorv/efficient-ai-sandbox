"""Compression entry point.

    python compress.py \
        --model_name <model_name> \
        --checkpoint_path <path to base checkpoint dir> \
        --output_path <path for the compressed checkpoint dir>

These three arguments are the complete command line. Every tuning constant
lives in ``compression/config.py``; none of the three arguments above is ever
hard-coded.

The run always prints two diagnostics before it writes anything: the bit-width
plan with the projected size ratio, and (for GPTQ) the coverage report showing
what fraction of the parameters GPTQ can actually reach.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from compression.planner import format_plan_report, plan_bit_widths
from compression.quantize import (
    QuantResult,
    estimate_error,
    is_quantizable,
    quantize_tensor_rtn,
    quantized_nbytes,
)


def build_plan(base_dir: Path):
    """Pass 1: size and sensitivity of every tensor, read straight off disk."""
    candidates, fixed_bytes, original_bytes, seen = [], 0, 0, 0
    t0 = time.time()

    for name, tensor in iter_tensors(base_dir):
        seen += 1
        original_bytes += tensor.numel() * tensor.element_size()

        if not is_quantizable(tensor, config.GROUP_SIZE, config.MIN_NUMEL):
            fixed_bytes += tensor.numel() * tensor.element_size()
            continue

        rows = tensor.numel() // tensor.shape[-1]
        cols = tensor.shape[-1]
        candidates.append(
            {
                "name": name,
                "numel": tensor.numel(),
                "bytes": {
                    b: quantized_nbytes(rows, cols, b, config.GROUP_SIZE)
                    for b in (config.BASE_BITS, config.HIGH_BITS)
                },
                "err": {
                    b: estimate_error(tensor, b, config.GROUP_SIZE)
                    for b in (config.BASE_BITS, config.HIGH_BITS)
                },
            }
        )
        if seen % 200 == 0:
            print(f"  scanned {seen} tensors ({time.time() - t0:.0f}s)", flush=True)
        del tensor

    bits_by_name, projected = plan_bit_widths(
        candidates,
        fixed_bytes,
        original_bytes,
        config.TARGET_RATIO,
        config.BASE_BITS,
        config.HIGH_BITS,
    )
    return candidates, bits_by_name, fixed_bytes, original_bytes, projected


def run_gptq(base_dir: Path, bits_by_name: dict) -> dict:
    """Pass 1.5: load the model and GPTQ everything reachable."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compression.calibration import build_calibration, suggest_n_samples
    from compression.modelscan import coverage_report, format_coverage
    from compression.pipeline import gptq_quantize_model

    device = torch.device(config.resolve_device())
    print(f"[compress] loading model for GPTQ (device={device}) ...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(format_coverage(coverage_report(model, config.GROUP_SIZE, config.MIN_NUMEL)))

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)

    n_samples = config.CALIB_SAMPLES
    if n_samples <= 0:
        cfg = model.config
        n_samples = suggest_n_samples(
            n_experts=getattr(cfg, "num_experts", 1) or 1,
            top_k=getattr(cfg, "num_experts_per_tok", 1) or 1,
            hessian_dim=getattr(cfg, "moe_intermediate_size", None)
            or getattr(cfg, "hidden_size", 2048),
            seqlen=config.CALIB_SEQLEN,
        )
        print(f"[compress] auto-selected {n_samples} calibration sequences")

    samples = build_calibration(
        tokenizer,
        n_samples=n_samples,
        seqlen=config.CALIB_SEQLEN,
        calib_file=config.CALIB_FILE,
        seed=config.CALIB_SEED,
        use_chat_template=config.USE_CHAT_TEMPLATE,
    )

    # Non-layer modules go to the device once; decoder layers are moved in and
    # out one at a time by the driver.
    if device.type == "cuda":
        model.to("cpu")
        if hasattr(model, "model"):
            for attr in ("embed_tokens", "norm", "rotary_emb"):
                sub = getattr(model.model, attr, None)
                if sub is not None:
                    sub.to(device)

    payloads = gptq_quantize_model(
        model,
        samples,
        bits_by_name=bits_by_name,
        group_size=config.GROUP_SIZE,
        min_numel=config.MIN_NUMEL,
        device=device,
        hessian_budget_bytes=int(config.HESSIAN_BUDGET_GB * 2**30),
        percdamp=config.PERCDAMP,
        act_order=config.ACT_ORDER,
        blocksize=config.BLOCK_SIZE,
        mse=config.MSE_CLIPPING,
    )

    print(f"[compress] GPTQ produced {len(payloads)} quantized tensors")
    del model, tokenizer, samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payloads


def write_compressed(
    base_dir: Path,
    out_dir: Path,
    model_name: str,
    bits_by_name: dict,
    gptq_payloads: dict,
    original_bytes: int,
) -> dict:
    """Pass 2: assemble the compressed repo from GPTQ payloads plus RTN fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(out_dir, config.SHARD_PREFIX, config.MAX_SHARD_BYTES)
    tensor_meta, n_gptq, n_rtn, n_raw = {}, 0, 0, 0
    t0 = time.time()

    for name, tensor in iter_tensors(base_dir):
        result: QuantResult | None = gptq_payloads.get(name)
        if result is not None:
            n_gptq += 1
        elif name in bits_by_name:
            result = quantize_tensor_rtn(
                tensor, bits_by_name[name], config.GROUP_SIZE, mse=config.MSE_CLIPPING
            )
            result.meta["method"] = "rtn"
            n_rtn += 1
        else:
            writer.add(name, tensor)
            tensor_meta[name] = {
                "mode": "raw",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
            n_raw += 1
            del tensor
            continue

        writer.add(f"{name}::codes", result.codes)
        writer.add(f"{name}::scales", result.scales)
        writer.add(f"{name}::zps", result.zps)
        if result.perm is not None:
            writer.add(f"{name}::perm", result.perm)
        tensor_meta[name] = result.meta
        del tensor

        if (n_gptq + n_rtn + n_raw) % 200 == 0:
            print(
                f"  encoded {n_gptq + n_rtn + n_raw} tensors ({time.time() - t0:.0f}s)",
                flush=True,
            )

    writer.finalize()

    compressed = {
        "format": FORMAT_VERSION,
        "model_name": model_name,
        "method": config.METHOD,
        "group_size": config.GROUP_SIZE,
        "min_numel": config.MIN_NUMEL,
        "target_ratio": config.TARGET_RATIO,
        "act_order": config.ACT_ORDER,
        "mse_clipping": config.MSE_CLIPPING,
        "percdamp": config.PERCDAMP,
        "calib_seqlen": config.CALIB_SEQLEN,
        "n_gptq_tensors": n_gptq,
        "n_rtn_tensors": n_rtn,
        "n_raw_tensors": n_raw,
        "original_total_bytes": original_bytes,
        "compressed_total_bytes": writer.total_bytes,
        "achieved_ratio": writer.total_bytes / max(original_bytes, 1),
        "shard_prefix": config.SHARD_PREFIX,
        "tensors": tensor_meta,
    }
    (out_dir / "compression_config.json").write_text(json.dumps(compressed, indent=2))
    return compressed


def convert_from_hf_checkpoint(
    model_name: str, checkpoint_path: str, output_path: str
) -> dict:
    base_dir = resolve_checkpoint_dir(checkpoint_path, model_name)
    out_dir = Path(output_path).expanduser()

    print(f"[compress] base checkpoint : {base_dir}")
    print(f"[compress] output          : {out_dir}")
    print(
        f"[compress] method {config.METHOD}, group {config.GROUP_SIZE}, "
        f"target ratio {config.TARGET_RATIO} (from compression/config.py)"
    )
    print("[compress] pass 1: planning bit widths ...", flush=True)

    candidates, bits_by_name, fixed_bytes, original_bytes, projected = build_plan(base_dir)
    print(format_plan_report(candidates, bits_by_name, fixed_bytes, original_bytes, projected))

    if projected > config.TARGET_RATIO * original_bytes:
        print(
            "[compress] WARNING: uniform int4 already exceeds the target ratio.",
            file=sys.stderr,
        )

    gptq_payloads = {}
    if config.METHOD == "gptq":
        gptq_payloads = run_gptq(base_dir, bits_by_name)

    print("[compress] pass 2: encoding and writing shards ...", flush=True)
    compressed = write_compressed(
        base_dir, out_dir, model_name, bits_by_name, gptq_payloads, original_bytes
    )

    copied = copy_auxiliary_files(base_dir, out_dir)
    print(f"[compress] copied config/tokenizer files: {', '.join(copied)}")
    print(
        f"[compress] done. {compressed['n_gptq_tensors']} GPTQ / "
        f"{compressed['n_rtn_tensors']} RTN / {compressed['n_raw_tensors']} lossless, "
        f"{compressed['compressed_total_bytes'] / 2**30:.3f} GiB, "
        f"ratio {compressed['achieved_ratio']:.4f}"
    )
    return compressed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face checkpoint into a compressed checkpoint."
    )
    parser.add_argument(
        "--model_name", "--model-name", dest="model_name", type=str, required=True,
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", type=str,
        required=True, help="Path to the base model checkpoint directory.",
    )
    parser.add_argument(
        "--output_path", "--output-path", dest="output_path", type=str, required=True,
        help="Path where the compressed model checkpoint will be saved.",
    )

    args = parser.parse_args()
    convert_from_hf_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()