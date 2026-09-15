"""Discovering what GPTQ can actually reach in a given model.

GPTQ needs the *inputs* to each weight matrix, which it gets by hooking
``nn.Linear`` modules.  That works for attention projections, dense MLPs and
MoE experts implemented as individual ``Linear`` layers.

It does **not** work for two things:

* ``nn.Embedding`` -- a gather, not a matmul, so there is no input activation
  to build a Hessian from.  (Quantizing it saves memory but never saves time.)
* **Fused expert stacks.**  Recent ``transformers`` versions implement Qwen3-MoE
  experts as 3-D ``nn.Parameter`` tensors (``experts.gate_up_proj`` of shape
  ``[E, hidden, 2*inter]``) consumed by a batched matmul rather than by ``E``
  separate ``Linear`` modules.  There is no module to hook, and recovering each
  expert's private input slice would mean reimplementing the routing.

Anything GPTQ cannot reach falls back to RTN with MSE clipping.  Since experts
are the bulk of an MoE model's parameters, **the coverage number this module
reports is the single most important diagnostic in the pipeline** -- run
``--dry-run`` and read it before committing to a long quantization run.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def find_decoder_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """Locate the transformer block list without hard-coding an architecture.

    Returns the longest ``nn.ModuleList`` in the model, which for every decoder
    LM is the stack of decoder layers.
    """
    best_name, best = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            if best is None or len(module) > len(best):
                best_name, best = name, module
    if best is None:
        raise RuntimeError(
            "Could not locate a decoder layer stack (no nn.ModuleList found). "
            "Use --method rtn, or quantize with GPTQModel / llm-compressor."
        )
    return best_name, best


def linear_targets(
    layer: nn.Module, group_size: int, min_numel: int
) -> Dict[str, nn.Linear]:
    """Every ``nn.Linear`` inside one decoder layer worth quantizing.

    Small modules -- MoE routers above all -- are excluded.  A router's weight
    is a few hundred KB but decides *which experts fire*; perturbing it changes
    the routing itself, which is far more damaging than a small error inside an
    expert. Those stay at full precision.
    """
    out: Dict[str, nn.Linear] = {}
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < group_size or module.weight.numel() < min_numel:
            continue
        out[name] = module
    return out


def find_fused_expert_params(layer: nn.Module) -> Dict[str, torch.Tensor]:
    """3-D parameters that look like fused expert stacks (GPTQ cannot reach)."""
    out: Dict[str, torch.Tensor] = {}
    for name, param in layer.named_parameters(recurse=True):
        if param.ndim == 3 and param.numel() > (1 << 20):
            out[name] = param
    return out


def hessian_bytes(module: nn.Linear) -> int:
    """float32 Hessian footprint for one linear: in_features^2 * 4."""
    return module.in_features * module.in_features * 4


def chunk_by_memory(
    targets: Dict[str, nn.Linear], budget_bytes: int
) -> List[List[str]]:
    """Split a layer's linears into groups that fit a Hessian memory budget.

    An MoE layer with 128 experts can need tens of GiB of Hessians if every
    expert is hooked at once, so the layer's forward pass is simply replayed
    once per chunk. Replaying one layer is cheap compared with a full model
    forward.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for name, module in targets.items():
        need = hessian_bytes(module)
        if current and current_bytes + need > budget_bytes:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += need
    if current:
        chunks.append(current)
    return chunks


def coverage_report(
    model: nn.Module, group_size: int, min_numel: int
) -> dict:
    """Pre-flight: how much of the model will GPTQ actually touch?"""
    layers_name, layers = find_decoder_layers(model)

    total_params = sum(p.numel() for p in model.parameters())
    gptq_params = 0
    fused_params = 0
    n_linears = 0
    max_hessian = 0

    for layer in layers:
        targets = linear_targets(layer, group_size, min_numel)
        n_linears += len(targets)
        for module in targets.values():
            gptq_params += module.weight.numel()
            max_hessian = max(max_hessian, hessian_bytes(module))
        for param in find_fused_expert_params(layer).values():
            fused_params += param.numel()

    embed_params = sum(
        m.weight.numel() for m in model.modules() if isinstance(m, nn.Embedding)
    )

    return {
        "layers_attr": layers_name,
        "n_layers": len(layers),
        "n_linear_targets_per_layer": n_linears // max(len(layers), 1),
        "total_params": total_params,
        "gptq_params": gptq_params,
        "fused_expert_params": fused_params,
        "embedding_params": embed_params,
        "gptq_coverage": gptq_params / max(total_params, 1),
        "largest_hessian_bytes": max_hessian,
    }


def format_coverage(report: dict) -> str:
    cov = report["gptq_coverage"]
    lines = [
        "",
        "=" * 72,
        "GPTQ coverage pre-flight",
        "=" * 72,
        f"  decoder stack           : {report['layers_attr']} "
        f"({report['n_layers']} layers)",
        f"  nn.Linear targets/layer : {report['n_linear_targets_per_layer']}",
        f"  total parameters        : {report['total_params'] / 1e9:8.3f} B",
        f"  reachable by GPTQ       : {report['gptq_params'] / 1e9:8.3f} B "
        f"({cov * 100:.1f}%)",
        f"  fused expert stacks     : {report['fused_expert_params'] / 1e9:8.3f} B "
        f"(RTN fallback)",
        f"  embeddings / lm_head    : {report['embedding_params'] / 1e9:8.3f} B "
        f"(RTN, gather not GEMM)",
        f"  largest Hessian         : {report['largest_hessian_bytes'] / 2**20:8.1f} MiB",
        "=" * 72,
    ]
    if cov < 0.40:
        lines += [
            "  WARNING: GPTQ reaches under 40% of the parameters. The experts are",
            "  almost certainly fused 3-D tensors in this transformers version, so",
            "  most of the model will fall back to RTN and the accuracy gain will",
            "  be small. Options: pin an older transformers that builds experts as",
            "  nn.Linear, or quantize with GPTQModel / llm-compressor, which",
            "  implement fused-MoE GPTQ directly.",
            "=" * 72,
        ]
    return "\n".join(lines)