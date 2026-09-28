"""Environment checks that run before any expensive work.

molab (RTX Pro 6000 Blackwell, sm120) has repeatedly failed with third-party
attention kernels -- flash-attn, flash-linear-attention, causal-conv1d,
flashinfer. This pipeline needs none of them:

* Gated DeltaNet layers run transformers' pure-PyTorch ``torch_chunk_gated_delta_rule``
  (exact; a Python loop over 64-token chunks, so linear in sequence length).
* Full-attention layers run ``torch.nn.functional.scaled_dot_product_attention``,
  which dispatches to torch's OWN built-in flash / memory-efficient kernels
  (compiled into the torch wheel -- nothing extra to install).

``force_torch_kernels`` pins the torch GDN path even if fla or causal-conv1d
are importable. ``probe_long_sequence`` then pushes one full-attention layer
and one Gated DeltaNet layer through a sequence as long as the longest
calibration sample, so an out-of-memory or kernel failure surfaces in the
first minute instead of hours in.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def environment_report() -> str:
    import transformers

    lines = ["", "=" * 72, "Environment", "=" * 72]
    lines.append(f"  torch                   : {torch.__version__} (cuda {torch.version.cuda})")
    lines.append(f"  transformers            : {transformers.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines.append(
            f"  gpu                     : {p.name}, sm{p.major}{p.minor}, "
            f"{p.total_memory / 2**30:.1f} GiB"
        )
    else:
        lines.append("  gpu                     : NONE (CPU run -- only for tiny tests)")
    for mod in ("fla", "causal_conv1d", "flash_attn", "flashinfer"):
        try:
            __import__(mod)
            state = "importable (NOT used)"
        except Exception:
            state = "not installed (fine)"
        lines.append(f"  {mod:<24}: {state}")
    try:
        import psutil

        lines.append(f"  host RAM                : {psutil.virtual_memory().total / 2**30:.1f} GiB")
    except Exception:
        pass
    lines.append("=" * 72)
    return "\n".join(lines)


def force_torch_kernels(model: nn.Module) -> dict:
    """Route every Gated DeltaNet module through the pure-PyTorch path."""
    counts = {"gdn_modules": 0, "norm_swapped": 0}
    mod_cls = None
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            mod_cls = __import__(type(m).__module__, fromlist=["x"])
            break
    if mod_cls is None:
        return counts

    torch_chunk = getattr(mod_cls, "torch_chunk_gated_delta_rule")
    torch_recur = getattr(mod_cls, "torch_recurrent_gated_delta_rule", None)
    torch_conv_update = getattr(mod_cls, "torch_causal_conv1d_update", None)
    gated_norm_cls = next(
        (getattr(mod_cls, n) for n in dir(mod_cls) if n.endswith("RMSNormGated")), None
    )

    for m in model.modules():
        if not type(m).__name__.endswith("GatedDeltaNet"):
            continue
        counts["gdn_modules"] += 1
        m.causal_conv1d_fn = None
        m.chunk_gated_delta_rule = torch_chunk
        if torch_recur is not None:
            m.recurrent_gated_delta_rule = torch_recur
        if torch_conv_update is not None:
            m.causal_conv1d_update = torch_conv_update
        # With fla installed, the gated norm is fla's FusedRMSNormGated (a Triton
        # kernel). Swap in the torch module; the math and the weight are the same.
        if gated_norm_cls is not None and not isinstance(m.norm, gated_norm_cls):
            old = m.norm
            new = gated_norm_cls(old.weight.shape[0], eps=getattr(old, "eps", 1e-6))
            new.weight.data.copy_(old.weight.data)
            m.norm = new.to(device=old.weight.device, dtype=old.weight.dtype)
            counts["norm_swapped"] += 1
    return counts


def sdpa_backends(seq: int = 4096, head_dim: int = 256, heads: int = 16) -> str:
    """Which of torch's built-in SDPA kernels accept this head_dim on this GPU."""
    if not torch.cuda.is_available():
        return "cpu"
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
    res = []
    for name, be in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ):
        try:
            with sdpa_kernel([be]):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
            res.append(f"{name}=ok")
        except Exception:
            res.append(f"{name}=no")
    del q
    torch.cuda.empty_cache()
    return ", ".join(res)


@torch.no_grad()
def probe_long_sequence(model: nn.Module, seq_len: int, device: torch.device) -> dict:
    """Time + peak memory of one GDN and one full-attention layer at ``seq_len``."""
    from compression.modelscan import find_decoder_layers
    from compression.pipeline import capture_layer0_inputs, run_layer

    _, layers = find_decoder_layers(model)
    types = getattr(model.config, "layer_types", None) or getattr(
        getattr(model.config, "text_config", None), "layer_types", None
    )
    idx_gdn = types.index("linear_attention") if types and "linear_attention" in types else 0
    idx_full = types.index("full_attention") if types and "full_attention" in types else None

    vocab = model.get_input_embeddings().weight.shape[0]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, min(vocab, 150000), (1, seq_len), generator=g)
    hidden, kwargs = capture_layer0_inputs(model, layers, [ids], device, device)

    out = {"seq_len": seq_len}
    for tag, idx in (("gdn", idx_gdn), ("full_attn", idx_full)):
        if idx is None:
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t0 = time.time()
        run_layer(layers[idx], hidden, kwargs, device, keep_output=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            out[f"{tag}_peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2**30
        out[f"{tag}_sec"] = time.time() - t0
        out[f"{tag}_layer"] = idx
    del hidden, kwargs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
