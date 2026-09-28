"""Model loading and the diagonal-Fisher pass.

Fisher ``F_ij = E[(dL/dw_ij)^2]`` of the trace NLL, one value per weight. The
planner uses ``0.5 * sum F * dW^2`` as the predicted loss increase of a
quantization choice; in codec_lab2 this proxy ranked 8 configurations in the
same order as the measured KL.

No flash kernels: Gated DeltaNet is pinned to transformers' PyTorch path and
full attention uses torch SDPA.
"""

from __future__ import annotations

import re
import time

import torch


def force_torch_kernels(model):
    n = 0
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            mod = __import__(type(m).__module__, fromlist=["x"])
            m.causal_conv1d_fn = None
            m.chunk_gated_delta_rule = mod.torch_chunk_gated_delta_rule
            if hasattr(mod, "torch_recurrent_gated_delta_rule"):
                m.recurrent_gated_delta_rule = mod.torch_recurrent_gated_delta_rule
            if hasattr(mod, "torch_causal_conv1d_update"):
                m.causal_conv1d_update = mod.torch_causal_conv1d_update
            n += 1
    return n


def load_model(base_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    n = force_torch_kernels(model)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    return model, tok, n


_TAIL = re.compile(r"((?:layers\.\d+\..*)|(?:embed_tokens\.weight)|(?:lm_head\.weight))$")


def tail(name):
    m = _TAIL.search(name)
    return m.group(1) if m else name


def map_params(model, ckpt_names):
    """ckpt name -> live model parameter (matched on the part after the prefix)."""
    by_tail = {}
    for n, p in model.named_parameters():
        by_tail.setdefault(tail(n), p)
    out, missing = {}, []
    for c in ckpt_names:
        p = by_tail.get(tail(c))
        if p is None or tuple(p.shape) != tuple(ckpt_names[c]):
            missing.append(c)
        else:
            out[c] = p
    return out, missing


def collect_fisher(model, params, samples, device, log=print):
    """params: ckpt name -> parameter. Returns ckpt name -> bf16 Fisher tensor."""
    live = {id(p) for p in params.values()}
    for p in model.parameters():
        p.requires_grad_(id(p) in live)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as exc:
        log(f"  (gradient checkpointing unavailable: {exc!r})")
    model.train()
    acc = {c: torch.zeros_like(p, dtype=torch.float32) for c, p in params.items()}
    t0 = time.time()
    try:
        for i, (ids, start) in enumerate(samples):
            ids = ids.to(device)
            logits = model(input_ids=ids, use_cache=False).logits[0, start - 1:-1].float()
            loss = torch.nn.functional.cross_entropy(logits, ids[0, start:])
            loss.backward()
            for c, p in params.items():
                if p.grad is not None:
                    acc[c].add_(p.grad.float().pow(2))
            for p in params.values():
                p.grad = None
            del logits, loss
            if (i + 1) % 8 == 0 or i + 1 == len(samples):
                log(f"  [fisher] {i + 1}/{len(samples)} samples ({time.time() - t0:.0f}s)")
    finally:
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.eval()
    out = {}
    for c in list(acc):
        out[c] = (acc.pop(c) / max(len(samples), 1)).to(torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
