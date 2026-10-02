"""Per-sample layer replay (from week7): capture the inputs of decoder layer 0 for
every calibration sample (batch 1, no padding, own position ids), then run any
layer over the cache. An attention mask is never built; one is asserted absent.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

_DROP_KWARGS = ("past_key_value", "past_key_values", "cache_position")


class _Abort(Exception):
    pass


def text_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model
    return model


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


@torch.no_grad()
def capture_layer0_inputs(model, layers, samples, device, cache_device) -> Tuple[List, List]:
    hidden, kwargs_list = [], []
    tm = text_model(model)

    def pre_hook(_m, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        kw = {k: v for k, v in kwargs.items() if k != "hidden_states" and k not in _DROP_KWARGS}
        if kw.get("attention_mask") is not None:
            raise RuntimeError("layer 0 received an attention mask; samples must be unpadded")
        hidden.append(h.detach().to(cache_device))
        kwargs_list.append(to_device(kw, cache_device))
        raise _Abort

    handle = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        for s in samples:
            try:
                tm(input_ids=s.to(device), use_cache=False)
            except _Abort:
                pass
    finally:
        handle.remove()
    if len(hidden) != len(samples):
        raise RuntimeError(f"captured {len(hidden)} of {len(samples)} samples")
    return hidden, kwargs_list


@torch.no_grad()
def run_layer(layer, hidden, kwargs_list, device, update=False):
    """Forward every cached sample through ``layer``; with update=True replace the cache."""
    for i in range(len(hidden)):
        h = hidden[i]
        out = layer(h.to(device), **to_device(kwargs_list[i], device))
        if isinstance(out, tuple):
            out = out[0]
        if update:
            hidden[i] = out.detach().to(h.device)
        del out
