"""Sequential GPTQ driver -- variable-length samples, one conversation each.

1. Capture the hidden states entering layer 0, and the keyword arguments the
   model passes to its layers, for EVERY sample separately. Samples have
   different lengths, so their rotary embeddings and position ids differ; a
   single shared kwargs dict (week6) is only correct for fixed-length windows.
2. For each layer: hook its linears, replay that layer over every cached
   sample to accumulate Hessians, run GPTQ, write the dequantized weights back.
3. Replay the now-quantized layer to produce the next layer's inputs, so each
   layer is fit against the degraded activations it will really see.

No attention mask is ever built: every sample is run alone (batch 1, no
padding), so full-attention layers take torch SDPA's ``is_causal`` path and
Gated DeltaNet layers see no padding mask. This is asserted, not assumed --
a non-None mask would mean the per-sample kwargs are not what we think.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from compression.gptq import GPTQQuantizer
from compression.modelscan import (
    chunk_by_memory,
    find_decoder_layers,
    linear_targets,
    make_key_resolver,
)
from compression.quantize import QuantResult

_DROP_KWARGS = ("past_key_value", "past_key_values", "cache_position")


class _CatcherAbort(Exception):
    pass


def text_model(model: nn.Module) -> nn.Module:
    """The decoder-only text model (``model.model``), so lm_head never runs."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model
    return model


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


@torch.no_grad()
def capture_layer0_inputs(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: List[torch.Tensor],
    device: torch.device,
    cache_device: str | torch.device,
) -> Tuple[List[torch.Tensor], List[dict]]:
    """Per-sample (hidden_states, layer kwargs) entering decoder layer 0."""
    hidden: List[torch.Tensor] = []
    kwargs_list: List[dict] = []
    tm = text_model(model)

    def pre_hook(_mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        kw = {k: v for k, v in kwargs.items() if k != "hidden_states" and k not in _DROP_KWARGS}
        if kw.get("attention_mask") is not None:
            raise RuntimeError(
                "Layer 0 received a non-None attention_mask. Samples are run one at "
                "a time without padding, so no mask should exist; refusing to cache "
                "kwargs that would be wrong for the other samples."
            )
        hidden.append(h.detach().to(cache_device))
        kwargs_list.append(_to_device(kw, cache_device))
        raise _CatcherAbort

    handle = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        for sample in samples:
            try:
                tm(input_ids=sample.to(device), use_cache=False)
            except _CatcherAbort:
                pass
    finally:
        handle.remove()

    if len(hidden) != len(samples):
        raise RuntimeError(
            f"captured {len(hidden)} of {len(samples)} samples; the decoder stack "
            "was not reached for some inputs."
        )
    return hidden, kwargs_list


@torch.no_grad()
def run_layer(
    layer: nn.Module,
    hidden: List[torch.Tensor],
    kwargs_list: List[dict],
    device: torch.device,
    keep_output: bool = True,
    out_device: str | torch.device | None = None,
) -> List[torch.Tensor] | None:
    out: List[torch.Tensor] = [] if keep_output else None
    for h, kw in zip(hidden, kwargs_list):
        result = layer(h.to(device), **_to_device(kw, device))
        if isinstance(result, tuple):
            result = result[0]
        if keep_output:
            out.append(result.detach().to(out_device or h.device))
        del result
    return out


@torch.no_grad()
def gptq_quantize_model(
    model: nn.Module,
    samples: List[torch.Tensor],
    bits_by_name: Dict[str, int],
    group_size: int,
    min_numel: int,
    device: torch.device,
    cache_device: str | torch.device = "cuda",
    hessian_budget_bytes: int = 8 * 2**30,
    percdamp: float = 0.01,
    act_order: bool = True,
    blocksize: int = 128,
    mse: bool = True,
    verbose: bool = True,
) -> Dict[str, QuantResult]:
    """GPTQ every reachable linear. Returns payloads keyed by checkpoint name."""
    layers_attr, layers = find_decoder_layers(model)
    model.config.use_cache = False
    resolve_key = make_key_resolver(bits_by_name.keys(), n_layers=len(layers))

    t_cap = time.time()
    hidden, kwargs_list = capture_layer0_inputs(model, layers, samples, device, cache_device)
    n_tok = sum(h.shape[1] for h in hidden)
    if verbose:
        print(
            f"[gptq] captured {len(hidden)} samples / {n_tok:,} tokens "
            f"({time.time() - t_cap:.0f}s), layer kwargs: {sorted(kwargs_list[0].keys())}, "
            f"cache on {cache_device}",
            flush=True,
        )

    payloads: Dict[str, QuantResult] = {}
    t_start = time.time()

    for layer_idx, layer in enumerate(layers):
        t_layer = time.time()
        layer.to(device)
        found = linear_targets(layer, group_size, min_numel)
        targets, keys = {}, {}
        for mod_name, module in found.items():
            key = resolve_key(layers_attr, layer_idx, mod_name)
            if key is not None and key in bits_by_name:
                targets[mod_name] = module
                keys[mod_name] = key

        if layer_idx == 0 and found and not targets:
            sample_ckpt = sorted(bits_by_name.keys())[:5]
            raise RuntimeError(
                f"GPTQ found {len(found)} quantizable linears in layer 0 but could not "
                "match any of them to a checkpoint tensor.\n"
                f"  module path built : {layers_attr}.0.{next(iter(found))}.weight\n"
                f"  checkpoint keys   : {sample_ckpt}"
            )

        for chunk in chunk_by_memory(targets, hessian_budget_bytes):
            quantizers: Dict[str, GPTQQuantizer] = {}
            handles = []
            for name in chunk:
                module = targets[name]
                quantizers[name] = GPTQQuantizer(module.weight, device)

                def make_hook(key: str):
                    def hook(_mod, inputs, _out):
                        quantizers[key].add_batch(inputs[0].detach())

                    return hook

                handles.append(module.register_forward_hook(make_hook(name)))

            run_layer(layer, hidden, kwargs_list, device, keep_output=False)
            for handle in handles:
                handle.remove()

            for name in chunk:
                module = targets[name]
                full = keys[name]
                result, dequantized = quantizers[name].quantize(
                    module.weight,
                    bits=bits_by_name[full],
                    group_size=group_size,
                    percdamp=percdamp,
                    act_order=act_order,
                    blocksize=blocksize,
                    mse=mse,
                )
                module.weight.data.copy_(dequantized)
                payloads[full] = result
                quantizers[name].free()

            del quantizers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Propagate the *quantized* layer's outputs to the next layer, replacing
        # the cache sample by sample so peak memory is one extra sample, not a
        # second full copy.
        for i in range(len(hidden)):
            h_in = hidden[i]
            res = layer(h_in.to(device), **_to_device(kwargs_list[i], device))
            if isinstance(res, tuple):
                res = res[0]
            hidden[i] = res.detach().to(h_in.device)
            del h_in, res

        layer.to(device)  # stays wherever the model lives
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            bits_note = ",".join(
                f"{n.split('.')[-1]}:{bits_by_name[keys[n]]}" for n in targets
            )
            elapsed = time.time() - t_start
            eta = elapsed / (layer_idx + 1) * (len(layers) - layer_idx - 1)
            print(
                f"[gptq] layer {layer_idx + 1:>2}/{len(layers)} "
                f"({time.time() - t_layer:5.0f}s, elapsed {elapsed / 60:5.1f} min, "
                f"eta {eta / 60:5.1f} min) [{bits_note}]",
                flush=True,
            )

    del hidden, kwargs_list
    return payloads
