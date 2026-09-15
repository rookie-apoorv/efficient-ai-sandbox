"""Sequential GPTQ driver.

Quantizing a 4B model needs the input activations of every linear layer, but
materialising them all at once is impossible -- an MoE layer alone can need
tens of GiB of Hessians.  The standard solution, used here, walks the decoder
stack one layer at a time:

1. Capture the hidden states entering layer 0 (and the keyword arguments the
   model passes to its layers) by running the calibration batches through the
   model with layer 0 replaced by a catcher that records and aborts.
2. For each layer: hook its linears, replay that single layer's forward over
   the cached hidden states to accumulate Hessians, run GPTQ, and write the
   dequantized weights back into the live module.
3. Replay the layer once more -- now quantized -- to produce the hidden states
   entering the next layer.

Step 3 is what makes this work better than quantizing layers independently:
each layer is fit against the *already degraded* activations it will really
see at inference, so errors are partly cancelled downstream instead of
accumulating.

Peak memory is one layer's Hessian chunk plus the cached hidden states
(``n_samples x seqlen x hidden x 2`` bytes, a few hundred MiB at typical
settings), not the whole model's worth of statistics.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from compression.gptq import GPTQQuantizer
from compression.modelscan import chunk_by_memory, find_decoder_layers, linear_targets
from compression.quantize import QuantResult


class _CatcherAbort(Exception):
    pass


class _Catcher(nn.Module):
    """Records the inputs and kwargs of the first decoder layer, then aborts."""

    def __init__(self, inner: nn.Module, store: list, kwargs_store: dict):
        super().__init__()
        self.inner = inner
        self.store = store
        self.kwargs_store = kwargs_store

    def forward(self, hidden_states, **kwargs):
        self.store.append(hidden_states.detach().cpu())
        if not self.kwargs_store:
            self.kwargs_store.update(kwargs)
        raise _CatcherAbort


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def capture_layer0_inputs(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[torch.Tensor], dict]:
    hidden: List[torch.Tensor] = []
    kwargs: dict = {}

    layers[0] = _Catcher(layers[0], hidden, kwargs)
    for sample in samples:
        try:
            model(sample.to(device))
        except _CatcherAbort:
            pass
        except Exception as exc:  # pragma: no cover - surfaces as a clear message
            layers[0] = layers[0].inner
            raise RuntimeError(
                f"Calibration forward pass failed: {exc}. If this model needs "
                "extra inputs, use --method rtn or an external GPTQ toolchain."
            ) from exc
    layers[0] = layers[0].inner

    if not hidden:
        raise RuntimeError("No hidden states captured; the decoder stack was never called.")
    kwargs.pop("past_key_value", None)
    kwargs.pop("past_key_values", None)
    return hidden, kwargs


@torch.no_grad()
def run_layer(
    layer: nn.Module, hidden: List[torch.Tensor], kwargs: dict, device: torch.device
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    moved = _to_device(kwargs, device)
    for h in hidden:
        result = layer(h.to(device), **moved)
        if isinstance(result, tuple):
            result = result[0]
        out.append(result.detach().cpu())
    return out


@torch.no_grad()
def gptq_quantize_model(
    model: nn.Module,
    samples: List[torch.Tensor],
    bits_by_name: Dict[str, int],
    group_size: int,
    min_numel: int,
    device: torch.device,
    hessian_budget_bytes: int = 8 * 2**30,
    percdamp: float = 0.01,
    act_order: bool = True,
    blocksize: int = 128,
    mse: bool = True,
    verbose: bool = True,
) -> Dict[str, QuantResult]:
    """GPTQ every reachable linear. Returns payloads keyed by state-dict name."""
    layers_attr, layers = find_decoder_layers(model)
    model.config.use_cache = False

    hidden, layer_kwargs = capture_layer0_inputs(model, layers, samples, device)
    if verbose:
        print(
            f"[gptq] captured {len(hidden)} calibration batches, "
            f"layer kwargs: {sorted(layer_kwargs.keys())}",
            flush=True,
        )

    payloads: Dict[str, QuantResult] = {}
    t_start = time.time()

    for layer_idx, layer in enumerate(layers):
        layer.to(device)
        targets = linear_targets(layer, group_size, min_numel)
        # Only quantize what the planner actually assigned a bit width to.
        targets = {
            n: m
            for n, m in targets.items()
            if f"{layers_attr}.{layer_idx}.{n}.weight" in bits_by_name
        }

        if targets:
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

                run_layer(layer, hidden, layer_kwargs, device)
                for handle in handles:
                    handle.remove()

                for name in chunk:
                    module = targets[name]
                    full = f"{layers_attr}.{layer_idx}.{name}.weight"
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

        # Propagate the *quantized* layer's outputs to the next layer.
        hidden = run_layer(layer, hidden, layer_kwargs, device)
        layer.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print(
                f"[gptq] layer {layer_idx + 1}/{len(layers)} done "
                f"({len(targets)} linears, {time.time() - t_start:.0f}s elapsed)",
                flush=True,
            )

    return payloads