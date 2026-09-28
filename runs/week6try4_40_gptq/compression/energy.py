"""Activation-energy pre-pass for the bit-width planner.

For a linear ``y = W x`` quantized to ``W + dW``, the expected output error is

    E ||dW x||^2 = sum_ij dW_ij^2 E[x_j^2]          (off-diagonal terms ~ 0)
                ~= ||dW||^2 * mean_j E[x_j^2]        (scalar approximation)

so a tensor's weight error only matters in proportion to how much energy its
input carries. The diagnostics notebook showed (a) the data-free week6 score
is rank-identical to "mean squared weight", (b) weighting by E[x^2] halves the
knapsack objective at the same budget, and (c) the per-tensor SCALAR mean gives
exactly the same plan as the full per-channel diagonal. So one scalar per
linear is all we measure here.

Measured on the bf16 model, before any quantization, on a seeded subset of the
calibration samples (``ENERGY_MAX_TOKENS``). The lm_head input is the final
normed hidden state; with tied embeddings, the embedding matrix inherits that
energy, since its error reaches the loss through the lm_head.
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch
import torch.nn as nn

from compression.modelscan import find_decoder_layers, make_key_resolver
from compression.pipeline import text_model


@torch.no_grad()
def collect_energy(
    model: nn.Module,
    samples: List[torch.Tensor],
    ckpt_keys,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, float]:
    """Mean ``E[x^2]`` of the input of every decoder linear, keyed by ckpt name.

    Also returns the special key ``"__final_hidden__"`` (lm_head input) and
    ``"__layer0_input__"`` (embedding output), used for the embedding matrix.
    Asserts that no layer ever receives an attention mask.
    """
    layers_attr, layers = find_decoder_layers(model)
    resolve = make_key_resolver(ckpt_keys, n_layers=len(layers))
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def add(key: str, x: torch.Tensor):
        v = torch.linalg.vector_norm(x, dtype=torch.float32) ** 2
        if key in sums:
            sums[key] += v.double()
        else:
            sums[key] = v.double()
        counts[key] = counts.get(key, 0) + x.numel()

    for idx, layer in enumerate(layers):
        for name, mod in layer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            key = resolve(layers_attr, idx, name)
            if key is None:
                continue

            def hook(_m, inputs, _k=key):
                add(_k, inputs[0].detach())

            handles.append(mod.register_forward_pre_hook(hook))

        def mask_check(_m, args, kwargs, _i=idx):
            if kwargs.get("attention_mask") is not None:
                raise RuntimeError(
                    f"decoder layer {_i} received a non-None attention_mask; "
                    "calibration samples must run unpadded at batch size 1."
                )
            if _i == 0:
                h = args[0] if args else kwargs["hidden_states"]
                add("__layer0_input__", h.detach())

        handles.append(layer.register_forward_pre_hook(mask_check, with_kwargs=True))

    tm = text_model(model)
    t0 = time.time()
    n_tok = 0
    try:
        for i, sample in enumerate(samples):
            out = tm(input_ids=sample.to(device), use_cache=False)
            add("__final_hidden__", out.last_hidden_state.detach())
            n_tok += sample.shape[1]
            del out
            if verbose and (i + 1) % 10 == 0:
                print(
                    f"  [energy] {i + 1}/{len(samples)} samples, {n_tok:,} tokens "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    finally:
        for h in handles:
            h.remove()

    energy = {k: float(sums[k].item() / max(counts[k], 1)) for k in sums}
    if verbose:
        print(
            f"[energy] {len(energy)} inputs measured over {len(samples)} samples / "
            f"{n_tok:,} tokens in {time.time() - t0:.0f}s",
            flush=True,
        )
    return energy


def energy_for_candidates(candidates, energy: Dict[str, float]) -> Dict[str, float]:
    """Map every planner candidate to an input energy.

    * decoder linears: their measured value
    * ``lm_head.weight``: the final hidden state's energy
    * ``embed_tokens.weight``: the final hidden state's energy if embeddings are
      tied (no separate lm_head in the checkpoint), otherwise the energy of the
      layer-0 input (its error then enters the residual stream directly)
    * anything unmeasured: the median of the measured values, with a warning
    """
    names = {c["name"] for c in candidates}
    has_lm_head = any(n.endswith("lm_head.weight") for n in names)
    measured = [v for k, v in energy.items() if not k.startswith("__")]
    median = sorted(measured)[len(measured) // 2] if measured else 1.0

    out, missing = {}, []
    for c in candidates:
        n = c["name"]
        if n in energy:
            out[n] = energy[n]
        elif n.endswith("lm_head.weight"):
            out[n] = energy["__final_hidden__"]
        elif n.endswith("embed_tokens.weight"):
            out[n] = energy["__layer0_input__"] if has_lm_head else energy["__final_hidden__"]
        else:
            out[n] = median
            missing.append(n)
    if missing:
        print(
            f"[energy] WARNING: {len(missing)} candidates had no measured energy and "
            f"use the median ({median:.4g}); first: {missing[0]}"
        )
    return out
