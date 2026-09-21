# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "datasets",
#     "pandas",
#     "accelerate",
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import time
    from contextlib import contextmanager

    import marimo as mo
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        F,
        contextmanager,
        json,
        load_dataset,
        mo,
        pd,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Qwen3.5-4B Quantization Sensitivity Analysis

        Fake-quantizes each weight tensor one at a time (symmetric, per-group,
        round-to-nearest) and measures the resulting calibration-loss delta,
        with everything else left at bf16. Produces a per-tensor, per-bit-width
        sensitivity ranking across MLP / linear-attention / full-attention /
        embed-lm_head, for joint bit-budget allocation at your 40% / 20% / 10%
        compression targets.

        Qwen3.5 requires `transformers` installed from source (declared in the
        script header above) since it isn't in a stable PyPI release yet.

        Run the cells top to bottom. Loading the model and running the full
        sweep are gated behind buttons since they're the expensive steps.
        """
    )
    return


@app.cell
def _(mo):
    hf_token = mo.ui.text(
        value="", label="HF token (optional, only needed for gated repos)", kind="password"
    )
    model_path = mo.ui.text(
        value="Qwen/Qwen3.5-4B", label="Model path or HF repo id", full_width=True
    )
    calib_dataset = mo.ui.dropdown(
        options=[
            "AI-MO/aimo-validation-aime",
            "HuggingFaceH4/aime_2024",
            "Maxwell-Jia/AIME_2024",
        ],
        value="AI-MO/aimo-validation-aime",
        label="Calibration dataset (math-domain, from AIME)",
    )
    bits_list = mo.ui.multiselect(
        options=["8", "6", "4", "3", "2"],
        value=["8", "4", "3", "2"],
        label="Bit-widths to test per tensor",
    )
    group_size = mo.ui.slider(start=32, stop=256, step=32, value=128, label="Quant group size")
    calib_seqs = mo.ui.slider(
        start=4, stop=128, step=4, value=32, label="# calibration sequences"
    )
    seq_len = mo.ui.slider(start=256, stop=4096, step=256, value=1024, label="Sequence length")
    device = mo.ui.text(value="cuda", label="Device")
    out_path = mo.ui.text(value="sensitivity_results.jsonl", label="Output JSONL path")
    return (
        bits_list,
        calib_dataset,
        calib_seqs,
        device,
        group_size,
        hf_token,
        model_path,
        out_path,
        seq_len,
    )


@app.cell
def _(
    bits_list,
    calib_dataset,
    calib_seqs,
    device,
    group_size,
    hf_token,
    mo,
    model_path,
    out_path,
    seq_len,
):
    mo.vstack(
        [
            hf_token,
            model_path,
            calib_dataset,
            bits_list,
            group_size,
            calib_seqs,
            seq_len,
            device,
            out_path,
        ]
    )
    return


@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="1. Load model + tokenizer")
    load_button
    return (load_button,)


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, device, hf_token, load_button, mo, model_path, torch):
    mo.stop(not load_button.value, mo.md("Click **Load model + tokenizer** above to begin."))

    if hf_token.value:
        from huggingface_hub import login as _hf_login

        _hf_login(token=hf_token.value)

    tokenizer = AutoTokenizer.from_pretrained(model_path.value)
    model = AutoModelForCausalLM.from_pretrained(
        model_path.value, dtype=torch.bfloat16, device_map=device.value
    )
    model.eval()
    mo.md(
        f"Loaded `{model_path.value}` "
        f"({sum(_p.numel() for _p in model.parameters()):,} total params)."
    )
    return model, tokenizer


@app.cell
def _(load_dataset, torch):
    def build_calibration_text(dataset_name: str, n_seqs: int) -> list[str]:
        """Math-domain calibration text: AIME problems + worked solutions."""
        ds = load_dataset(dataset_name, split="train")
        texts = []
        for row in ds:
            problem = row.get("problem") or row.get("Problem") or ""
            solution = row.get("solution") or row.get("Solution") or ""
            texts.append(f"Problem: {problem}\n\nSolution: {solution}")
        if not texts:
            raise RuntimeError(f"No calibration texts loaded from {dataset_name}.")
        out = []
        i = 0
        while len(out) < n_seqs:
            out.append(texts[i % len(texts)])
            i += 1
        return out[:n_seqs]

    def tokenize_calibration(tokenizer, texts, seq_len, device):
        batches = []
        for t in texts:
            enc = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
            batches.append({k: v.to(device) for k, v in enc.items()})
        return batches

    @torch.no_grad()
    def calibration_loss(model, batches) -> float:
        total_loss, total_tokens = 0.0, 0
        for batch in batches:
            input_ids = batch["input_ids"]
            attn_mask = batch.get("attention_mask")
            labels = input_ids.clone()
            if attn_mask is not None:
                labels[attn_mask == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            n_tok = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
        return total_loss / max(total_tokens, 1)

    return build_calibration_text, calibration_loss, tokenize_calibration


@app.cell
def _(F, contextmanager, torch):
    def quantize_dequantize(w: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
        """Symmetric per-group RTN fake quantization."""
        orig_shape = w.shape
        orig_dtype = w.dtype
        w = w.float()

        out_features, in_features = w.shape
        gs = group_size if group_size > 0 else in_features
        pad = (-in_features) % gs
        if pad:
            w = F.pad(w, (0, pad))
        n_groups = w.shape[1] // gs
        w_g = w.view(out_features, n_groups, gs)

        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        max_abs = w_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = max_abs / qmax

        q = torch.clamp(torch.round(w_g / scale), qmin, qmax)
        deq = (q * scale).view(out_features, w.shape[1])
        if pad:
            deq = deq[:, :in_features]
        return deq.to(orig_dtype).view(orig_shape)

    @contextmanager
    def fake_quantized(module, bits, group_size):
        original = module.weight.data.clone()
        try:
            module.weight.data = quantize_dequantize(module.weight.data, bits, group_size)
            yield
        finally:
            module.weight.data = original

    return fake_quantized, quantize_dequantize


@app.cell
def _(torch):
    def categorize(name: str) -> str:
        if "mtp" in name:
            return "mtp"
        if "visual" in name:
            return "visual"
        if "embed_tokens" in name or "lm_head" in name:
            return "embed_lm_head"
        if "linear_attn" in name:
            if any(s in name for s in ("A_log", "dt_bias", "conv1d", "norm")):
                return "linear_attn_control"
            return "linear_attn_proj"
        if "self_attn" in name:
            if "norm" in name:
                return "full_attn_norm"
            return "full_attn_proj"
        if "mlp" in name:
            return "mlp"
        return "other"

    def find_target_modules(model):
        """(name, category, module) for every weight tensor worth sensitivity-testing.
        Skips MTP (zeroed separately), vision (irrelevant to score), tiny SSM
        control params, and norms."""
        skip_categories = {"mtp", "visual", "linear_attn_control", "full_attn_norm", "other"}
        targets = []
        for name, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue
            if not isinstance(module.weight, torch.nn.Parameter):
                continue
            if module.weight.dim() != 2:
                continue
            cat = categorize(name)
            if cat in skip_categories:
                continue
            targets.append((name, cat, module))
        return targets

    return categorize, find_target_modules


@app.cell
def _(
    build_calibration_text,
    calib_dataset,
    calib_seqs,
    device,
    mo,
    seq_len,
    tokenize_calibration,
    tokenizer,
):
    _texts = build_calibration_text(calib_dataset.value, calib_seqs.value)
    batches = tokenize_calibration(tokenizer, _texts, seq_len.value, device.value)
    mo.md(f"Built **{len(batches)}** calibration sequences from `{calib_dataset.value}`.")
    return (batches,)


@app.cell
def _(batches, calibration_loss, mo, model):
    baseline_loss = calibration_loss(model, batches)
    mo.md(f"**Baseline calibration loss (unquantized):** {baseline_loss:.4f}")
    return (baseline_loss,)


@app.cell
def _(find_target_modules, mo, model):
    targets = find_target_modules(model)
    mo.md(f"Found **{len(targets)}** target tensors to sensitivity-test.")
    return (targets,)


@app.cell
def _(mo):
    sweep_button = mo.ui.run_button(label="2. Run sensitivity sweep (slow — grab a coffee)")
    sweep_button
    return (sweep_button,)


@app.cell
def _(
    baseline_loss,
    batches,
    bits_list,
    calibration_loss,
    fake_quantized,
    group_size,
    json,
    mo,
    model,
    out_path,
    pd,
    sweep_button,
    targets,
    time,
):
    mo.stop(not sweep_button.value, mo.md("Click **Run sensitivity sweep** above to start."))

    _bits_to_test = [int(b) for b in bits_list.value]
    _records = []
    _t0 = time.time()

    with open(out_path.value, "w") as _f:
        _f.write(json.dumps({"baseline_loss": baseline_loss}) + "\n")
        for _name, _cat, _module in mo.status.progress_bar(targets, title="Sweeping tensors"):
            _numel = _module.weight.numel()
            for _bits in _bits_to_test:
                with fake_quantized(_module, _bits, group_size.value):
                    _loss = calibration_loss(model, batches)
                _delta = _loss - baseline_loss
                _record = {
                    "name": _name,
                    "category": _cat,
                    "numel": _numel,
                    "bits": _bits,
                    "loss": _loss,
                    "delta_loss": _delta,
                }
                _records.append(_record)
                _f.write(json.dumps(_record) + "\n")
                _f.flush()

    results_df = pd.DataFrame(_records)
    mo.md(
        f"Sweep complete in {time.time() - _t0:.0f}s. "
        f"{len(results_df)} measurements written to `{out_path.value}`."
    )
    return (results_df,)


@app.cell
def _(mo, results_df):
    mo.ui.table(results_df)
    return


@app.cell
def _(mo, results_df):
    _pivot = results_df.pivot_table(
        index="category", columns="bits", values="delta_loss", aggfunc="mean"
    )
    mo.vstack([mo.md("### Mean delta-loss by category and bit-width"), _pivot.round(4)])
    return


@app.cell
def _(mo, results_df):
    _d4 = results_df[results_df["bits"] == 4].sort_values("delta_loss", ascending=False)
    mo.vstack(
        [
            mo.md("### Most sensitive tensors at 4-bit (top 20 — need MORE bits)"),
            _d4[["name", "category", "numel", "delta_loss"]].head(20),
            mo.md("### Least sensitive tensors at 4-bit (bottom 20 — safe to push lower)"),
            _d4[["name", "category", "numel", "delta_loss"]].tail(20),
        ]
    )
    return


@app.cell
def _(mo, results_df):
    _bits_sorted = sorted(results_df["bits"].unique(), reverse=True)
    _lines = ["### Marginal loss per bit removed, by category\n"]
    for _cat, _group in results_df.groupby("category"):
        _cat_pivot = _group.groupby("bits")["delta_loss"].mean().reindex(_bits_sorted)
        _lines.append(f"\n**{_cat}**\n")
        _prev_bits, _prev_loss = None, None
        for _b, _l in _cat_pivot.items():
            if _prev_bits is not None and _l == _l and _prev_loss == _prev_loss:
                _slope = (_l - _prev_loss) / (_prev_bits - _b) if _prev_bits != _b else float("nan")
                _lines.append(
                    f"- {_prev_bits}\u2192{_b} bits: {_prev_loss:.4f} \u2192 {_l:.4f} "
                    f"(marginal loss/bit: {_slope:.4f})\n"
                )
            _prev_bits, _prev_loss = _b, _l
    mo.md("".join(_lines))
    return


@app.cell
def _(mo, out_path, results_df):
    _csv_path = out_path.value.rsplit(".", 1)[0] + "_summary.csv"
    results_df.to_csv(_csv_path, index=False)
    mo.md(f"Full results also saved to `{_csv_path}`.")
    return


if __name__ == "__main__":
    app.run()