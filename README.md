# CS6013 sandbox

One loop, repeatable every week:

```
LOCAL          make runs/<RUN_ID>/  →  git push
KAGGLE (CPU)   git pull → compress.py → SIZE GATE → decompress.py
               → verify+patch → upload fp16 to HF (repo ROOT) → git push results
MOLAB (GPU)    git pull → download fp16 → vllm serve → eval → git push results
```

Kaggle runs on a **CPU session**: compress/decompress stream safetensors and
never build the model, so they need no GPU and don't touch your 30 h/week GPU
quota. Molab is the only GPU step.

---

## Files

| File | Where |
|---|---|
| `kaggle_sandbox.ipynb` | import into Kaggle (File → Import Notebook) |
| `molab_eval.py` | upload to molab |
| `runs/<RUN_ID>/` | one folder per experiment — you create these |
| `results/<RUN_ID>/` | written by the notebooks, committed back |

---

## Repo layout

```
cs6013-sandbox/                 <- private GitHub repo
├── kaggle_sandbox.ipynb
├── molab_eval.py
├── .gitignore
├── runs/
│   ├── w01_c40_int4/
│   │   ├── compress.py         REQUIRED
│   │   ├── decompress.py       REQUIRED
│   │   ├── pyproject.toml      optional (pip install -e)
│   │   ├── run.yaml            optional
│   │   ├── compression/
│   │   └── decompression/
│   └── w02_c20_mixed/
└── results/
    └── w01_c40_int4/
        ├── compression.json
        ├── eval_w01_c40_int4_gsm8k.json
        └── summary_w01_c40_int4_gsm8k.json
```

`.gitignore` — never commit weights:

```
*.safetensors
*.pt
*.bin
work/
__pycache__/
```

---

## The run-folder contract

A run folder is valid if these two commands work with no manual edits:

```bash
python compress.py   --model_name <name> --checkpoint_path <base>       --output_path <out>
python decompress.py --model_name <name> --checkpoint_path <compressed> --output_path <out>
```

That is exactly the spec's required interface, so **a run folder is a
submission folder** — copy it to
`CS6013/<roll>/Week<NN>/Compression_<target>/Submission<NN>/` unchanged.

### `run.yaml` (optional)

```yaml
target: 40                                     # percent — drives the size gate
compress_args:   ["--profile", "int4"]
decompress_args: ["--out-dtype", "float16"]
notes: "int4 g128, math calibration"
```

Overrides the notebook defaults, so one notebook config runs any experiment.

---

## Starting a new experiment

```bash
cp -r runs/w01_c40_int4 runs/w02_c20_mixed
# edit runs/w02_c20_mixed/run.yaml and the compression code
git add runs/w02_c20_mixed && git commit -m "w02: mixed20" && git push
```

Then set `RUN_ID = "w02_c20_mixed"` in both notebooks.

---

## Setup (once)

**GitHub** — create a **private** repo `cs6013-sandbox`. Make a fine-grained PAT
with `Contents: read and write` scoped to it.

> Keep this repo private and unshared. It is not the submission repo; TAs go on
> `CS6013` only. Sharing it is an honor-code violation.

**Kaggle** — Add-ons → Secrets: `GITHUB_PAT`, `HF_TOKEN` (write). Attach the
Qwen3.5-4B base as a **Model**. Accelerator **None (CPU)**, Internet **On**.

**HuggingFace** — a write token. Two repo families, kept separate:

| Repo | Contents | Visibility |
|---|---|---|
| `<user>/qwen35-<RUN_ID>-fp16` | restored fp16, for molab | private |
| `<user>/<Enroll>-Week<NN>-Compression<T>-Submission<NN>` | compressed, graded | **public** |

Never put the 9.3 GB fp16 model in a submission repo — it breaks the size ratio
and the spec.

---

## Running

### Kaggle

Import `kaggle_sandbox.ipynb`, edit cell 0, Run All. It will:

1. pull the sandbox repo, validate the run folder
2. auto-discover the base model under `/kaggle/input`, check disk
3. `compress.py` → **measured** size ratio vs the 10/20/40 target
4. `decompress.py`
5. verify + patch the restored checkpoint
6. upload to HF **at the repo root**
7. commit `results/<RUN_ID>/compression.json` and push

`UPLOAD_SUBMISSION` is off by default. When you turn it on, the notebook
**refuses** to upload if the size gate failed or stray files are present.

### Molab

Upload `molab_eval.py`, attach the GPU, set `RUN_ID` + `HF_EVAL_REPO`, Run All.

Order that saves time: `smoke`/5 → `smoke`/10 → `gsm8k`/50 → `math500`/50.

**Always eval the base model first at identical settings** (`HF_EVAL_REPO =
"Qwen/Qwen3.5-4B"`, `DTYPE = "auto"`) and report every compressed result as a
delta from it.

---

## Issues already handled

Everything below is fixed in the templates — listed so you recognize them if
they resurface.

| Issue | Where it bit | Handling |
|---|---|---|
| `Could not find nvcc` | molab has GPU drivers but no CUDA toolkit; FlashInfer JIT fails | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Server dies on cell interrupt | subprocess shares the process group | `start_new_session=True` |
| No startup visibility | polling and log-reading were separate | log tailed inside the poll loop |
| `Invalid repository ID` | vLLM can't read a repo **subdirectory** | upload at repo root; molab downloads locally and finds a nested root if needed |
| Gated dataset (GPQA) | needs accepted terms + token | HF login cell |
| Multi-config dataset | `load_dataset` needs the config name | `dataset_config` threaded through |
| **fp16 → bf16 silent upcast** | config says `bfloat16`, weights are fp16; `--dtype auto` upcasts (10 mantissa bits → 7) | Kaggle patches `config.json` to `float16`; molab pins `DTYPE="float16"` |
| **Missing `chat_template.jinja`** | `enable_thinking` silently does nothing, no `<think>`, **no error** | checked on both sides; Kaggle copies it from base |
| Report files inside the checkpoint | inflate the ratio, violate the spec | reports written outside; cleanliness check |
| `max_new_tokens ≥ max_model_len` | leaves no prompt budget | 8000/16000 defaults |

Two of these — the dtype upcast and the missing chat template — **fail silently
and corrupt your numbers without erroring**. That's why both notebooks verify
rather than assume.

---

## Still open

1. **Is the hidden math set multiple-choice or free-form?** The TA harness is
   hardcoded to A–D (`_VALID_CHOICES`), yet the functions are named
   `build_math_prompt`/`run_math_eval`. Both modes are implemented
   (`GRADING_MODE`), but which proxy to trust depends on the answer. **Ask a TA.**
2. **Does the TA's `--dtype auto` upcast submitted fp16 checkpoints to bf16?**
   If so it affects everyone's scores, and patching `config.json` inside
   `decompress.py` is the fix. Worth raising.
3. **Size denominator: weights-only or all files?** Kaggle reports both.
