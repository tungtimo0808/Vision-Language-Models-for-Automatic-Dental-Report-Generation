# Colab / A100 notebooks (bf16 LoRA)

Five notebooks — one per model. They are all generated from one template by
`generate_notebooks.py`, so they are identical except for the model. Train on a Colab **A100**;
checkpoints are saved on **Google Drive**, so re-running the train cell **continues** instead of
restarting.

| Notebook | Model |
|---|---|
| `train_qwen.ipynb` | Qwen/Qwen2.5-VL-7B-Instruct |
| `train_internvl.ipynb` | OpenGVLab/InternVL3-8B |
| `train_llava.ipynb` | llava-hf/llava-onevision-qwen2-7b-ov-hf |
| `train_phi.ipynb` | microsoft/Phi-3.5-vision-instruct (set `ATTN_IMPL='eager'` if no flash-attn) |
| `train_paligemma.ipynb` | google/paligemma2-3b-pt-448 (gated — `notebook_login()`) |

> Edit the template in `generate_notebooks.py` and re-run it; do not hand-edit the `.ipynb` files
> (that keeps all five consistent).

## Hugging Face, not ModelScope
ms-swift defaults to **ModelScope** (very slow from outside China — we saw ~73 kB/s / 14h ETA on
Colab). The install cell sets **`USE_HF=1`** (before importing swift) so everything downloads from
**Hugging Face** instead, where `hf_transfer` makes it fast. If you ever see `modelscope.cn` in the
log, `USE_HF` wasn't set early enough — re-run the install cell.

## Base model: re-downloaded each session (fast, with a progress bar), not cached on Drive
The base model lives in the VM's default cache (wiped on disconnect), so it re-downloads on each fresh
session. **Step 6 downloads it in-notebook so you see a live progress bar** (confirming it's running,
not hung). With **`hf_transfer`** (installed in step 2, `HF_HUB_ENABLE_HF_TRANSFER=1` set there too)
that's only ~2–3 min for ~16GB — FASTER than reading it back from Drive FUSE, and needs no paid Drive
storage. (Caching the model on Drive was tried and dropped: Drive free is 15 GB, 5 models ≈ 60–70 GB,
and FUSE loads are slower than a fresh download.) Only the LoRA checkpoints live on Drive, which is
what makes training resumable.

## How to run (each notebook, top to bottom)
1. `Runtime -> Change runtime type -> A100 GPU`.
2. **Install** cell (sets `USE_HF=1` + installs hf_transfer for fast Hugging Face downloads; PaliGemma logs into HF).
3. **Mount Drive** — this is what makes training resumable.
4. **Get data** — put it on Drive in `Thesis/`. **Prefer the zip** `pan924_vlm.zip` (the cell unzips it
   locally in ~1–3 min; one big file copies far faster than 4620 small ones). An extracted
   `pan924_vlm/vlm_report_dataset/` folder also works (slower). Either way the cell lands the data on the
   local VM disk `/content/pan924`. (Data = `converted/qwen/*.jsonl`, `common/images/`, eval scripts —
   ~2.3 GB, no original panoramics.)
5. **Settings** cell (shared across all 5 notebooks).
6. **Download base model** — pulls the ~16GB model to the VM with a live progress bar (re-run after a
   disconnect; ~2–3 min).
7. **Train** — re-run after a disconnect; it resumes from the last Drive checkpoint
   under `Thesis/pan924_runs/<model>/` (model already downloaded in step 6).
8. **Pick best checkpoint** by val macro-F1.
9. **Final metrics** on test (full per-condition report).
10. **Compare** all 5 models.

## Settings vs the RTX 3090 path
Same LoRA / epochs / lr / effective-batch-16 / eval-every-100. Differences: **bf16 LoRA** (no
4-bit) and `MAX_PIXELS=1003520` (higher) — the A100 has the VRAM, so it's faster and sharper.

## Attention backend (why no flash-attn by default)
The install cell does **not** build `flash-attn` — on Colab it takes 15–30 min to compile and often
fails. ms-swift falls back to PyTorch **SDPA**, which is reliable and nearly as fast on an A100, so
`ATTN_IMPL` stays `None` (Phi uses `'eager'`). Want the extra ~10–20%? Uncomment the `flash-attn`
line in the install cell — nothing else changes.

## If you run out of memory (40GB A100)
The default is the safe floor: `PER_DEVICE_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=16` (effective 16). If
you still OOM, lower `MAX_PIXELS` (1003520 → 802816 → 602112) in the **Settings** cell and re-run the
train cell (it resumes). If memory is comfortable (or on an 80GB A100), set `PER_DEVICE_BATCH_SIZE=2`
/ `GRAD_ACCUM_STEPS=8` (still 16) to train ~2× faster.
