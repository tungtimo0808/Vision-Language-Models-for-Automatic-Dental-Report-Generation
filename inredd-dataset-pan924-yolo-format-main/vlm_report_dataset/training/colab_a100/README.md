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

## How to run (each notebook, top to bottom)
1. `Runtime -> Change runtime type -> A100 GPU`.
2. **Install** cell (PaliGemma also logs into Hugging Face).
3. **Mount Drive** — this is what makes training resumable.
4. **Get data** — zip `vlm_report_dataset/` (with images), upload to Drive, set `DATA_ZIP`.
5. **Settings** cell (shared across all 5 notebooks).
6. **Train** — re-run after a disconnect; it resumes from the last Drive checkpoint.
7. **Pick best checkpoint** by val macro-F1.
8. **Final metrics** on test (full per-condition report).
9. **Compare** all 5 models.

## Settings vs the RTX 3090 path
Same LoRA / epochs / lr / effective-batch-16 / eval-every-100. Differences: **bf16 LoRA** (no
4-bit) and `MAX_PIXELS=1003520` (higher) — the A100 has the VRAM, so it's faster and sharper.

## If you run out of memory (40GB A100)
In the **Settings** cell lower `MAX_PIXELS` (1003520 → 802816 → 602112), then
`PER_DEVICE_BATCH_SIZE` 2 → 1, and re-run the train cell (it resumes). On an 80GB A100 you can
raise `PER_DEVICE_BATCH_SIZE` to 4 to go faster.
