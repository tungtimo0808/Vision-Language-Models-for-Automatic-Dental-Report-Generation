# RTX 3090 training scripts (QLoRA 4-bit)

Five plain Python scripts — one per model. They all share the settings in `shared_config.py`,
so the comparison between models is fair. Each `train_<model>.py` only sets the 4 things that are
specific to that model (name, output folder, MAX_PIXELS yes/no, attention).

| Script | Model |
|---|---|
| `train_qwen.py` | Qwen/Qwen2.5-VL-7B-Instruct |
| `train_internvl.py` | OpenGVLab/InternVL3-8B |
| `train_llava.py` | llava-hf/llava-onevision-qwen2-7b-ov-hf |
| `train_phi.py` | microsoft/Phi-3.5-vision-instruct |
| `train_paligemma.py` | google/paligemma2-3b-pt-448 (gated — `huggingface-cli login` first) |

## Files
- `shared_config.py` — the shared settings + `train_model()` + `evaluate_and_save()` + auto-resume helper.
- `train_<model>.py` — one entry point per model. **Trains AND evaluates** in one run.
- `select_best_checkpoint.py` — re-run the evaluation only (no retraining), if you ever need to.
- `smoke_test.py` — tiny 3-step run to prove the pipeline works before the real one.
- `setup_env.sh` — install ms-swift + all dependencies on a fresh box.

## How to run (on the rented 3090 box)
```bash
# 0) put the repo here, with vlm_report_dataset/ and its 4620 images
bash   vlm_report_dataset/training/rtx3090/setup_env.sh
python vlm_report_dataset/training/rtx3090/smoke_test.py        # ~2 min, run twice to test resume
python vlm_report_dataset/training/rtx3090/train_qwen.py        # trains, then evaluates
```
Running `train_qwen.py` does everything: it trains, picks the best checkpoint by validation
macro-F1, evaluates it on the test set, and writes the report to
`../results/qwen_results.txt`. Just re-run the same command if the box is killed — it resumes.

## Results
After a run, look in [`../results/`](../results/):
- `<model>_results.txt` — best checkpoint + every checkpoint's val macro-F1 + full test metrics.
- `metrics_<model>_test.json` — the same numbers as JSON.

## Resume
Every script auto-resumes: if `outputs/<model>/checkpoint-N` already exists, training continues
from it. So if the box is killed, just run the same command again.

## If you run out of memory (CUDA OOM)
Edit `shared_config.py` and lower in this order, then re-run (it resumes):
`MAX_PIXELS` 802816 → 602112 → 401408, then `MAX_LENGTH` 4096 → 3072, then `LORA_RANK` 8 → 4.
