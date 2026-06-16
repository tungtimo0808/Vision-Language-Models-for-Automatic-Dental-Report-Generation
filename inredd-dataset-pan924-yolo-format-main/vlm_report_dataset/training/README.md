# Training

Fine-tune the 5 VLMs on the PAN924 dental report dataset, then measure how well each one does.

## Two ways to train (pick one per model)
| Folder | Hardware | Method | When to use |
|---|---|---|---|
| [`rtx3090/`](rtx3090/) | rented RTX 3090 (24GB) | 5 Python scripts, QLoRA 4-bit | a cheap rented GPU |
| [`colab_a100/`](colab_a100/) | Colab A100 (40GB) | 5 notebooks, bf16 LoRA | Google Colab |

Both paths use the **same settings** (LoRA r8/α32, 2 epochs, lr 1e-4, effective batch 16,
eval/save every 100, pick best checkpoint by macro-F1). The A100 path uses bf16 instead of 4-bit
and a higher resolution because it has the VRAM — everything else matches, so results stay
comparable across hardware **and** across models.

## Shared tools ([`tools/`](tools/))
- `validate_setup.py` — offline check (no GPU): data, images, scripts, notebooks all present.
- `swift_pred_to_eval.py` — turns `swift infer` output into the scorer's input format.
- `compare_models.py` — builds one comparison table across all 5 models once they're trained.

The metric engine is [`../scripts/eval_report.py`](../scripts/eval_report.py). It reports, for any
model's predictions: per-condition + overall **precision / recall / F1 / accuracy**, **FDI
detection** (does it find the right teeth?), `condition_acc_on_detected` (does it label the disease
right?), hallucination & miss rates, exact-report match, and ROUGE-L on the text.

## Typical flow (RTX 3090)
```bash
python vlm_report_dataset/training/tools/validate_setup.py     # check first (no GPU)
bash   vlm_report_dataset/training/rtx3090/setup_env.sh        # install on the rented box
python vlm_report_dataset/training/rtx3090/smoke_test.py       # 2-min pipeline test
python vlm_report_dataset/training/rtx3090/train_qwen.py       # trains AND evaluates, then writes results
```
`train_qwen.py` trains, picks the best checkpoint, evaluates on test, and writes
`results/qwen_results.txt`. Re-run the same command to resume. For Colab, open the matching
notebook in [`colab_a100/`](colab_a100/) and run top to bottom (results are shown inline).

## Folder map
```
training/
├── rtx3090/      5 train_<model>.py + shared_config.py + select_best_checkpoint.py + smoke_test.py + setup_env.sh
├── colab_a100/   5 train_<model>.ipynb + generate_notebooks.py
├── results/      <model>_results.txt + metrics_<model>_test.json  (written by the 3090 scripts)
└── tools/        validate_setup.py + swift_pred_to_eval.py + compare_models.py
```
