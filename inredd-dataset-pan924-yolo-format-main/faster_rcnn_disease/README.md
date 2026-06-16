# Faster R-CNN Workspace (12-Class Disease Detection)

Baseline detector that predicts the **disease condition** per tooth on panoramic
X-rays. It mirrors `faster_rcnn_fdi/` exactly — same pipeline, same 3 RTX-3050-friendly
backbones, same fairness policy — but each box class is a **disease code** instead of
an FDI tooth id. This is the disease-side baseline to compare against the VLM.

## Goals

- Detect teeth on panoramic X-ray images.
- Predict one of **12 disease classes** per box (+ background).
- Keep split deterministic by patient ID, identical to the FDI pipeline (no leakage,
  same patients land in the same split for a fair FDI-vs-disease comparison).

## Disease classes (12)

`H, R, Te, CpuM, M3i, M3f, Di, C, Rr, P, Im, Dc`

Raw-label remap applied during prep: `RiM/Ri/TeM -> Te`, `I -> M3i`.

> Note: classes are heavily imbalanced (`H` and `R` dominate; `Dc`, `Im`, `P` are rare).
> Read `prepare_summary.json` → `disease_counts_all_splits` and prefer per-class /
> macro metrics when interpreting results.

## Structure

- `configs/train_config.yaml`: training/inference config (`num_classes: 13` = 12 + background)
- `scripts/prepare_disease_dataset.py`: converts clean JSON into COCO splits (disease class)
- `scripts/train_frcnn.py`, `scripts/benchmark_frcnn_models.py`, `scripts/infer_frcnn.py`,
  `scripts/frcnn_common.py`, `scripts/frcnn_dataset.py`: same engine as the FDI workspace
- `data/annotations/{train,val,test}_coco.json`: generated COCO annotations

## Build Disease Dataset

Run from repository root:

```powershell
python faster_rcnn_disease/scripts/prepare_disease_dataset.py --root .
```

## Train (RTX 3050 4GB profile)

```powershell
python faster_rcnn_disease/scripts/train_frcnn.py --root . --config faster_rcnn_disease/configs/train_config.yaml
```

Quick smoke test (1 train+val batch):

```powershell
python faster_rcnn_disease/scripts/train_frcnn.py --root . --dry-run --device cpu
```

## Benchmark 3 Models x 3 Fine-Tuning Depths

Same 3 backbones chosen for RTX 3050 4GB as the FDI benchmark:
`mobilenet_v3_large_320_fpn`, `mobilenet_v3_large_fpn`, `resnet50_fpn`
× depths `shallow (1)`, `deep (3)`, `full (5)` = 9 runs.

```powershell
python faster_rcnn_disease/scripts/benchmark_frcnn_models.py --root . --config faster_rcnn_disease/configs/train_config.yaml --device cuda
```

Fairness policy: all runs share the same base config (`train`, `data`, `eval`) and input
resolution; only `architecture` and `trainable_backbone_layers` change per run.

### Stop & resume (checkpointing)

You can stop at any time (Ctrl+C / shutdown) and just re-run the same command — it continues:

- **Within a run**: `last.pth` is saved every epoch; training resumes from the last
  completed epoch via `--resume` (the benchmark passes this automatically).
- **Across the 9 runs**: a run whose `test_results.json` already exists is detected as
  completed and **skipped**. Summary files are rewritten after every run, so a partial
  benchmark is always valid. Use `--force` to re-run completed experiments.

```powershell
# stop anytime, then run the SAME line again to continue
python faster_rcnn_disease/scripts/benchmark_frcnn_models.py --root . --config faster_rcnn_disease/configs/train_config.yaml --device cuda
```

### Metrics reported

Per run (`<run>/test_results.json`) and in the benchmark table:

- Global precision / recall / **F1**, **mAP@0.5**, **mAP@0.5:0.95**
- **macro-F1** and **weighted-F1** (important under heavy class imbalance)
- **per-class** precision / recall / F1 / support (12 disease classes)
- **confusion matrix** (which diseases get confused with which)

Benchmark outputs:

- `faster_rcnn_disease/outputs/benchmark/benchmark_results.json`
- `faster_rcnn_disease/outputs/benchmark/benchmark_results.csv`
- `faster_rcnn_disease/outputs/benchmark/benchmark_table.md`

## Plots

After the benchmark, render the key figures (needs `matplotlib`):

```powershell
python faster_rcnn_disease/scripts/plot_benchmark.py --root . --benchmark-dir faster_rcnn_disease/outputs/benchmark
```

Outputs to `faster_rcnn_disease/outputs/benchmark/plots/`:

- `model_comparison.png` — F1 / macro-F1 / mAP@0.5 across all 9 runs
- `per_class_f1_<run>.png` — per-class F1 (+support) for the best run
- `confusion_<run>.png` — row-normalized confusion matrix for the best run
- `training_curves_<run>.png` — loss + val F1 over epochs for the best run

## Inference

```powershell
python faster_rcnn_disease/scripts/infer_frcnn.py --root . --image images/1000-F-19.jpg
```

Each prediction row contains `disease_label`, `category_id`, `score`, `bbox_xyxy`.

## Relationship to the FDI workspace

| | `faster_rcnn_fdi/` | `faster_rcnn_disease/` |
|---|---|---|
| Box class | FDI tooth id (32) | disease code (12) |
| `num_classes` | 33 | 13 |
| Label source | `fdi_label` | `disease_label` (remapped) |
| Split | patient-hash | identical patient-hash |
| Backbones | 3 (RTX 3050) | same 3 |

Everything else (training loop, metrics, fairness policy) is identical.
