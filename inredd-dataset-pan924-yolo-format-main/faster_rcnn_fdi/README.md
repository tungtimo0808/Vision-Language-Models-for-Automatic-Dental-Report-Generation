# Faster R-CNN Workspace (32-Tooth FDI Detection)

This folder contains training assets for tooth localization + **FDI numbering** using
Faster R-CNN. The parallel workspace `faster_rcnn_disease/` runs the identical pipeline
but classifies the **disease condition** per tooth instead of the FDI id.

## Goals

- Detect teeth on panoramic X-ray images.
- Predict one of 32 FDI tooth IDs per box.
- Keep split deterministic by patient ID to avoid data leakage.

## Structure

- `configs/train_config.yaml`: baseline training/inference config
- `scripts/prepare_frcnn_dataset.py`: converts clean JSON into COCO splits
- `data/annotations/train_coco.json`: generated COCO train annotations
- `data/annotations/val_coco.json`: generated COCO val annotations
- `data/annotations/test_coco.json`: generated COCO test annotations

## Build Detection Dataset

Run from repository root:

```powershell
python faster_rcnn_fdi/scripts/prepare_frcnn_dataset.py --root .
```

## Train (RTX 3050 4GB profile)

The default config is tuned for low VRAM:

- backbone: `fasterrcnn_mobilenet_v3_large_320_fpn`
- batch size: `1`
- mixed precision: enabled
- workers: `0` (Windows-friendly)

Run training:

```powershell
python faster_rcnn_fdi/scripts/train_frcnn.py --root . --config faster_rcnn_fdi/configs/train_config.yaml
```

Resume from checkpoint (no restart from scratch):

```powershell
python faster_rcnn_fdi/scripts/train_frcnn.py --root . --config faster_rcnn_fdi/configs/train_config.yaml --resume faster_rcnn_fdi/outputs/last.pth
```

Quick smoke test (1 train+val batch):

```powershell
python faster_rcnn_fdi/scripts/train_frcnn.py --root . --dry-run
```

CPU smoke test (debug only):

```powershell
python faster_rcnn_fdi/scripts/train_frcnn.py --root . --dry-run --device cpu
```

Early stopping is enabled from config:

- `early_stopping.enabled`
- `early_stopping.patience`
- `early_stopping.min_delta`

## Benchmark 3 Models x 2 Fine-Tuning Depths

Run full benchmark:

```powershell
python faster_rcnn_fdi/scripts/benchmark_frcnn_models.py --root . --config faster_rcnn_fdi/configs/train_config.yaml --device cuda
```

Fairness policy used by benchmark:

- All runs share the same base config (`train`, `data`, `eval`).
- Shared input resolution (`min_size`, `max_size`) is read from the base config.
- Only 2 factors change per run: `architecture` and `trainable_backbone_layers`.

Recommendation rule for RTX3050:

- default: `f1_then_runtime` (higher F1 first, then lower runtime)
- optional: `f1_only`

Example:

```powershell
python faster_rcnn_fdi/scripts/benchmark_frcnn_models.py --root . --config faster_rcnn_fdi/configs/train_config.yaml --device cuda --recommend-by f1_then_runtime
```

Quick benchmark smoke test:

```powershell
python faster_rcnn_fdi/scripts/benchmark_frcnn_models.py --root . --config faster_rcnn_fdi/configs/train_config.yaml --device cpu --dry-run
```

Benchmark outputs:

- `faster_rcnn_fdi/outputs/benchmark/benchmark_results.json`
- `faster_rcnn_fdi/outputs/benchmark/benchmark_results.csv` (includes test_precision, test_recall, test_f1, test_map_50, test_map_50_95)
- `faster_rcnn_fdi/outputs/benchmark/benchmark_table.md`

## Inference

```powershell
python faster_rcnn_fdi/scripts/infer_frcnn.py --root . --image images/1000-F-19.jpg
```

Outputs:

- `faster_rcnn_fdi/outputs/prediction.json`
- `faster_rcnn_fdi/outputs/prediction_vis.jpg`

## Training JSON Artifacts

After training, the pipeline saves metrics/results to JSON:

- `faster_rcnn_fdi/outputs/metrics_history.json` (epoch train/val losses + val P/R/F1)
- `faster_rcnn_fdi/outputs/results/epoch_XXX.json` (detailed per-epoch metrics)
- `faster_rcnn_fdi/outputs/test_results.json` (final test metrics: precision/recall/F1 + mAP@0.5 and mAP@0.5:0.95)
- `faster_rcnn_fdi/outputs/results_summary.json` (history + final summary)
- `faster_rcnn_fdi/outputs/best.pth`, `faster_rcnn_fdi/outputs/last.pth` (checkpoints)

## Notes

- The generator keeps only boxes with positive width/height.
- Duplicate same-FDI labels in one image are deduplicated by highest confidence.
- Confidence threshold is configurable (`--min-confidence`, default `0.7`).
- Crop sync values in config should match `image_processor.py` defaults (`1.8`, `1.2`).
