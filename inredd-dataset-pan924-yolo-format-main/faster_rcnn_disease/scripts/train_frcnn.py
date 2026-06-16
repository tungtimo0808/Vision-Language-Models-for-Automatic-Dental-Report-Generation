"""Train Faster R-CNN for 12-class disease detection with low-VRAM settings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from frcnn_common import build_faster_rcnn_model, load_yaml, resolve_device, save_json, set_seed
from frcnn_dataset import CocoDetectionDataset, collate_fn


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Train Faster R-CNN (PAN924)")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("faster_rcnn_disease/configs/train_config.yaml"),
        help="YAML config path",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device: cuda or cpu")
    parser.add_argument("--architecture", type=str, default=None, help="Override detector architecture")
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=None,
        help="Override trainable backbone layers",
    )
    parser.add_argument("--min-size", type=int, default=None, help="Override model transform min size")
    parser.add_argument("--max-size", type=int, default=None, help="Override model transform max size")
    parser.add_argument("--output-subdir", type=str, default=None, help="Override output sub-directory")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Override early stop patience")
    parser.add_argument("--early-stop-min-delta", type=float, default=None, help="Override early stop min delta")
    parser.add_argument("--dry-run", action="store_true", help="Run one train+val batch only")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI overrides to nested config dictionary."""
    train_cfg = config.setdefault("train", {})
    model_cfg = config.setdefault("model", {})
    data_cfg = config.setdefault("data", {})
    project_cfg = config.setdefault("project", {})
    early_cfg = config.setdefault("early_stopping", {})

    if args.device:
        train_cfg["device"] = args.device
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.architecture:
        model_cfg["architecture"] = str(args.architecture)
    if args.trainable_backbone_layers is not None:
        model_cfg["trainable_backbone_layers"] = int(args.trainable_backbone_layers)
    if args.min_size is not None:
        data_cfg["min_size"] = int(args.min_size)
    if args.max_size is not None:
        data_cfg["max_size"] = int(args.max_size)
    if args.output_subdir:
        project_cfg["output_dir"] = str(args.output_subdir)
    if args.early_stop_patience is not None:
        early_cfg["patience"] = int(args.early_stop_patience)
        early_cfg["enabled"] = True
    if args.early_stop_min_delta is not None:
        early_cfg["min_delta"] = float(args.early_stop_min_delta)
        early_cfg["enabled"] = True


def move_to_device(
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Move detection batch to target device."""
    images_out = [img.to(device, non_blocking=True) for img in images]
    targets_out = []
    for t in targets:
        t_out = {k: v.to(device, non_blocking=True) for k, v in t.items()}
        targets_out.append(t_out)
    return images_out, targets_out


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
) -> float:
    """Compute validation loss in training mode without gradient updates."""
    model.train()
    total = 0.0
    count = 0

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, start=1):
            try:
                images, targets = move_to_device(images, targets, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled and device.type == "cuda"):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                    continue
                raise
            total += float(loss.item())
            count += 1
            if max_batches > 0 and step >= max_batches:
                break

    return total / max(count, 1)


def box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for two XYXY boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_tp_fp_fn(
    pred_boxes: list[list[float]],
    pred_labels: list[int],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    gt_labels: list[int],
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Match predictions with GT by label and IoU, then return TP/FP/FN."""
    filtered = [
        (box, label, score)
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores)
        if score >= score_threshold
    ]
    filtered.sort(key=lambda item: item[2], reverse=True)

    used_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred_box, pred_label, _ in filtered:
        best_gt_index = -1
        best_iou = 0.0

        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in used_gt:
                continue
            if gt_label != pred_label:
                continue

            iou = box_iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = idx

        if best_gt_index >= 0 and best_iou >= iou_threshold:
            tp += 1
            used_gt.add(best_gt_index)
        else:
            fp += 1

    fn = len(gt_boxes) - len(used_gt)
    return tp, fp, fn


def evaluate_per_class_and_confusion(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_batches: int,
    num_classes: int,
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    """Compute per-class P/R/F1/support, macro/weighted F1 and a confusion matrix.

    `num_classes` counts foreground classes only (background excluded). Class ids
    run 1..num_classes; index 0 in the confusion matrix is the background row/column
    (missed GT in the last column, spurious predictions in the first row).
    """
    model.eval()

    tp_c = {c: 0 for c in range(1, num_classes + 1)}
    fp_c = {c: 0 for c in range(1, num_classes + 1)}
    fn_c = {c: 0 for c in range(1, num_classes + 1)}

    size = num_classes + 1  # +1 for background at index 0
    confusion = [[0 for _ in range(size)] for _ in range(size)]

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, start=1):
            try:
                images, targets = move_to_device(images, targets, device)
                outputs = model(images)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                    continue
                raise

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu().numpy().tolist()
                pred_labels = output["labels"].detach().cpu().numpy().astype(int).tolist()
                pred_scores = output["scores"].detach().cpu().numpy().astype(float).tolist()

                gt_boxes = target["boxes"].detach().cpu().numpy().tolist()
                gt_labels = target["labels"].detach().cpu().numpy().astype(int).tolist()

                # Per-class TP/FP/FN: match within each class separately.
                for c in range(1, num_classes + 1):
                    p_idx = [i for i, lbl in enumerate(pred_labels) if lbl == c]
                    g_idx = [i for i, lbl in enumerate(gt_labels) if lbl == c]
                    if not p_idx and not g_idx:
                        continue
                    tp, fp, fn = match_tp_fp_fn(
                        pred_boxes=[pred_boxes[i] for i in p_idx],
                        pred_labels=[pred_labels[i] for i in p_idx],
                        pred_scores=[pred_scores[i] for i in p_idx],
                        gt_boxes=[gt_boxes[i] for i in g_idx],
                        gt_labels=[gt_labels[i] for i in g_idx],
                        score_threshold=score_threshold,
                        iou_threshold=iou_threshold,
                    )
                    tp_c[c] += tp
                    fp_c[c] += fp
                    fn_c[c] += fn

                # Confusion matrix: class-agnostic greedy IoU matching by score.
                order = sorted(
                    [i for i, s in enumerate(pred_scores) if s >= score_threshold],
                    key=lambda i: pred_scores[i],
                    reverse=True,
                )
                used_gt: set[int] = set()
                for pi in order:
                    best_iou = 0.0
                    best_gi = -1
                    for gi, gbox in enumerate(gt_boxes):
                        if gi in used_gt:
                            continue
                        iou = box_iou_xyxy(pred_boxes[pi], gbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    p_lbl = int(pred_labels[pi])
                    if best_gi >= 0 and best_iou >= iou_threshold:
                        g_lbl = int(gt_labels[best_gi])
                        confusion[g_lbl][p_lbl] += 1
                        used_gt.add(best_gi)
                    else:
                        confusion[0][p_lbl] += 1  # predicted something where no GT (FP)
                for gi, g_lbl in enumerate(gt_labels):
                    if gi not in used_gt:
                        confusion[int(g_lbl)][0] += 1  # missed GT (FN)

            if max_batches > 0 and step >= max_batches:
                break

    per_class: dict[str, Any] = {}
    macro_f1_sum = 0.0
    macro_count = 0
    weighted_f1_sum = 0.0
    support_total = 0
    for c in range(1, num_classes + 1):
        tp, fp, fn = tp_c[c], fp_c[c], fn_c[c]
        support = tp + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        name = id_to_name.get(c, str(c))
        per_class[name] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "support": int(support),
        }
        if support > 0:
            macro_f1_sum += f1
            macro_count += 1
            weighted_f1_sum += f1 * support
            support_total += support

    macro_f1 = macro_f1_sum / max(macro_count, 1)
    weighted_f1 = weighted_f1_sum / max(support_total, 1)

    class_order = ["background"] + [id_to_name.get(c, str(c)) for c in range(1, num_classes + 1)]

    return {
        "per_class": per_class,
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "confusion_matrix": confusion,
        "confusion_labels": class_order,
    }


def evaluate_detection_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_batches: int,
) -> dict[str, float | int]:
    """Evaluate detection metrics on a dataset split."""
    model.eval()

    tp_total = 0
    fp_total = 0
    fn_total = 0
    processed_batches = 0

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, start=1):
            try:
                images, targets = move_to_device(images, targets, device)
                outputs = model(images)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                    continue
                raise

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu().numpy().tolist()
                pred_labels = output["labels"].detach().cpu().numpy().astype(int).tolist()
                pred_scores = output["scores"].detach().cpu().numpy().astype(float).tolist()

                gt_boxes = target["boxes"].detach().cpu().numpy().tolist()
                gt_labels = target["labels"].detach().cpu().numpy().astype(int).tolist()

                tp, fp, fn = match_tp_fp_fn(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                tp_total += tp
                fp_total += fp
                fn_total += fn

            processed_batches += 1
            if max_batches > 0 and step >= max_batches:
                break

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)

    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "processed_batches": int(processed_batches),
    }


def evaluate_coco_map(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    annotation_path: Path,
    max_batches: int,
) -> dict[str, Any]:
    """Evaluate COCO mAP metrics (AP@[.50:.95] and AP@.50)."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {
            "map_50_95": 0.0,
            "map_50": 0.0,
            "processed_images": 0,
            "available": False,
            "error": "pycocotools_not_installed",
        }

    model.eval()
    predictions: list[dict[str, Any]] = []
    image_ids: list[int] = []

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, start=1):
            try:
                images = [img.to(device, non_blocking=True) for img in images]
                outputs = model(images)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                    continue
                raise

            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].detach().cpu().item())
                image_ids.append(image_id)

                pred_boxes = output["boxes"].detach().cpu().numpy().tolist()
                pred_labels = output["labels"].detach().cpu().numpy().astype(int).tolist()
                pred_scores = output["scores"].detach().cpu().numpy().astype(float).tolist()

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0.0 or h <= 0.0:
                        continue
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, w, h],
                            "score": float(score),
                        }
                    )

            if max_batches > 0 and step >= max_batches:
                break

    unique_img_ids = sorted(set(image_ids))
    if not unique_img_ids:
        return {
            "map_50_95": 0.0,
            "map_50": 0.0,
            "processed_images": 0,
            "available": True,
        }

    if not predictions:
        return {
            "map_50_95": 0.0,
            "map_50": 0.0,
            "processed_images": len(unique_img_ids),
            "available": True,
        }

    coco_gt = COCO(str(annotation_path))
    # Some exported COCO files omit optional metadata keys used by loadRes.
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = unique_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map_50_95": round(float(coco_eval.stats[0]), 6),
        "map_50": round(float(coco_eval.stats[1]), 6),
        "processed_images": len(unique_img_ids),
        "available": True,
    }


def main() -> None:
    """Training entrypoint."""
    args = parse_args()

    root = args.root.resolve()
    config = load_yaml((root / args.config).resolve())
    apply_cli_overrides(config, args)

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    output_dir = (root / config.get("project", {}).get("output_dir", "faster_rcnn_disease/outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    train_ann = (root / data_cfg.get("train_annotations")).resolve()
    val_ann = (root / data_cfg.get("val_annotations")).resolve()
    test_ann = (root / data_cfg.get("test_annotations")).resolve()
    images_root = (root / data_cfg.get("images_root")).resolve()

    train_ds = CocoDetectionDataset(images_root=images_root, annotation_path=train_ann)
    val_ds = CocoDetectionDataset(images_root=images_root, annotation_path=val_ann)
    test_ds = CocoDetectionDataset(images_root=images_root, annotation_path=test_ann)

    id_to_name = {int(c["id"]): str(c["name"]) for c in test_ds.categories}
    num_fg_classes = len(test_ds.categories)

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = resolve_device(str(train_cfg.get("device", "cuda")))
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda | GPU: {gpu_name}")
    else:
        print("Using device: cpu")

    model = build_faster_rcnn_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_val_loss = float("inf")
    best_macro_f1 = float("-inf")
    best_epoch = 0
    no_improve_epochs = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_macro_f1 = float(ckpt.get("best_macro_f1", best_macro_f1))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        no_improve_epochs = int(ckpt.get("no_improve_epochs", no_improve_epochs))

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 30))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    clip_grad_norm = float(train_cfg.get("clip_grad_norm", 5.0))
    val_max_batches = int(train_cfg.get("val_max_batches", 120))
    save_every = int(train_cfg.get("save_every", 1))

    eval_cfg = config.get("eval", {})
    eval_iou_threshold = float(eval_cfg.get("iou_threshold", 0.5))
    eval_score_threshold = float(
        eval_cfg.get("score_threshold", config.get("inference", {}).get("score_threshold", 0.3))
    )
    test_max_batches = int(eval_cfg.get("test_max_batches", 0))

    early_cfg = config.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_patience = int(early_cfg.get("patience", 8))
    early_min_delta = float(early_cfg.get("min_delta", 0.0005))
    monitor = str(early_cfg.get("monitor", "val_loss")).strip()
    if monitor not in ("val_loss", "macro_f1"):
        print(f"[WARN] unknown early_stopping.monitor='{monitor}', falling back to val_loss")
        monitor = "val_loss"

    metrics_history = []
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, (images, targets) in enumerate(pbar, start=1):
            try:
                images, targets = move_to_device(images, targets, device)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                    loss_for_backward = loss / grad_accum_steps

                scaler.scale(loss_for_backward).backward()
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and device.type == "cuda":
                    print("[WARN] CUDA OOM on one batch. Skipping batch. Consider lowering min_size/max_size further.")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            if step % grad_accum_steps == 0:
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            batch_count += 1
            pbar.set_postfix(loss=f"{running_loss / max(batch_count,1):.4f}")

            if args.dry_run and step >= 1:
                break

        train_loss = running_loss / max(batch_count, 1)
        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=scaler.is_enabled(),
            max_batches=1 if args.dry_run else val_max_batches,
        )
        val_det = evaluate_detection_metrics(
            model=model,
            loader=val_loader,
            device=device,
            score_threshold=eval_score_threshold,
            iou_threshold=eval_iou_threshold,
            max_batches=1 if args.dry_run else val_max_batches,
        )

        # Macro-F1 on val is only needed when it drives early stopping (extra pass over val).
        val_macro_f1 = None
        if monitor == "macro_f1":
            val_pc = evaluate_per_class_and_confusion(
                model=model,
                loader=val_loader,
                device=device,
                score_threshold=eval_score_threshold,
                iou_threshold=eval_iou_threshold,
                max_batches=1 if args.dry_run else val_max_batches,
                num_classes=num_fg_classes,
                id_to_name=id_to_name,
            )
            val_macro_f1 = float(val_pc["macro_f1"])

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_precision": float(val_det["precision"]),
            "val_recall": float(val_det["recall"]),
            "val_f1": float(val_det["f1"]),
            "val_macro_f1": round(val_macro_f1, 6) if val_macro_f1 is not None else None,
            "val_tp": int(val_det["tp"]),
            "val_fp": int(val_det["fp"]),
            "val_fn": int(val_det["fn"]),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metrics_history.append(epoch_info)
        print(epoch_info)

        save_json(
            results_dir / f"epoch_{epoch:03d}.json",
            {
                "epoch": int(epoch),
                "loss": {
                    "train": round(train_loss, 6),
                    "val": round(val_loss, 6),
                },
                "val_detection": val_det,
                "eval_config": {
                    "score_threshold": eval_score_threshold,
                    "iou_threshold": eval_iou_threshold,
                },
                "time": epoch_info["time"],
            },
        )

        if (epoch % save_every) == 0 or args.dry_run:
            last_ckpt = output_dir / "last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_val_loss": best_val_loss,
                    "best_macro_f1": best_macro_f1,
                    "best_epoch": best_epoch,
                    "no_improve_epochs": no_improve_epochs,
                    "config": config,
                },
                last_ckpt,
            )

        # Pick the "best" epoch by the configured monitor.
        if monitor == "macro_f1":
            improved = val_macro_f1 is not None and val_macro_f1 > (best_macro_f1 + early_min_delta)
        else:
            improved = val_loss < (best_val_loss - early_min_delta)

        if improved:
            best_val_loss = val_loss
            if val_macro_f1 is not None:
                best_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_epochs = 0
            best_ckpt = output_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_val_loss": best_val_loss,
                    "best_macro_f1": best_macro_f1,
                    "best_epoch": best_epoch,
                    "no_improve_epochs": no_improve_epochs,
                    "config": config,
                },
                best_ckpt,
            )
        else:
            no_improve_epochs += 1

        save_json(output_dir / "metrics_history.json", {"history": metrics_history})

        if args.dry_run:
            break

        if early_enabled and no_improve_epochs >= early_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"No improvement for {no_improve_epochs} epochs (patience={early_patience})."
            )
            break

    # Final test evaluation with best checkpoint.
    best_ckpt = output_dir / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        best_epoch = int(state.get("best_epoch", best_epoch))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))

    best_val_loss_num = float(best_val_loss)
    best_val_loss_out = round(best_val_loss_num, 6) if best_val_loss_num != float("inf") else None
    best_val_loss = float(state.get("best_val_loss", best_val_loss))

    test_loss = evaluate_loss(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=scaler.is_enabled(),
        max_batches=1 if args.dry_run else test_max_batches,
    )
    test_det = evaluate_detection_metrics(
        model=model,
        loader=test_loader,
        device=device,
        score_threshold=eval_score_threshold,
        iou_threshold=eval_iou_threshold,
        max_batches=1 if args.dry_run else test_max_batches,
    )
    test_map = evaluate_coco_map(
        model=model,
        loader=test_loader,
        device=device,
        annotation_path=test_ann,
        max_batches=1 if args.dry_run else test_max_batches,
    )

    test_per_class = evaluate_per_class_and_confusion(
        model=model,
        loader=test_loader,
        device=device,
        score_threshold=eval_score_threshold,
        iou_threshold=eval_iou_threshold,
        max_batches=1 if args.dry_run else test_max_batches,
        num_classes=num_fg_classes,
        id_to_name=id_to_name,
    )

    final_results = {
        "best_epoch": int(best_epoch),
        "best_val_loss": best_val_loss_out,
        "test_loss": round(float(test_loss), 6),
        "test_detection": test_det,
        "test_map": test_map,
        "test_macro_f1": test_per_class["macro_f1"],
        "test_weighted_f1": test_per_class["weighted_f1"],
        "test_per_class": test_per_class["per_class"],
        "confusion_matrix": test_per_class["confusion_matrix"],
        "confusion_labels": test_per_class["confusion_labels"],
        "eval_config": {
            "score_threshold": eval_score_threshold,
            "iou_threshold": eval_iou_threshold,
            "test_max_batches": int(test_max_batches),
        },
        "early_stopping": {
            "enabled": early_enabled,
            "patience": early_patience,
            "min_delta": early_min_delta,
            "monitor": monitor,
            "best_val_macro_f1": round(float(best_macro_f1), 6) if best_macro_f1 != float("-inf") else None,
        },
    }
    save_json(output_dir / "test_results.json", final_results)
    save_json(output_dir / "results_summary.json", {"history": metrics_history, "final": final_results})

    print(f"Training done. Best val loss: {best_val_loss:.6f}")
    print(f"Test detection (global): {test_det}")
    print(f"Test macro-F1: {test_per_class['macro_f1']} | weighted-F1: {test_per_class['weighted_f1']}")
    print(f"Test mAP: {test_map}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
