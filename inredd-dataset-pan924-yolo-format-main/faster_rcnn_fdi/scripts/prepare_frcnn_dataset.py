"""Prepare Faster R-CNN COCO dataset from merged PAN924 annotations.

Input:
- annotations/dataset_final_v2.json

Outputs:
- faster_rcnn_fdi/data/annotations/train_coco.json
- faster_rcnn_fdi/data/annotations/val_coco.json
- faster_rcnn_fdi/data/annotations/test_coco.json
- faster_rcnn_fdi/data/annotations/prepare_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare Faster R-CNN COCO splits")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("annotations/dataset_final_v2.json"),
        help="Input clean annotation JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("faster_rcnn_fdi/data/annotations"),
        help="Output directory for COCO annotations",
    )
    parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="Keep images that end up with zero boxes in detection dataset",
    )
    return parser.parse_args()


def patient_id_from_filename(file_name: str) -> str:
    """Extract patient ID from file name."""
    return Path(file_name).stem.split("-")[0]


def assign_split(patient_id: str) -> str:
    """Deterministic split assignment by patient hash."""
    digest = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def create_categories() -> list[dict[str, Any]]:
    """Create fixed 32 FDI category list."""
    labels = [
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
    ]
    categories = []
    for idx, label in enumerate(labels, start=1):
        categories.append({"id": idx, "name": str(label), "supercategory": "tooth"})
    return categories


def prepare_coco_splits(data: dict[str, Any], keep_empty: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Convert clean JSON records into train/val/test COCO dictionaries."""
    categories = create_categories()
    label_to_category_id = {int(cat["name"]): int(cat["id"]) for cat in categories}

    split_coco = {
        "train": {"images": [], "annotations": [], "categories": categories},
        "val": {"images": [], "annotations": [], "categories": categories},
        "test": {"images": [], "annotations": [], "categories": categories},
    }

    split_image_ids = {"train": 1, "val": 1, "test": 1}
    split_ann_ids = {"train": 1, "val": 1, "test": 1}

    dropped_invalid_bbox = 0
    dropped_duplicate_same_fdi = 0
    dropped_unknown_label = 0
    dropped_empty_images = 0

    for image_item in data.get("images", []):
        file_name = str(image_item.get("file_name", "")).strip()
        if not file_name:
            continue

        patient_id = patient_id_from_filename(file_name)
        split = assign_split(patient_id)

        teeth = image_item.get("teeth", [])

        # Deduplicate by FDI label in same image, keep highest confidence.
        best_by_fdi: dict[int, dict[str, Any]] = {}
        for tooth in teeth:
            label = tooth.get("fdi_label")
            if label is None:
                dropped_unknown_label += 1
                continue

            label = int(label)
            if label not in label_to_category_id:
                dropped_unknown_label += 1
                continue

            bbox = tooth.get("bbox_xywh")
            if not isinstance(bbox, list) or len(bbox) != 4:
                dropped_invalid_bbox += 1
                continue

            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                dropped_invalid_bbox += 1
                continue

            if label in best_by_fdi:
                dropped_duplicate_same_fdi += 1
            else:
                best_by_fdi[label] = tooth

        teeth_filtered = list(best_by_fdi.values())

        if not teeth_filtered and not keep_empty:
            dropped_empty_images += 1
            continue

        image_id = split_image_ids[split]
        split_image_ids[split] += 1

        split_coco[split]["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(image_item.get("width", 0)),
                "height": int(image_item.get("height", 0)),
            }
        )

        for tooth in teeth_filtered:
            label = int(tooth["fdi_label"])
            cat_id = label_to_category_id[label]
            x, y, w, h = [float(v) for v in tooth["bbox_xywh"]]

            ann_id = split_ann_ids[split]
            split_ann_ids[split] += 1

            split_coco[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "tooth_id": label,
                    "disease_label": str(tooth.get("disease_label", "")),
                }
            )

    summary = {
        "keep_empty_images": keep_empty,
        "train_images": len(split_coco["train"]["images"]),
        "val_images": len(split_coco["val"]["images"]),
        "test_images": len(split_coco["test"]["images"]),
        "train_annotations": len(split_coco["train"]["annotations"]),
        "val_annotations": len(split_coco["val"]["annotations"]),
        "test_annotations": len(split_coco["test"]["annotations"]),
        "dropped_invalid_bbox": dropped_invalid_bbox,
        "dropped_duplicate_same_fdi": dropped_duplicate_same_fdi,
        "dropped_unknown_label": dropped_unknown_label,
        "dropped_empty_images": dropped_empty_images,
    }

    return split_coco["train"], split_coco["val"], split_coco["test"], summary


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    root = args.root.resolve()
    input_json = (root / args.input_json).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    train_coco, val_coco, test_coco, summary = prepare_coco_splits(
        data=data,
        keep_empty=bool(args.keep_empty_images),
    )

    (output_dir / "train_coco.json").write_text(json.dumps(train_coco, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "val_coco.json").write_text(json.dumps(val_coco, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "test_coco.json").write_text(json.dumps(test_coco, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Faster R-CNN COCO Build Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
