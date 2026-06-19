"""Prepare Faster R-CNN COCO dataset for DISEASE detection from PAN924 annotations.

Mirrors faster_rcnn_fdi/scripts/prepare_frcnn_dataset.py but the per-box class is
the disease label (12 clean classes) instead of the FDI tooth id.

Input:
- annotations/dataset_final_v2.json

Outputs:
- faster_rcnn_disease/data/annotations/train_coco.json
- faster_rcnn_disease/data/annotations/val_coco.json
- faster_rcnn_disease/data/annotations/test_coco.json
- faster_rcnn_disease/data/annotations/prepare_summary.json

Notes:
- Split is the SAME deterministic patient hash as the FDI pipeline, so the FDI and
  disease detectors share identical train/val/test images (fair comparison).
- One box per physical tooth (deduplicated by FDI), labelled with its disease class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

# 12 canonical disease classes (after remap). Order fixes the category_id mapping.
DISEASE_CLASSES: list[str] = [
    "H",     # healthy
    "R",     # restored (filling/crown)
    "Te",    # endodontically treated
    "CpuM",  # crown / mixed prosthesis
    "M3i",   # impacted tooth
    "M3f",   # developing tooth
    "Di",    # worn / attrition
    "C",     # caries
    "Rr",    # residual root
    "P",     # pontic
    "Im",    # implant
    "Dc",    # destroyed crown
]

# Raw -> canonical remap (same policy as the rest of the pipeline).
DISEASE_REMAP: dict[str, str] = {
    "RiM": "Te",
    "Ri": "Te",
    "TeM": "Te",
    "I": "M3i",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare Faster R-CNN COCO splits (disease)")
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
        default=Path("faster_rcnn_disease/data/annotations"),
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
    """Deterministic split assignment by patient hash (identical to FDI pipeline)."""
    digest = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def normalize_disease(raw_label: Any) -> str | None:
    """Map a raw disease label to one of the 12 canonical classes, or None if unknown."""
    if raw_label is None:
        return None
    label = str(raw_label).strip()
    if not label:
        return None
    label = DISEASE_REMAP.get(label, label)
    if label not in DISEASE_CLASSES:
        return None
    return label


def create_categories() -> list[dict[str, Any]]:
    """Create the fixed 12-class disease category list."""
    categories = []
    for idx, label in enumerate(DISEASE_CLASSES, start=1):
        categories.append({"id": idx, "name": label, "supercategory": "disease"})
    return categories


def prepare_coco_splits(
    data: dict[str, Any], keep_empty: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Convert clean JSON records into train/val/test COCO dictionaries (disease class)."""
    categories = create_categories()
    label_to_category_id = {str(cat["name"]): int(cat["id"]) for cat in categories}

    split_coco = {
        "train": {"images": [], "annotations": [], "categories": categories},
        "val": {"images": [], "annotations": [], "categories": categories},
        "test": {"images": [], "annotations": [], "categories": categories},
    }

    split_image_ids = {"train": 1, "val": 1, "test": 1}
    split_ann_ids = {"train": 1, "val": 1, "test": 1}

    dropped_invalid_bbox = 0
    dropped_duplicate_same_fdi = 0
    dropped_unknown_disease = 0
    dropped_empty_images = 0
    disease_counts: dict[str, int] = {label: 0 for label in DISEASE_CLASSES}

    for image_item in data.get("images", []):
        file_name = str(image_item.get("file_name", "")).strip()
        if not file_name:
            continue

        patient_id = patient_id_from_filename(file_name)
        split = assign_split(patient_id)

        teeth = image_item.get("teeth", [])

        # One box per physical tooth: deduplicate by FDI, keep first occurrence.
        best_by_fdi: dict[int, dict[str, Any]] = {}
        for tooth in teeth:
            fdi = tooth.get("fdi_label")
            if fdi is None:
                continue
            fdi = int(fdi)

            disease = normalize_disease(tooth.get("disease_label"))
            if disease is None:
                dropped_unknown_disease += 1
                continue

            bbox = tooth.get("bbox_xywh")
            if not isinstance(bbox, list) or len(bbox) != 4:
                dropped_invalid_bbox += 1
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                dropped_invalid_bbox += 1
                continue

            if fdi in best_by_fdi:
                dropped_duplicate_same_fdi += 1
            else:
                best_by_fdi[fdi] = tooth

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
            fdi = int(tooth["fdi_label"])
            disease = normalize_disease(tooth.get("disease_label"))
            if disease is None:
                continue
            cat_id = label_to_category_id[disease]
            x, y, w, h = [float(v) for v in tooth["bbox_xywh"]]

            ann_id = split_ann_ids[split]
            split_ann_ids[split] += 1

            disease_counts[disease] += 1

            split_coco[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "disease_label": disease,
                    "tooth_id": fdi,
                }
            )

    summary = {
        "task": "disease_detection",
        "num_disease_classes": len(DISEASE_CLASSES),
        "disease_classes": DISEASE_CLASSES,
        "disease_remap": DISEASE_REMAP,
        "keep_empty_images": keep_empty,
        "train_images": len(split_coco["train"]["images"]),
        "val_images": len(split_coco["val"]["images"]),
        "test_images": len(split_coco["test"]["images"]),
        "train_annotations": len(split_coco["train"]["annotations"]),
        "val_annotations": len(split_coco["val"]["annotations"]),
        "test_annotations": len(split_coco["test"]["annotations"]),
        "disease_counts_all_splits": disease_counts,
        "dropped_invalid_bbox": dropped_invalid_bbox,
        "dropped_duplicate_same_fdi": dropped_duplicate_same_fdi,
        "dropped_unknown_disease": dropped_unknown_disease,
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

    print("=== Faster R-CNN Disease COCO Build Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
