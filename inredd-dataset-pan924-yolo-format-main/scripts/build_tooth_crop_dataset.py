"""Build a structured tooth-crop dataset from PAN924 clean annotations.

This script creates a new dataset optimized for instruction tuning workflows:
- Macro labels template: one row per panoramic image
- Micro labels: one row per tooth crop with FDI and disease metadata
- Deterministic train/val/test split by patient ID to avoid leakage
- Quality-control artifacts for missing images and invalid boxes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from image_processor import get_contextual_square_crop_from_array


DISEASE_LABEL_REMAP = {
    "RiM": "Te",
    "Ri": "Te",
    "TeM": "Te",
    "I": "M3i",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build PAN924 tooth crop dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Dataset root folder containing images/ and annotations/",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("annotations/dataset_final_clean.json"),
        help="Input clean JSON annotations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_dataset/pan924_instruction_v1"),
        help="Output folder for generated dataset",
    )
    parser.add_argument(
        "--expand-width",
        type=float,
        default=1.8,
        help="Width expansion factor for contextual square crop",
    )
    parser.add_argument(
        "--expand-height",
        type=float,
        default=1.2,
        help="Height expansion factor for contextual square crop",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=32,
        help="Minimum crop side length to keep",
    )
    parser.add_argument(
        "--max-crop-size",
        type=int,
        default=0,
        help="Maximum crop side length to keep (0 = no limit). Use to skip arch-level bboxes.",
    )
    return parser.parse_args()


def patient_id_from_filename(file_name: str) -> str:
    """Extract anonymized patient ID from image file name."""
    stem = Path(file_name).stem
    return stem.split("-")[0]


def assign_split(patient_id: str) -> str:
    """Assign deterministic split by patient ID hash.

    Distribution target:
    - train: 80%
    - val: 10%
    - test: 10%
    """
    digest = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def normalize_path(path: Path) -> str:
    """Convert path to forward-slash string."""
    return path.as_posix()


def remap_disease_label(disease_label: str) -> str:
    """Merge ultra-rare labels into stable parent classes for training."""
    return DISEASE_LABEL_REMAP.get(disease_label, disease_label)


def make_seed_report(disease_label: str) -> str:
    """Create a lightweight seed report from disease code.

    This is only a bootstrap hint. Final medical text should be reviewed by experts.
    """
    mapping = {
        "H": "Khỏe mạnh",
        "C": "Nghi ngờ sâu răng",
        "R": "Răng đã phục hồi",
        "Te": "Răng đã điều trị nội nha",
        "Im": "Răng cấy ghép implant",
        "Rr": "Chân răng còn lại",
        "M3i": "Răng số 8 mọc ngầm",
        "M3f": "Răng số 8 đang phát triển",
        "CpuM": "Có mão răng (mixed)",
        "Dc": "Phá hủy thân răng",
        "Di": "Mòn/incisal wear",
        "P": "Pontic",
        "Cp": "Mão răng đơn",
    }
    return mapping.get(disease_label, f"Nhãn bệnh: {disease_label}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV file with UTF-8 encoding."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Build crop dataset and return summary metadata."""
    root = args.root.resolve()
    input_json = (root / args.input_json).resolve()
    output_dir = (root / args.output_dir).resolve()

    images_dir = root / "images"

    macro_dir = output_dir / "macro"
    micro_dir = output_dir / "micro"
    micro_images_dir = micro_dir / "images"
    metadata_dir = output_dir / "metadata"

    macro_dir.mkdir(parents=True, exist_ok=True)
    micro_images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    macro_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    split_counter = Counter()
    disease_counter = Counter()

    for image_item in data.get("images", []):
        file_name = str(image_item.get("file_name", "")).strip()
        if not file_name:
            continue

        patient_id = patient_id_from_filename(file_name)
        split = assign_split(patient_id)

        src_image_path = images_dir / file_name
        src_image_rel = normalize_path(src_image_path.relative_to(root))

        macro_rows.append(
            {
                "image_path": src_image_rel,
                "general_report": "",
                "general_report_seed": "Mời bác sĩ nhập nhận xét tổng quan toàn hàm.",
                "patient_id": patient_id,
                "split": split,
            }
        )

        split_counter[split] += 1

        if not src_image_path.exists():
            skipped_rows.append(
                {
                    "file_name": file_name,
                    "disease_annotation_id": "image_level",
                    "reason": "missing_source_image",
                }
            )
            continue

        image = cv2.imread(str(src_image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped_rows.append(
                {
                    "file_name": file_name,
                    "disease_annotation_id": "image_level",
                    "reason": "cannot_read_image",
                }
            )
            continue

        stem = Path(file_name).stem
        image_crop_dir = micro_images_dir / stem
        crop_dir_created = False

        for tooth in image_item.get("teeth", []):
            ann_id = tooth.get("disease_annotation_id")
            bbox = tooth.get("bbox_xywh")
            fdi_label = tooth.get("fdi_label")
            raw_disease_label = str(tooth.get("disease_label", "")).strip()
            disease_label = remap_disease_label(raw_disease_label)

            if fdi_label is None:
                skipped_rows.append(
                    {
                        "file_name": file_name,
                        "disease_annotation_id": ann_id,
                        "reason": "missing_fdi_label",
                    }
                )
                continue

            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped_rows.append(
                    {
                        "file_name": file_name,
                        "disease_annotation_id": ann_id,
                        "reason": "invalid_bbox",
                    }
                )
                continue

            crop_result = get_contextual_square_crop_from_array(
                image=image,
                bbox=bbox,
                expand_width=float(args.expand_width),
                expand_height=float(args.expand_height),
            )
            if crop_result is None:
                skipped_rows.append(
                    {
                        "file_name": file_name,
                        "disease_annotation_id": ann_id,
                        "reason": "crop_failed",
                    }
                )
                continue

            crop_h, crop_w = crop_result.image.shape[:2]
            if min(crop_h, crop_w) < int(args.min_crop_size):
                skipped_rows.append(
                    {
                        "file_name": file_name,
                        "disease_annotation_id": ann_id,
                        "reason": "crop_too_small",
                    }
                )
                continue

            if int(args.max_crop_size) > 0 and max(crop_h, crop_w) > int(args.max_crop_size):
                skipped_rows.append(
                    {
                        "file_name": file_name,
                        "disease_annotation_id": ann_id,
                        "reason": "crop_too_large",
                    }
                )
                continue

            safe_ann = str(ann_id)
            safe_fdi = str(fdi_label)
            crop_name = f"{stem}_ann{safe_ann}_fdi{safe_fdi}.jpg"

            if not crop_dir_created:
                image_crop_dir.mkdir(parents=True, exist_ok=True)
                crop_dir_created = True

            crop_path = image_crop_dir / crop_name

            ok = cv2.imwrite(str(crop_path), crop_result.image)
            if not ok:
                skipped_rows.append(
                    {
                        "file_name": file_name,
                        "disease_annotation_id": ann_id,
                        "reason": "crop_save_failed",
                    }
                )
                continue

            crop_rel = normalize_path(crop_path.relative_to(root))

            micro_rows.append(
                {
                    "crop_image_path": crop_rel,
                    "tooth_id": fdi_label,
                    "detailed_report": "",
                    "detailed_report_seed": make_seed_report(disease_label),
                    "disease_label": disease_label,
                    "source_image_path": src_image_rel,
                    "source_image_name": file_name,
                    "disease_annotation_id": ann_id,
                    "patient_id": patient_id,
                    "split": split,
                    "bbox_x": round(float(bbox[0]), 4),
                    "bbox_y": round(float(bbox[1]), 4),
                    "bbox_w": round(float(bbox[2]), 4),
                    "bbox_h": round(float(bbox[3]), 4),
                    "crop_req_x1": crop_result.req_x1,
                    "crop_req_y1": crop_result.req_y1,
                    "crop_req_x2": crop_result.req_x2,
                    "crop_req_y2": crop_result.req_y2,
                }
            )

            disease_counter[disease_label] += 1

    macro_fields = [
        "image_path",
        "general_report",
        "general_report_seed",
        "patient_id",
        "split",
    ]
    micro_fields = [
        "crop_image_path",
        "tooth_id",
        "detailed_report",
        "detailed_report_seed",
        "disease_label",
        "source_image_path",
        "source_image_name",
        "disease_annotation_id",
        "patient_id",
        "split",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "crop_req_x1",
        "crop_req_y1",
        "crop_req_x2",
        "crop_req_y2",
    ]

    write_csv(macro_dir / "macro_labels.csv", macro_rows, macro_fields)
    write_csv(micro_dir / "micro_labels.csv", micro_rows, micro_fields)

    skipped_fields = ["file_name", "disease_annotation_id", "reason"]
    write_csv(metadata_dir / "skipped_items.csv", skipped_rows, skipped_fields)

    summary = {
        "input_json": normalize_path(input_json.relative_to(root)),
        "output_dir": normalize_path(output_dir.relative_to(root)),
        "macro_rows": len(macro_rows),
        "micro_rows": len(micro_rows),
        "skipped_rows": len(skipped_rows),
        "split_distribution_images": dict(split_counter),
        "disease_distribution_micro": dict(disease_counter),
        "disease_classes_micro": len(disease_counter),
        "disease_label_remap": DISEASE_LABEL_REMAP,
        "expand_width": float(args.expand_width),
        "expand_height": float(args.expand_height),
        "min_crop_size": int(args.min_crop_size),
        "max_crop_size": int(args.max_crop_size),
    }

    save_json_path = metadata_dir / "build_summary.json"
    with save_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    readme_text = (
        "PAN924 Instruction Dataset (Macro + Micro)\n\n"
        "Structure:\n"
        "- macro/macro_labels.csv: Panorama-level labels template\n"
        "- micro/images/: Contextual square tooth crops\n"
        "- micro/micro_labels.csv: Tooth-level metadata and labels\n"
        "- metadata/build_summary.json: Build statistics\n"
        "- metadata/skipped_items.csv: Items skipped during processing\n\n"
        "Notes:\n"
        "- Columns detailed_report and general_report are left blank for expert annotation.\n"
        "- *_seed columns provide optional draft hints only.\n"
    )

    (output_dir / "README.txt").write_text(readme_text, encoding="utf-8")

    return summary


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    summary = build_dataset(args)

    print("=== PAN924 Tooth Crop Dataset Build ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
