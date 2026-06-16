"""Build the model-agnostic 5-view VLM report dataset.

This script only uses ground-truth annotations for FDI and condition labels.
It creates deterministic raw comments first; the clinical report generator
rewrites only the comment and summary fields later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image


REGIONS = [
    "image_upper_left",
    "image_upper_right",
    "image_lower_left",
    "image_lower_right",
]


@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PAN924 5-view VLM common dataset")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/dataset_final_v2.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--condition-map", type=Path, default=Path("vlm_report_dataset/config/condition_map.json"))
    parser.add_argument("--label-remap", type=Path, default=Path("vlm_report_dataset/config/disease_label_remap.json"))
    parser.add_argument("--full-prompt", type=Path, default=Path("vlm_report_dataset/config/prompt_templates/full_report_prompt.txt"))
    parser.add_argument("--regional-prompt", type=Path, default=Path("vlm_report_dataset/config/prompt_templates/regional_report_prompt.txt"))
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--overlap-x", type=float, default=0.10)
    parser.add_argument("--overlap-y", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tooth_center(tooth: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = tooth["bbox_xywh"]
    return x + w / 2.0, y + h / 2.0


def canonical_condition(raw_condition: str, label_remap: dict[str, str]) -> str:
    return label_remap.get(raw_condition, raw_condition)


def normalize_tooth(
    tooth: dict[str, Any],
    condition_map: dict[str, Any],
    label_remap: dict[str, str],
) -> dict[str, Any]:
    fdi = str(int(tooth["fdi_label"]))
    raw_condition = str(tooth["disease_label"])
    condition = canonical_condition(raw_condition, label_remap)
    cx, cy = tooth_center(tooth)
    x, y, w, h = tooth["bbox_xywh"]
    condition_info = condition_map.get(condition, {"name": condition})
    return {
        "fdi": fdi,
        "condition": condition,
        "raw_condition": raw_condition,
        "condition_name": condition_info["name"],
        "bbox_xywh": [float(x), float(y), float(w), float(h)],
        "center_xy": [float(cx), float(cy)],
        "disease_annotation_id": tooth.get("disease_annotation_id"),
    }


def region_for_point(x: float, y: float, x_split: float, y_split: float) -> str:
    upper_lower = "upper" if y < y_split else "lower"
    left_right = "left" if x < x_split else "right"
    return f"image_{upper_lower}_{left_right}"


def compute_split_and_crops(image_record: dict[str, Any]) -> tuple[float, float, dict[str, CropBox]]:
    width = int(image_record["width"])
    height = int(image_record["height"])
    teeth = image_record.get("teeth", [])

    x_split = width / 2.0
    upper_ys: list[float] = []
    lower_ys: list[float] = []

    for tooth in teeth:
        fdi = int(tooth["fdi_label"])
        _, cy = tooth_center(tooth)
        quadrant = fdi // 10
        if quadrant in (1, 2):
            upper_ys.append(cy)
        elif quadrant in (3, 4):
            lower_ys.append(cy)

    if upper_ys and lower_ys:
        y_split = (median(upper_ys) + median(lower_ys)) / 2.0
    else:
        y_split = height / 2.0

    overlap_x = width * 0.10
    overlap_y = height * 0.10
    crops = {
        "image_upper_left": CropBox(0, 0, math.ceil(x_split + overlap_x), math.ceil(y_split + overlap_y)),
        "image_upper_right": CropBox(math.floor(x_split - overlap_x), 0, width, math.ceil(y_split + overlap_y)),
        "image_lower_left": CropBox(0, math.floor(y_split - overlap_y), math.ceil(x_split + overlap_x), height),
        "image_lower_right": CropBox(math.floor(x_split - overlap_x), math.floor(y_split - overlap_y), width, height),
    }
    return x_split, y_split, crops


def compute_crop_boxes(
    image_record: dict[str, Any],
    overlap_x_ratio: float,
    overlap_y_ratio: float,
) -> tuple[float, float, dict[str, CropBox]]:
    width = int(image_record["width"])
    height = int(image_record["height"])
    x_split, y_split, _ = compute_split_and_crops(image_record)

    overlap_x = width * overlap_x_ratio
    overlap_y = height * overlap_y_ratio

    def clamp_x(value: float) -> int:
        return max(0, min(width, int(round(value))))

    def clamp_y(value: float) -> int:
        return max(0, min(height, int(round(value))))

    crops = {
        "image_upper_left": CropBox(0, 0, clamp_x(x_split + overlap_x), clamp_y(y_split + overlap_y)),
        "image_upper_right": CropBox(clamp_x(x_split - overlap_x), 0, width, clamp_y(y_split + overlap_y)),
        "image_lower_left": CropBox(0, clamp_y(y_split - overlap_y), clamp_x(x_split + overlap_x), height),
        "image_lower_right": CropBox(clamp_x(x_split - overlap_x), clamp_y(y_split - overlap_y), width, height),
    }
    return x_split, y_split, crops


def copy_and_crop_views(
    source_image_path: Path,
    image_output_dir: Path,
    repo_root: Path,
    crops: dict[str, CropBox],
    jpeg_quality: int,
    overwrite: bool,
) -> dict[str, str]:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "full": image_output_dir / "full.jpg",
        **{region: image_output_dir / f"{region}.jpg" for region in REGIONS},
    }

    if overwrite or not outputs["full"].exists():
        shutil.copy2(source_image_path, outputs["full"])

    with Image.open(source_image_path) as img:
        rgb = img.convert("RGB")
        for region, box in crops.items():
            out_path = outputs[region]
            if out_path.exists() and not overwrite:
                continue
            crop = rgb.crop((box.x1, box.y1, box.x2, box.y2))
            crop.save(out_path, quality=jpeg_quality)

    return {
        view: str(path.resolve().relative_to(repo_root).as_posix())
        for view, path in outputs.items()
    }


def teeth_for_region(teeth: list[dict[str, Any]], crop: CropBox) -> list[dict[str, Any]]:
    selected = []
    for tooth in teeth:
        cx, cy = tooth["center_xy"]
        if crop.contains(cx, cy):
            selected.append(tooth)
    return sort_teeth(selected)


def sort_teeth(teeth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(teeth, key=lambda item: int(item["fdi"]))


def teeth_for_model_output(teeth: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "fdi": tooth["fdi"],
            "condition": tooth["condition"],
            "condition_name": tooth["condition_name"],
        }
        for tooth in teeth
    ]


def template_region_comment(region: str, teeth: list[dict[str, Any]]) -> str:
    if not teeth:
        return f"No annotated teeth are listed for {region}."
    findings = [f"tooth {t['fdi']} is {t['condition_name']}" for t in teeth]
    return f"{region} includes " + ", ".join(findings) + "."


def template_summary(regions: dict[str, dict[str, Any]]) -> str:
    counts = Counter()
    total_teeth = 0
    for payload in regions.values():
        for tooth in payload["teeth"]:
            counts[tooth["condition_name"]] += 1
            total_teeth += 1
    if total_teeth == 0:
        return "No annotated teeth are available for this panoramic radiograph."
    common = ", ".join(f"{name}: {count}" for name, count in counts.most_common(4))
    return f"The annotated panorama contains {total_teeth} teeth across the four image regions. Main labels: {common}."


def build_full_target(
    teeth: list[dict[str, Any]],
    x_split: float,
    y_split: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    region_teeth: dict[str, list[dict[str, Any]]] = {region: [] for region in REGIONS}
    for tooth in teeth:
        cx, cy = tooth["center_xy"]
        region_teeth[region_for_point(cx, cy, x_split, y_split)].append(tooth)

    regions: dict[str, dict[str, Any]] = {}
    labels: dict[str, list[dict[str, str]]] = {}
    for region in REGIONS:
        output_teeth = teeth_for_model_output(sort_teeth(region_teeth[region]))
        regions[region] = {
            "teeth": output_teeth,
            "comment": template_region_comment(region, output_teeth),
        }
        labels[region] = output_teeth

    target = {
        "regions": regions,
        "summary": template_summary(regions),
    }
    return target, {"regions": labels}


def build_regional_target(region: str, teeth: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    output_teeth = teeth_for_model_output(teeth)
    target = {
        "region": region,
        "teeth": output_teeth,
        "comment": template_region_comment(region, output_teeth),
    }
    return target, {"region": region, "teeth": output_teeth}


def image_feature_counts(image_record: dict[str, Any], label_remap: dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for tooth in image_record.get("teeth", []):
        counts[canonical_condition(str(tooth["disease_label"]), label_remap)] += 1
    if not counts:
        counts["NO_ANNOTATED_TEETH"] = 1
    return counts


def stratified_group_split(
    image_records: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
    label_remap: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    split_names = ["train", "val", "test"]
    total_images = len(image_records)
    targets = {
        "train": round(total_images * ratios["train"]),
        "val": round(total_images * ratios["val"]),
    }
    targets["test"] = total_images - targets["train"] - targets["val"]

    global_counts: Counter = Counter()
    per_image_counts: dict[str, Counter] = {}
    for record in image_records:
        key = record["file_name"]
        counts = image_feature_counts(record, label_remap)
        per_image_counts[key] = counts
        global_counts.update(counts)

    assignments: dict[str, str] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def stable_hash(text: str) -> str:
        return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()

    def primary_label(record: dict[str, Any]) -> str:
        counts = per_image_counts[record["file_name"]]
        return min(counts.keys(), key=lambda label: (global_counts[label], label))

    for record in image_records:
        buckets[primary_label(record)].append(record)

    for bucket_label, bucket_records in sorted(buckets.items()):
        ordered = sorted(bucket_records, key=lambda record: stable_hash(record["file_name"]))
        n = len(ordered)
        bucket_targets = {
            "train": round(n * ratios["train"]),
            "val": round(n * ratios["val"]),
        }
        bucket_targets["test"] = n - bucket_targets["train"] - bucket_targets["val"]

        cursor = 0
        for split in split_names:
            for record in ordered[cursor : cursor + bucket_targets[split]]:
                assignments[record["file_name"]] = split
            cursor += bucket_targets[split]

    split_sizes = Counter(assignments.values())

    def move_one(source_split: str, target_split: str) -> bool:
        candidates = [
            record
            for record in image_records
            if assignments[record["file_name"]] == source_split
        ]
        candidates = sorted(
            candidates,
            key=lambda record: (
                primary_label(record),
                stable_hash(record["file_name"]),
            ),
        )
        for record in candidates:
            assignments[record["file_name"]] = target_split
            split_sizes[source_split] -= 1
            split_sizes[target_split] += 1
            return True
        return False

    while any(split_sizes[split] != targets[split] for split in split_names):
        surplus = next((split for split in split_names if split_sizes[split] > targets[split]), None)
        deficit = next((split for split in split_names if split_sizes[split] < targets[split]), None)
        if surplus is None or deficit is None or not move_one(surplus, deficit):
            break

    split_counts = {split: Counter() for split in split_names}
    for record in image_records:
        split_counts[assignments[record["file_name"]]].update(per_image_counts[record["file_name"]])

    report = {
        "seed": seed,
        "ratios": ratios,
        "target_image_counts": targets,
        "actual_image_counts": dict(split_sizes),
        "global_condition_counts": dict(global_counts),
        "split_condition_counts": {
            split: dict(split_counts[split]) for split in split_names
        },
        "bucket_counts": {label: len(records) for label, records in sorted(buckets.items())},
    }
    return assignments, report


def make_sample(
    sample_id: str,
    image_path: str,
    source_image: str,
    view: str,
    split: str,
    task: str,
    user_prompt: str,
    target: dict[str, Any],
    labels: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "image": image_path,
        "source_image": source_image,
        "view": view,
        "task": task,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "labels": labels,
        "metadata": {
            **metadata,
            "split": split,
            "comment_source": "template",
        },
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    annotation_path = root / args.annotations
    images_dir = root / args.images_dir
    output_dir = root / args.output_dir
    image_output_root = output_dir / "images"
    raw_output_dir = output_dir / "raw"

    condition_map = load_json(root / args.condition_map)
    label_remap = load_json(root / args.label_remap)
    full_prompt = read_text(root / args.full_prompt)
    regional_prompt_template = read_text(root / args.regional_prompt)
    dataset = load_json(annotation_path)
    image_records = dataset["images"]

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    assignments, split_report = stratified_group_split(image_records, ratios, args.seed, label_remap)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    crop_rows: list[dict[str, Any]] = []

    for image_record in image_records:
        source_file = image_record["file_name"]
        source_path = images_dir / source_file
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        stem = Path(source_file).stem
        split = assignments[source_file]
        x_split, y_split, crop_boxes = compute_crop_boxes(
            image_record,
            overlap_x_ratio=args.overlap_x,
            overlap_y_ratio=args.overlap_y,
        )
        image_paths = copy_and_crop_views(
            source_path,
            image_output_root / stem,
            root,
            crop_boxes,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite_images,
        )

        normalized_teeth = [
            normalize_tooth(tooth, condition_map, label_remap)
            for tooth in image_record.get("teeth", [])
        ]
        normalized_teeth = sort_teeth(normalized_teeth)
        common_metadata = {
            "image_id": image_record.get("image_id"),
            "width": image_record["width"],
            "height": image_record["height"],
            "x_split": x_split,
            "y_split": y_split,
            "crop_boxes": {region: box.as_list() for region, box in crop_boxes.items()},
        }

        full_target, full_labels = build_full_target(normalized_teeth, x_split, y_split)
        rows_by_split[split].append(
            make_sample(
                sample_id=f"{stem}_full_report",
                image_path=image_paths["full"],
                source_image=source_file,
                view="full",
                split=split,
                task="full_quadrant_report",
                user_prompt=full_prompt,
                target=full_target,
                labels=full_labels,
                metadata=common_metadata,
            )
        )

        for region in REGIONS:
            regional_teeth = teeth_for_region(normalized_teeth, crop_boxes[region])
            regional_target, regional_labels = build_regional_target(region, regional_teeth)
            rows_by_split[split].append(
                make_sample(
                    sample_id=f"{stem}_{region}_report",
                    image_path=image_paths[region],
                    source_image=source_file,
                    view=region,
                    split=split,
                    task="regional_report",
                    user_prompt=regional_prompt_template.replace("{region}", region),
                    target=regional_target,
                    labels=regional_labels,
                    metadata=common_metadata,
                )
            )
            crop_rows.append(
                {
                    "source_image": source_file,
                    "view": region,
                    "split": split,
                    "crop_box": json.dumps(crop_boxes[region].as_list()),
                    "teeth_count": len(regional_teeth),
                    "fdi": " ".join(tooth["fdi"] for tooth in regional_teeth),
                    "conditions": " ".join(tooth["condition"] for tooth in regional_teeth),
                }
            )

    for split in ["train", "val", "test"]:
        rows = rows_by_split[split]
        write_jsonl(raw_output_dir / f"{split}.jsonl", rows)
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    build_summary = {
        "input_json": str(args.annotations.as_posix()),
        "output_dir": str(args.output_dir.as_posix()),
        "source_images": len(image_records),
        "views_per_image": 5,
        "total_samples": sum(len(rows) for rows in rows_by_split.values()),
        "samples_by_split": {split: len(rows_by_split[split]) for split in ["train", "val", "test"]},
        "image_split_counts": split_report["actual_image_counts"],
        "overlap_x": args.overlap_x,
        "overlap_y": args.overlap_y,
        "comment_source": "template",
        "label_remap": label_remap,
    }
    write_json(output_dir / "metadata" / "build_summary.json", build_summary)
    write_json(output_dir / "metadata" / "split_report.json", split_report)
    write_csv(
        output_dir / "metadata" / "crop_manifest.csv",
        crop_rows,
        ["source_image", "view", "split", "crop_box", "teeth_count", "fdi", "conditions"],
    )

    print(json.dumps(build_summary, indent=2))


if __name__ == "__main__":
    main()
