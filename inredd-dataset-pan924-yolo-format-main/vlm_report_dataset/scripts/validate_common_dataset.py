"""Validate PAN924 VLM common JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REGIONS = [
    "image_upper_left",
    "image_upper_right",
    "image_lower_left",
    "image_lower_right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VLM common dataset")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--condition-map", type=Path, default=Path("vlm_report_dataset/config/condition_map.json"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--report-path", type=Path, default=Path("vlm_report_dataset/common/metadata/validation_report.json"))
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSONL row: {exc}") from exc
    return rows


def teeth_key(teeth: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return sorted((str(tooth["fdi"]), str(tooth["condition"])) for tooth in teeth)


def validate_full(row: dict[str, Any], target: dict[str, Any], errors: list[str]) -> None:
    labels = row.get("labels", {}).get("regions")
    regions = target.get("regions")
    if not isinstance(labels, dict) or not isinstance(regions, dict):
        errors.append(f"{row['id']}: full row missing regions in labels or target")
        return
    if set(labels.keys()) != set(REGIONS) or set(regions.keys()) != set(REGIONS):
        errors.append(f"{row['id']}: full row region keys mismatch")
        return
    if not isinstance(target.get("summary"), str) or not target["summary"].strip():
        errors.append(f"{row['id']}: full row empty summary")
    for region in REGIONS:
        if teeth_key(labels[region]) != teeth_key(regions[region].get("teeth", [])):
            errors.append(f"{row['id']}: target teeth do not match labels for {region}")
        if not isinstance(regions[region].get("comment"), str) or not regions[region]["comment"].strip():
            errors.append(f"{row['id']}: empty comment for {region}")


def validate_regional(row: dict[str, Any], target: dict[str, Any], errors: list[str]) -> None:
    labels = row.get("labels", {})
    if labels.get("region") != target.get("region"):
        errors.append(f"{row['id']}: regional target region does not match labels")
    if teeth_key(labels.get("teeth", [])) != teeth_key(target.get("teeth", [])):
        errors.append(f"{row['id']}: regional target teeth do not match labels")
    if not isinstance(target.get("comment"), str) or not target["comment"].strip():
        errors.append(f"{row['id']}: regional row empty comment")


def iter_target_teeth(row: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    if row["task"] == "full_quadrant_report":
        teeth: list[dict[str, Any]] = []
        for region in REGIONS:
            teeth.extend(target["regions"][region]["teeth"])
        return teeth
    if row["task"] == "regional_report":
        return target["teeth"]
    return []


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = root / args.input_dir
    condition_map = load_json(root / args.condition_map)
    valid_conditions = set(condition_map.keys())

    errors: list[str] = []
    source_to_split: dict[str, str] = {}
    stats = {
        "rows_by_split": {},
        "rows_by_task": Counter(),
        "rows_by_view": Counter(),
        "condition_counts": Counter(),
        "comment_source_counts": Counter(),
    }

    for split in args.splits:
        path = input_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        stats["rows_by_split"][split] = len(rows)

        for row in rows:
            for key in ["id", "image", "source_image", "view", "task", "messages", "labels", "metadata"]:
                if key not in row:
                    errors.append(f"{row.get('id', '<missing-id>')}: missing key {key}")
                    continue

            source = row.get("source_image")
            if source in source_to_split and source_to_split[source] != split:
                errors.append(f"{source}: split leak between {source_to_split[source]} and {split}")
            source_to_split[source] = split

            image_path = root / row["image"]
            if not image_path.exists():
                errors.append(f"{row['id']}: image file not found: {row['image']}")

            messages = row.get("messages", [])
            if len(messages) != 2:
                errors.append(f"{row['id']}: expected exactly 2 messages")
                continue
            if messages[0].get("role") != "user" or "<image>" not in messages[0].get("content", ""):
                errors.append(f"{row['id']}: user message must contain <image>")
            if messages[1].get("role") != "assistant":
                errors.append(f"{row['id']}: second message must be assistant")

            try:
                target = json.loads(messages[1]["content"])
            except json.JSONDecodeError as exc:
                errors.append(f"{row['id']}: assistant content is not valid JSON: {exc}")
                continue

            if row["task"] == "full_quadrant_report":
                validate_full(row, target, errors)
            elif row["task"] == "regional_report":
                validate_regional(row, target, errors)
            else:
                errors.append(f"{row['id']}: unknown task {row['task']}")

            for tooth in iter_target_teeth(row, target):
                condition = tooth.get("condition")
                if condition not in valid_conditions:
                    errors.append(f"{row['id']}: unknown condition {condition}")
                stats["condition_counts"][condition] += 1

            stats["rows_by_task"][row["task"]] += 1
            stats["rows_by_view"][row["view"]] += 1
            stats["comment_source_counts"][row["metadata"].get("comment_source", "unknown")] += 1

    report = {
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors[:200],
        "unique_source_images": len(source_to_split),
        "rows_by_split": stats["rows_by_split"],
        "rows_by_task": dict(stats["rows_by_task"]),
        "rows_by_view": dict(stats["rows_by_view"]),
        "condition_counts": dict(stats["condition_counts"]),
        "comment_source_counts": dict(stats["comment_source_counts"]),
    }

    report_path = root / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
