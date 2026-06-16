"""Generate clinical report text from fixed PAN924 ground-truth labels.

This script does not call any external API. It only writes the natural-language
`comment` and `summary` fields. The FDI numbers, condition labels, image paths,
splits, and JSON structure come from `common/raw/*.jsonl` and are preserved.

Report style:
- mention abnormal/non-healthy findings first
- group findings by condition
- avoid listing every healthy tooth when abnormalities exist
- keep empty/all-healthy regions explicit and short
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REGIONS = [
    "image_upper_left",
    "image_upper_right",
    "image_lower_left",
    "image_lower_right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clinical PAN924 VLM report comments")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def tooth_number_phrase(teeth: list[dict[str, Any]]) -> str:
    fdis = [str(tooth["fdi"]) for tooth in teeth]
    if not fdis:
        return ""
    if len(fdis) == 1:
        return f"tooth {fdis[0]}"
    if len(fdis) == 2:
        return f"teeth {fdis[0]} and {fdis[1]}"
    return "teeth " + ", ".join(fdis[:-1]) + f", and {fdis[-1]}"


def condition_noun(condition_name: str, count: int) -> str:
    names = {
        "caries": "caries",
        "restored": "restoration" if count == 1 else "restorations",
        "endodontic treatment": "endodontic treatment",
        "implant": "implant" if count == 1 else "implants",
        "residual root": "residual root" if count == 1 else "residual roots",
        "impacted third molar": "impacted third molar" if count == 1 else "impacted third molars",
        "developing third molar": "developing third molar" if count == 1 else "developing third molars",
        "prosthetic crown": "prosthetic crown" if count == 1 else "prosthetic crowns",
        "crown destruction": "crown destruction",
        "incisal or occlusal wear": "incisal or occlusal wear",
        "pontic": "pontic" if count == 1 else "pontics",
    }
    return names.get(condition_name, condition_name)


def finding_phrase(condition_name: str, teeth: list[dict[str, Any]]) -> str:
    numbers = tooth_number_phrase(teeth)
    if condition_name == "caries":
        return f"caries on {numbers}"
    if condition_name == "restored":
        if len(teeth) == 1:
            return f"a restoration on {numbers}"
        return f"restorations on {numbers}"
    if condition_name == "endodontic treatment":
        return f"endodontic treatment on {numbers}"
    if condition_name in {"incisal or occlusal wear", "crown destruction"}:
        return f"{condition_noun(condition_name, len(teeth))} on {numbers}"
    if condition_name in {"implant", "residual root", "impacted third molar", "developing third molar", "prosthetic crown", "pontic"}:
        noun = condition_noun(condition_name, len(teeth))
        article = "an" if condition_name in {"implant", "impacted third molar"} else "a"
        if len(teeth) == 1:
            return f"{article} {noun} at {numbers}"
        return f"{noun} at {numbers}"
    return f"{condition_noun(condition_name, len(teeth))} at {numbers}"


def join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def group_abnormal(teeth: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for tooth in teeth:
        if tooth.get("condition") == "H":
            continue
        grouped.setdefault(tooth["condition_name"], []).append(tooth)
    return grouped


def clinical_region_comment(teeth: list[dict[str, Any]], empty_scope: str = "region") -> str:
    if not teeth:
        return f"No annotated teeth are available in this {empty_scope}."

    grouped = group_abnormal(teeth)
    if not grouped:
        return f"The annotated teeth in this {empty_scope} are healthy."

    phrases = [
        finding_phrase(condition_name, grouped[condition_name])
        for condition_name in sorted(grouped.keys())
    ]
    healthy_count = len([tooth for tooth in teeth if tooth.get("condition") == "H"])
    comment = f"This {empty_scope} shows {join_phrases(phrases)}."
    if healthy_count:
        comment += " The remaining annotated teeth are healthy."
    return comment


def clinical_full_summary(regions: dict[str, dict[str, Any]]) -> str:
    teeth = []
    seen: set[tuple[str, str]] = set()
    for region in REGIONS:
        for tooth in regions[region]["teeth"]:
            key = (str(tooth["fdi"]), str(tooth["condition"]))
            if key in seen:
                continue
            seen.add(key)
            teeth.append(tooth)

    if not teeth:
        return "No annotated teeth are available in this panoramic radiograph."

    grouped = group_abnormal(teeth)
    if not grouped:
        return "All annotated teeth in this panoramic radiograph are healthy."

    phrases = [
        finding_phrase(condition_name, grouped[condition_name])
        for condition_name in sorted(grouped.keys())
    ]
    healthy_count = len([tooth for tooth in teeth if tooth.get("condition") == "H"])
    summary = f"Overall, the annotated findings include {join_phrases(phrases)}."
    if healthy_count:
        summary += " The remaining annotated teeth are healthy."
    return summary


def clinical_response_for_target(task: str, target: dict[str, Any]) -> dict[str, Any]:
    if task == "full_quadrant_report":
        return {
            "regions": {
                region: {"comment": clinical_region_comment(target["regions"][region]["teeth"])}
                for region in REGIONS
            },
            "summary": clinical_full_summary(target["regions"]),
        }
    if task == "regional_report":
        return {"comment": clinical_region_comment(target["teeth"])}
    raise ValueError(f"Unsupported task: {task}")


def merge_response_into_target(target: dict[str, Any], task: str, response: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(target, ensure_ascii=False))
    if task == "full_quadrant_report":
        for region in REGIONS:
            merged["regions"][region]["comment"] = response["regions"][region]["comment"]
        merged["summary"] = response["summary"]
    elif task == "regional_report":
        merged["comment"] = response["comment"]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return merged


def update_row(row: dict[str, Any]) -> dict[str, Any]:
    target = json.loads(row["messages"][1]["content"])
    response = clinical_response_for_target(row["task"], target)
    updated_target = merge_response_into_target(target, row["task"], response)
    updated = json.loads(json.dumps(row, ensure_ascii=False))
    updated["messages"][1]["content"] = json.dumps(updated_target, ensure_ascii=False)
    updated["metadata"]["comment_source"] = "clinical_template_v2"
    return updated


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir

    summary = {
        "generator": "clinical_template_v2",
        "input_dir": str(args.input_dir.as_posix()),
        "output_dir": str(args.output_dir.as_posix()),
        "splits": args.splits,
        "rows_by_split": {},
    }

    for split in args.splits:
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        updated_rows = [update_row(row) for row in rows]
        write_jsonl(output_dir / f"{split}.jsonl", updated_rows)
        summary["rows_by_split"][split] = len(updated_rows)

    write_json(output_dir / "metadata" / "clinical_generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
