"""Build micro.jsonl and macro.jsonl for Qwen2.5-VL instruction tuning (ms-swift format).

Micro : one entry per tooth crop  — image + disease classification
Macro : one entry per panorama    — image + overall report (skipped when report is empty)

Output format per line:
  {"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}], "images": ["path"]}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FDI_NAMES: dict[int, str] = {
    # Upper right
    11: "Răng cửa giữa hàm trên phải",
    12: "Răng cửa bên hàm trên phải",
    13: "Răng nanh hàm trên phải",
    14: "Răng tiền hàm nhỏ thứ nhất hàm trên phải",
    15: "Răng tiền hàm nhỏ thứ hai hàm trên phải",
    16: "Răng hàm lớn thứ nhất hàm trên phải",
    17: "Răng hàm lớn thứ hai hàm trên phải",
    18: "Răng khôn hàm trên phải",
    # Upper left
    21: "Răng cửa giữa hàm trên trái",
    22: "Răng cửa bên hàm trên trái",
    23: "Răng nanh hàm trên trái",
    24: "Răng tiền hàm nhỏ thứ nhất hàm trên trái",
    25: "Răng tiền hàm nhỏ thứ hai hàm trên trái",
    26: "Răng hàm lớn thứ nhất hàm trên trái",
    27: "Răng hàm lớn thứ hai hàm trên trái",
    28: "Răng khôn hàm trên trái",
    # Lower left
    31: "Răng cửa giữa hàm dưới trái",
    32: "Răng cửa bên hàm dưới trái",
    33: "Răng nanh hàm dưới trái",
    34: "Răng tiền hàm nhỏ thứ nhất hàm dưới trái",
    35: "Răng tiền hàm nhỏ thứ hai hàm dưới trái",
    36: "Răng hàm lớn thứ nhất hàm dưới trái",
    37: "Răng hàm lớn thứ hai hàm dưới trái",
    38: "Răng khôn hàm dưới trái",
    # Lower right
    41: "Răng cửa giữa hàm dưới phải",
    42: "Răng cửa bên hàm dưới phải",
    43: "Răng nanh hàm dưới phải",
    44: "Răng tiền hàm nhỏ thứ nhất hàm dưới phải",
    45: "Răng tiền hàm nhỏ thứ hai hàm dưới phải",
    46: "Răng hàm lớn thứ nhất hàm dưới phải",
    47: "Răng hàm lớn thứ hai hàm dưới phải",
    48: "Răng khôn hàm dưới phải",
}

# Single canonical prompt. The inference pipeline (Faster R-CNN → crop → VLM)
# auto-generates this exact string from FRCNN output, so training and inference
# use byte-identical prompts. This is the standard recipe for classification
# VLMs integrated into a deterministic pipeline (CheXagent, Qwen2.5-VL official).
MICRO_INSTRUCTION_TEMPLATE = "Đây là ảnh X-quang răng số {fdi}. Hãy chẩn đoán tình trạng của răng này."

MACRO_INSTRUCTION_TEMPLATE = "Hãy nhận xét tổng quan tình trạng hàm răng trong ảnh X-quang toàn cảnh này."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruction tuning JSONL files")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("prepared_dataset/pan924_instruction_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_dataset/pan924_instruction_v2"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to generate files for",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


DISEASE_DESCRIPTIONS: dict[str, str] = {
    "H":    "Răng khỏe mạnh, không có dấu hiệu bất thường.",
    "C":    "Nghi ngờ có tổn thương sâu răng.",
    "R":    "Răng đã được phục hồi (trám hoặc chụp răng).",
    "Te":   "Răng đã được điều trị nội nha (lấy tủy).",
    "Im":   "Răng cấy ghép implant.",
    "Rr":   "Chân răng còn lại sau khi mất thân răng.",
    "M3i":  "Răng mọc ngầm trong xương hàm.",
    "M3f":  "Răng đang trong giai đoạn phát triển.",
    "CpuM": "Răng có mão răng (phục hình hỗn hợp).",
    "Dc":   "Thân răng bị phá hủy nặng.",
    "Di":   "Răng bị mòn mặt nhai hoặc mòn rìa cắn.",
    "P":    "Răng trụ cầu (pontic) trong hàm giả cố định.",
}


def fdi_name(fdi: int) -> str:
    return FDI_NAMES.get(fdi, f"Răng số {fdi}")


def build_micro_entry(row: dict) -> dict:
    fdi = int(row["tooth_id"])
    disease_label = row["disease_label"].strip()
    crop_path = row["crop_image_path"]

    description = DISEASE_DESCRIPTIONS.get(disease_label, row["detailed_report_seed"].strip() + ".")
    instruction = MICRO_INSTRUCTION_TEMPLATE.format(fdi=fdi)
    answer = f"Răng {fdi} ({fdi_name(fdi)}) - {disease_label}: {description}"

    return {
        "messages": [
            {"role": "user",      "content": f"<image>{instruction}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [crop_path],
    }


def build_macro_entry(row: dict) -> dict | None:
    report = row["general_report"].strip()
    if not report:
        return None

    return {
        "messages": [
            {"role": "user",      "content": f"<image>{MACRO_INSTRUCTION_TEMPLATE}"},
            {"role": "assistant", "content": report},
        ],
        "images": [row["image_path"]],
    }


def build_split(
    micro_rows: list[dict],
    macro_rows: list[dict],
    split: str,
    output_dir: Path,
) -> None:
    micro_entries = [
        build_micro_entry(r)
        for r in micro_rows
        if r["split"] == split
    ]

    macro_entries = [
        e
        for r in macro_rows
        if r["split"] == split
        for e in [build_macro_entry(r)]
        if e is not None
    ]

    suffix = "" if split == "train" else f"_{split}"

    micro_path = output_dir / f"micro{suffix}.jsonl"
    write_jsonl(micro_path, micro_entries)
    print(f"[{split}] micro: {len(micro_entries)} entries -> {micro_path.name}")

    macro_path = output_dir / f"macro{suffix}.jsonl"
    write_jsonl(macro_path, macro_entries)
    print(f"[{split}] macro: {len(macro_entries)} entries -> {macro_path.name}")


def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    prepared_dir = (root / args.prepared_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    micro_csv = prepared_dir / "micro" / "micro_labels.csv"
    macro_csv = prepared_dir / "macro" / "macro_labels.csv"

    micro_rows = load_csv(micro_csv)
    macro_rows = load_csv(macro_csv)

    print(f"Loaded {len(micro_rows)} micro rows, {len(macro_rows)} macro rows")

    for split in args.splits:
        build_split(micro_rows, macro_rows, split, output_dir)


if __name__ == "__main__":
    main()
