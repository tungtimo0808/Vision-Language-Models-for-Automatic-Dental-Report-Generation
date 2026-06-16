"""Merge disease + FDI annotations into a single per-image JSON.

Combines two COCO annotation files into one unified dataset where each tooth
record carries both its disease label and its FDI number.

Inputs:
  annotations/mouth_and_teeth_labels.json
  annotations/teeth_fdi_labels.json

Output:
  annotations/dataset_final_v2.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ANN_DIR = ROOT / "annotations"
MOUTH_JSON = ANN_DIR / "mouth_and_teeth_labels.json"
FDI_JSON = ANN_DIR / "teeth_fdi_labels.json"
OUT_JSON = ANN_DIR / "dataset_final_v2.json"

IOU_THRESHOLD = 0.5
EXCLUDED_SUPERCATEGORY = "Mouth"

UPPER_FDI_LR = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI_LR = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def bbox_from_segmentation(seg: list) -> list[float]:
    pts = np.array(seg[0]).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def compute_iou(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def infer_row(bbox: list[float], img_height: int) -> str:
    cy = bbox[1] + bbox[3] / 2
    return "upper" if cy < img_height / 2 else "lower"


def match_fdi(disease_bboxes: list[list[float]], fdi_anns: list[dict], fdi_cat_map: dict) -> list[int | None]:
    result: list[int | None] = [None] * len(disease_bboxes)
    if not disease_bboxes or not fdi_anns:
        return result

    cost = np.zeros((len(disease_bboxes), len(fdi_anns)), dtype=np.float64)
    for i, db in enumerate(disease_bboxes):
        for j, fa in enumerate(fdi_anns):
            cost[i, j] = 1.0 - compute_iou(db, fa["bbox"])

    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost[r, c]
        if iou >= IOU_THRESHOLD:
            result[r] = fdi_cat_map[fdi_anns[c]["category_id"]]
    return result


def fill_remaining_by_anatomy(
    disease_bboxes: list[list[float]],
    fdi_labels: list[int | None],
    img_height: int,
) -> list[int | None]:
    upper_idx = [i for i, b in enumerate(disease_bboxes) if infer_row(b, img_height) == "upper"]
    lower_idx = [i for i, b in enumerate(disease_bboxes) if infer_row(b, img_height) == "lower"]

    result = list(fdi_labels)
    for arch_idx, fdi_order in ((upper_idx, UPPER_FDI_LR), (lower_idx, LOWER_FDI_LR)):
        arch_idx_sorted = sorted(arch_idx, key=lambda i: disease_bboxes[i][0])
        n = len(arch_idx_sorted)
        if n == 0:
            continue
        for slot, i in enumerate(arch_idx_sorted):
            if result[i] is not None:
                continue
            if n == 1:
                fdi_pos = len(fdi_order) // 2
            else:
                fdi_pos = round(slot * (len(fdi_order) - 1) / (n - 1))
            result[i] = fdi_order[min(fdi_pos, len(fdi_order) - 1)]
    return result


def main() -> None:
    print("Loading annotations...")
    mouth = json.loads(MOUTH_JSON.read_text(encoding="utf-8"))
    fdi_data = json.loads(FDI_JSON.read_text(encoding="utf-8"))

    disease_cat_map = {c["id"]: c["name"] for c in mouth["categories"]}
    disease_super_map = {c["id"]: c["supercategory"] for c in mouth["categories"]}
    fdi_cat_map = {c["id"]: int(c["name"]) for c in fdi_data["categories"]}

    disease_by_img: dict[int, list] = defaultdict(list)
    for ann in mouth["annotations"]:
        if disease_super_map.get(ann["category_id"]) == EXCLUDED_SUPERCATEGORY:
            continue
        disease_by_img[ann["image_id"]].append(ann)

    fdi_anns_by_img: dict[int, list] = defaultdict(list)
    for ann in fdi_data["annotations"]:
        fdi_anns_by_img[ann["image_id"]].append(ann)

    fdi_name_to_id = {img["file_name"]: img["id"] for img in fdi_data["images"]}

    output_images = []
    total_teeth = 0

    for mouth_img in tqdm(mouth["images"], desc="Merging images"):
        fname = mouth_img["file_name"]
        img_id = mouth_img["id"]
        disease_anns = disease_by_img.get(img_id, [])

        fdi_img_id = fdi_name_to_id.get(fname)
        fdi_anns = fdi_anns_by_img.get(fdi_img_id, []) if fdi_img_id is not None else []

        disease_bboxes = [bbox_from_segmentation(a["segmentation"]) for a in disease_anns]
        labels = match_fdi(disease_bboxes, fdi_anns, fdi_cat_map)
        labels = fill_remaining_by_anatomy(disease_bboxes, labels, mouth_img["height"])

        teeth_out = []
        for ann, bbox, fdi_label in zip(disease_anns, disease_bboxes, labels):
            teeth_out.append({
                "disease_annotation_id": ann["id"],
                "disease_label": disease_cat_map[ann["category_id"]],
                "fdi_label": fdi_label,
                "bbox_xywh": [round(v, 1) for v in bbox],
                "segmentation": ann["segmentation"],
                "row_inferred": infer_row(bbox, mouth_img["height"]),
            })

        total_teeth += len(teeth_out)

        output_images.append({
            "file_name": fname,
            "image_id": img_id,
            "width": mouth_img["width"],
            "height": mouth_img["height"],
            "teeth": teeth_out,
        })

    output = {
        "meta": {
            "name": "inredd_pan924_dataset_final_v2",
            "description": "924 panoramic X-rays with disease and FDI labels per tooth.",
            "images": len(output_images),
            "total_teeth_rows": total_teeth,
        },
        "images": output_images,
    }

    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Output: {OUT_JSON}")
    print(f"  images      : {len(output_images)}")
    print(f"  total_teeth : {total_teeth}")


if __name__ == "__main__":
    main()
