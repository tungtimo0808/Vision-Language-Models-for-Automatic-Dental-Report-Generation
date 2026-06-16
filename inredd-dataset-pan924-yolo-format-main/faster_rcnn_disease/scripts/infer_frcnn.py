"""Run Faster R-CNN inference on one panorama image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from frcnn_common import build_faster_rcnn_model, coco_id_to_name_map, load_yaml, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Inference for PAN924 Faster R-CNN")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("faster_rcnn_disease/configs/train_config.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("faster_rcnn_disease/outputs/best.pth"),
        help="Checkpoint path",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input panorama image path")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("faster_rcnn_disease/outputs/prediction.json"),
        help="Output predictions json",
    )
    parser.add_argument(
        "--output-visual",
        type=Path,
        default=Path("faster_rcnn_disease/outputs/prediction_vis.jpg"),
        help="Output visualization image",
    )
    parser.add_argument("--score-threshold", type=float, default=None, help="Override score threshold")
    return parser.parse_args()


def load_categories(coco_path: Path) -> list[dict[str, Any]]:
    """Load categories from a COCO annotation file."""
    with coco_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("categories", [])


def main() -> None:
    """Inference entrypoint."""
    args = parse_args()

    root = args.root.resolve()
    config = load_yaml((root / args.config).resolve())

    if args.score_threshold is not None:
        config.setdefault("inference", {})["score_threshold"] = float(args.score_threshold)

    ckpt_path = (root / args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = build_faster_rcnn_model(config)
    model.load_state_dict(ckpt["model"])

    device = resolve_device(str(config.get("train", {}).get("device", "cuda")))
    model.to(device)
    model.eval()

    image_path = (root / args.image).resolve()
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image_pil).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    # Categories are stable in exported COCO files, so loading train categories is sufficient.
    categories = load_categories((root / config.get("data", {}).get("train_annotations")).resolve())
    cat_to_name = coco_id_to_name_map(categories)

    scores = outputs["scores"].detach().cpu().numpy().tolist()
    labels = outputs["labels"].detach().cpu().numpy().tolist()
    boxes = outputs["boxes"].detach().cpu().numpy().tolist()

    pred_rows: list[dict[str, Any]] = []
    for score, label, box in zip(scores, labels, boxes):
        pred_rows.append(
            {
                "disease_label": str(cat_to_name.get(int(label), str(label))),
                "category_id": int(label),
                "score": float(score),
                "bbox_xyxy": [float(v) for v in box],
            }
        )

    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    for row in pred_rows:
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
        score = row["score"]
        disease_label = row["disease_label"]
        text = f"{disease_label}:{score:.2f}"
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image_cv, text, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    output_vis = (root / args.output_visual).resolve()
    output_vis.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_vis), image_cv)

    print(f"Predictions: {len(pred_rows)}")
    print(f"Saved json: {output_json}")
    print(f"Saved visual: {output_vis}")


if __name__ == "__main__":
    main()
