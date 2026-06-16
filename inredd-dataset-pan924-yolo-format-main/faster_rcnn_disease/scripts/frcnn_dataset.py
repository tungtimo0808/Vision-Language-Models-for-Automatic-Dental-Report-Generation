"""Dataset utilities for Faster R-CNN COCO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    """Minimal COCO loader for Faster R-CNN training/evaluation."""

    def __init__(self, images_root: Path, annotation_path: Path) -> None:
        self.images_root = images_root
        self.annotation_path = annotation_path

        with annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.categories = coco.get("categories", [])

        self.images = sorted(coco.get("images", []), key=lambda x: int(x["id"]))
        self.image_id_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            image_id = int(ann["image_id"])
            self.image_id_to_anns.setdefault(image_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])

        image_path = self.images_root / file_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)

        anns = self.image_id_to_anns.get(image_id, [])

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))
            area.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target: dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if target["boxes"].numel() == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Detection collate function."""
    images, targets = zip(*batch)
    return list(images), list(targets)
