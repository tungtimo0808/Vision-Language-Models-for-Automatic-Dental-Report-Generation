"""Common utilities for Faster R-CNN training and inference."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file as dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save dictionary to JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve target torch device with safe CUDA fallback."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_faster_rcnn_model(config: dict[str, Any]) -> torch.nn.Module:
    """Build Faster R-CNN model based on configuration."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    architecture = str(model_cfg.get("architecture", "fasterrcnn_mobilenet_v3_large_320_fpn"))
    num_classes = int(model_cfg.get("num_classes", 33))
    pretrained = bool(model_cfg.get("pretrained", True))
    trainable_backbone_layers = int(model_cfg.get("trainable_backbone_layers", 2))

    kwargs: dict[str, Any] = {
        "trainable_backbone_layers": trainable_backbone_layers,
    }

    # Version-safe weights handling.
    if architecture == "fasterrcnn_resnet50_fpn":
        if pretrained:
            kwargs["weights"] = "DEFAULT"
        model = fasterrcnn_resnet50_fpn(**kwargs)
    elif architecture == "fasterrcnn_mobilenet_v3_large_fpn":
        if pretrained:
            kwargs["weights"] = "DEFAULT"
        model = fasterrcnn_mobilenet_v3_large_fpn(**kwargs)
    elif architecture == "fasterrcnn_mobilenet_v3_large_320_fpn":
        if pretrained:
            kwargs["weights"] = "DEFAULT"
        model = fasterrcnn_mobilenet_v3_large_320_fpn(**kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    min_size = int(data_cfg.get("min_size", 640))
    max_size = int(data_cfg.get("max_size", 960))
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    inf_cfg = config.get("inference", {})
    model.roi_heads.score_thresh = float(inf_cfg.get("score_threshold", 0.3))
    model.roi_heads.nms_thresh = float(inf_cfg.get("nms_threshold", 0.5))
    model.roi_heads.detections_per_img = int(inf_cfg.get("max_detections_per_img", 64))

    return model


def coco_id_to_name_map(categories: list[dict[str, Any]]) -> dict[int, str]:
    """Build category_id -> class name mapping from COCO categories.

    For the disease detector the category name is a disease code string
    (e.g. "H", "C", "Te"), so the mapping value is kept as a string.
    """
    mapping: dict[int, str] = {}
    for cat in categories:
        cid = int(cat["id"])
        mapping[cid] = str(cat["name"])
    return mapping
