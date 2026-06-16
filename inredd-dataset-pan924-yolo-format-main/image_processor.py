"""Image processing utilities for dental panorama workflows.

This module is shared by:
- dataset building (offline)
- inference pipeline (online)

Using one crop implementation keeps training and inference behavior consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ContextualCropResult:
    """Container for contextual crop image and requested window coordinates."""

    image: np.ndarray
    req_x1: int
    req_y1: int
    req_x2: int
    req_y2: int


def get_contextual_square_crop_from_array(
    image: np.ndarray,
    bbox: list[float],
    expand_width: float = 1.8,
    expand_height: float = 1.2,
) -> ContextualCropResult | None:
    """Create contextual square crop from an in-memory image array.

    The crop side follows:
    max(width * expand_width, height * expand_height)

    If requested region exceeds image borders, black padding is applied.
    """
    if image is None or image.size == 0:
        return None

    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    side = max(w * float(expand_width), h * float(expand_height))
    side = max(side, 1.0)
    side_int = int(round(side))
    if side_int <= 0:
        return None

    req_x1 = int(round(cx - (side_int / 2.0)))
    req_y1 = int(round(cy - (side_int / 2.0)))
    req_x2 = req_x1 + side_int
    req_y2 = req_y1 + side_int

    img_h, img_w = image.shape[:2]

    pad_left = max(0, -req_x1)
    pad_top = max(0, -req_y1)
    pad_right = max(0, req_x2 - img_w)
    pad_bottom = max(0, req_y2 - img_h)

    work = image
    if pad_left or pad_top or pad_right or pad_bottom:
        work = cv2.copyMakeBorder(
            work,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    x1 = req_x1 + pad_left
    y1 = req_y1 + pad_top
    x2 = req_x2 + pad_left
    y2 = req_y2 + pad_top

    crop = work[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return ContextualCropResult(
        image=crop,
        req_x1=req_x1,
        req_y1=req_y1,
        req_x2=req_x2,
        req_y2=req_y2,
    )


def get_contextual_square_crop(
    image_path: str | Path,
    bbox: list[float],
    expand_width: float = 1.8,
    expand_height: float = 1.2,
) -> np.ndarray:
    """Read image from path and return contextual square crop array.

    Raises:
    - FileNotFoundError: when image path does not exist
    - ValueError: when crop cannot be generated
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    result = get_contextual_square_crop_from_array(
        image=image,
        bbox=bbox,
        expand_width=expand_width,
        expand_height=expand_height,
    )
    if result is None:
        raise ValueError("Cannot build contextual crop from provided bbox")

    return result.image
