"""Shared dataset schemas."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A class-labelled bounding box using normalized XYXY coordinates."""

    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """An image and its optional YOLO label file and bounding boxes."""

    image_path: Path
    label_path: Path | None
    boxes: tuple[BoundingBox, ...]
