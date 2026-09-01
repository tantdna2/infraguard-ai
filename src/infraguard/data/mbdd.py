"""MBDD2025 dataset integration."""

from dataclasses import dataclass
from pathlib import Path

from infraguard.data.schemas import BoundingBox, ImageRecord

_IMAGE_SUFFIXES = frozenset({".jpg"})


class AnnotationParseError(ValueError):
    """Raised when a YOLO annotation row has invalid syntax."""


class DatasetLayoutError(FileNotFoundError):
    """Raised when a required MBDD2025 dataset directory is unavailable."""


@dataclass(frozen=True, slots=True)
class YoloAnnotationRow:
    """A parsed YOLO annotation row in normalized center-width-height form."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_bounding_box(self) -> BoundingBox:
        """Convert the YOLO row to the normalized internal XYXY representation."""
        return BoundingBox(
            class_id=self.class_id,
            xmin=self.x_center - self.width / 2,
            ymin=self.y_center - self.height / 2,
            xmax=self.x_center + self.width / 2,
            ymax=self.y_center + self.height / 2,
        )


def parse_yolo_line(
    line: str,
    *,
    label_path: Path,
    line_number: int,
) -> YoloAnnotationRow:
    """Parse one YOLO annotation row without applying semantic validation."""
    fields = line.split()
    if len(fields) != 5:
        raise AnnotationParseError(
            f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}"
        )

    class_id_text, x_center_text, y_center_text, width_text, height_text = fields

    try:
        class_id = int(class_id_text)
    except ValueError as error:
        raise AnnotationParseError(
            f"{label_path}:{line_number}: "
            f"class_id must be an integer, got {class_id_text!r}"
        ) from error

    coordinate_names = ("x_center", "y_center", "width", "height")
    coordinate_texts = (
        x_center_text,
        y_center_text,
        width_text,
        height_text,
    )
    coordinates: list[float] = []
    for name, value in zip(coordinate_names, coordinate_texts, strict=True):
        try:
            coordinates.append(float(value))
        except ValueError as error:
            raise AnnotationParseError(
                f"{label_path}:{line_number}: {name} must be a float, got {value!r}"
            ) from error

    x_center, y_center, width, height = coordinates
    return YoloAnnotationRow(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def parse_yolo_label(label_path: Path) -> tuple[BoundingBox, ...]:
    """Parse a YOLO TXT label file into normalized XYXY bounding boxes."""
    boxes: list[BoundingBox] = []
    with label_path.open(encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            if not line.strip():
                continue
            row = parse_yolo_line(
                line,
                label_path=label_path,
                line_number=line_number,
            )
            boxes.append(row.to_bounding_box())

    return tuple(boxes)


def load_mbdd2025(dataset_root: Path) -> tuple[ImageRecord, ...]:
    """Load deterministic image records from an extracted MBDD2025 dataset."""
    images_directory = dataset_root / "JPEGImages"
    if not images_directory.is_dir():
        raise DatasetLayoutError(
            "MBDD2025 image directory does not exist or is not a directory: "
            f"{images_directory}"
        )

    labels_directory = dataset_root / "Labels"
    if not labels_directory.is_dir():
        raise DatasetLayoutError(
            "MBDD2025 label directory does not exist or is not a directory: "
            f"{labels_directory}"
        )

    image_paths = sorted(
        (
            path
            for path in images_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )

    records: list[ImageRecord] = []
    for image_path in image_paths:
        candidate_label_path = labels_directory / f"{image_path.stem}.txt"
        if candidate_label_path.is_file():
            label_path: Path | None = candidate_label_path
            boxes = parse_yolo_label(candidate_label_path)
        else:
            label_path = None
            boxes = ()

        records.append(
            ImageRecord(
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
            )
        )

    return tuple(records)
