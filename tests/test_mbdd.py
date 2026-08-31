"""Tests for the MBDD2025 dataset integration."""

from pathlib import Path

import infraguard
import pytest
from infraguard.data import mbdd
from infraguard.data.mbdd import (
    AnnotationParseError,
    DatasetLayoutError,
    load_mbdd2025,
    parse_yolo_label,
)
from infraguard.data.schemas import BoundingBox, ImageRecord


def test_infraguard_package_imports() -> None:
    """The top-level package is importable."""
    assert infraguard.__name__ == "infraguard"


def test_mbdd_module_imports() -> None:
    """The MBDD2025 integration module is importable."""
    assert mbdd.__name__ == "infraguard.data.mbdd"


def test_bounding_box_stores_normalized_xyxy_coordinates() -> None:
    """A bounding box stores its class and normalized XYXY coordinates."""
    box = BoundingBox(
        class_id=2,
        xmin=0.1,
        ymin=0.2,
        xmax=0.7,
        ymax=0.8,
    )

    assert box.class_id == 2
    assert box.xmin == 0.1
    assert box.ymin == 0.2
    assert box.xmax == 0.7
    assert box.ymax == 0.8


def test_image_record_preserves_paths() -> None:
    """An image record keeps pathlib paths without converting them."""
    image_path = Path("JPEGImages/example.jpg")
    label_path = Path("Labels/example.txt")

    record = ImageRecord(
        image_path=image_path,
        label_path=label_path,
        boxes=(),
    )

    assert record.image_path is image_path
    assert record.label_path is label_path


def test_missing_label_is_distinct_from_empty_label() -> None:
    """A missing label and an existing empty label remain distinguishable."""
    image_path = Path("JPEGImages/example.jpg")
    label_path = Path("Labels/example.txt")

    missing_label = ImageRecord(
        image_path=image_path,
        label_path=None,
        boxes=(),
    )
    empty_label = ImageRecord(
        image_path=image_path,
        label_path=label_path,
        boxes=(),
    )

    assert missing_label.label_path is None
    assert empty_label.label_path == label_path
    assert missing_label.boxes == empty_label.boxes == ()


def test_parse_yolo_label_converts_single_box_to_normalized_xyxy(
    tmp_path: Path,
) -> None:
    """A YOLO XYWH row is converted to the internal XYXY representation."""
    label_path = tmp_path / "single.txt"
    label_path.write_text("0 0.5 0.4 0.2 0.1\n", encoding="utf-8")

    boxes = parse_yolo_label(label_path)

    assert len(boxes) == 1
    assert boxes[0].class_id == 0
    assert boxes[0].xmin == pytest.approx(0.4)
    assert boxes[0].ymin == pytest.approx(0.35)
    assert boxes[0].xmax == pytest.approx(0.6)
    assert boxes[0].ymax == pytest.approx(0.45)


def test_parse_yolo_label_returns_multiple_boxes(tmp_path: Path) -> None:
    """Each non-empty YOLO row produces one bounding box in source order."""
    label_path = tmp_path / "multiple.txt"
    label_path.write_text(
        "0 0.5 0.5 0.2 0.4\n2 0.25 0.75 0.1 0.2\n",
        encoding="utf-8",
    )

    boxes = parse_yolo_label(label_path)

    assert boxes == (
        BoundingBox(class_id=0, xmin=0.4, ymin=0.3, xmax=0.6, ymax=0.7),
        BoundingBox(class_id=2, xmin=0.2, ymin=0.65, xmax=0.3, ymax=0.85),
    )


def test_parse_yolo_label_accepts_normal_whitespace(tmp_path: Path) -> None:
    """Spaces and tabs may separate YOLO fields."""
    label_path = tmp_path / "whitespace.txt"
    label_path.write_text("  1\t0.5  0.5\t0.2  0.2  \n", encoding="utf-8")

    boxes = parse_yolo_label(label_path)

    assert boxes == (BoundingBox(class_id=1, xmin=0.4, ymin=0.4, xmax=0.6, ymax=0.6),)


def test_parse_yolo_label_ignores_blank_lines(tmp_path: Path) -> None:
    """Blank and whitespace-only rows do not produce boxes."""
    label_path = tmp_path / "blank-lines.txt"
    label_path.write_text("\n  \t\n3 0.5 0.5 0.4 0.2\n\n", encoding="utf-8")

    boxes = parse_yolo_label(label_path)

    assert len(boxes) == 1
    assert boxes[0].class_id == 3


def test_parse_yolo_label_returns_empty_tuple_for_empty_file(tmp_path: Path) -> None:
    """An empty label file represents an image without annotated boxes."""
    label_path = tmp_path / "empty.txt"
    label_path.write_text("", encoding="utf-8")

    assert parse_yolo_label(label_path) == ()


def test_parse_yolo_label_rejects_malformed_field_count(tmp_path: Path) -> None:
    """A non-empty row must contain exactly five fields."""
    label_path = tmp_path / "field-count.txt"
    label_path.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")

    with pytest.raises(AnnotationParseError, match=r"field-count\.txt:1: expected 5"):
        parse_yolo_label(label_path)


def test_parse_yolo_label_rejects_non_integer_class_id(tmp_path: Path) -> None:
    """The first field must use integer syntax."""
    label_path = tmp_path / "class-id.txt"
    label_path.write_text("1.5 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(AnnotationParseError, match=r"class-id\.txt:1: class_id"):
        parse_yolo_label(label_path)


def test_parse_yolo_label_rejects_invalid_float(tmp_path: Path) -> None:
    """Each coordinate field must use float-compatible syntax."""
    label_path = tmp_path / "coordinate.txt"
    label_path.write_text("0 0.5 invalid 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(AnnotationParseError, match=r"coordinate\.txt:1: y_center"):
        parse_yolo_label(label_path)


def test_load_mbdd2025_discovers_images_in_deterministic_order(
    tmp_path: Path,
) -> None:
    """Only direct JPEG children are returned in stable filename order."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    images_directory.mkdir(parents=True)
    (images_directory / "b.jpg").touch()
    (images_directory / "a.jpg").touch()
    (images_directory / "c.png").touch()
    (images_directory / "notes.txt").touch()
    nested_directory = images_directory / "nested"
    nested_directory.mkdir()
    (nested_directory / "nested.jpg").touch()
    (dataset_root / "outside.jpg").touch()

    records = load_mbdd2025(dataset_root)

    assert tuple(record.image_path.name for record in records) == ("a.jpg", "b.jpg")


def test_load_mbdd2025_matches_and_parses_label_by_image_stem(
    tmp_path: Path,
) -> None:
    """An existing stem-matched label is retained and parsed."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    image_path = images_directory / "a.jpg"
    label_path = labels_directory / "a.txt"
    image_path.touch()
    label_path.write_text("4 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    records = load_mbdd2025(dataset_root)

    assert records == (
        ImageRecord(
            image_path=image_path,
            label_path=label_path,
            boxes=(
                BoundingBox(
                    class_id=4,
                    xmin=0.4,
                    ymin=0.3,
                    xmax=0.6,
                    ymax=0.7,
                ),
            ),
        ),
    )


def test_load_mbdd2025_represents_missing_label_explicitly(
    tmp_path: Path,
) -> None:
    """An image without a label has no invented label path or boxes."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    images_directory.mkdir(parents=True)
    image_path = images_directory / "missing.jpg"
    image_path.touch()

    records = load_mbdd2025(dataset_root)

    assert records == (ImageRecord(image_path=image_path, label_path=None, boxes=()),)


def test_load_mbdd2025_distinguishes_empty_label_from_missing_label(
    tmp_path: Path,
) -> None:
    """An existing empty label keeps its real path and has no boxes."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    image_path = images_directory / "empty.jpg"
    label_path = labels_directory / "empty.txt"
    image_path.touch()
    label_path.touch()

    records = load_mbdd2025(dataset_root)

    assert records == (
        ImageRecord(image_path=image_path, label_path=label_path, boxes=()),
    )


def test_load_mbdd2025_requires_image_directory(tmp_path: Path) -> None:
    """A missing JPEGImages directory produces a clear layout error."""
    dataset_root = tmp_path / "MBDD2025"
    dataset_root.mkdir()

    with pytest.raises(DatasetLayoutError, match=r"JPEGImages"):
        load_mbdd2025(dataset_root)
