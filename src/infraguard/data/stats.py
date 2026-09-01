"""Typed schemas and numeric utilities for dataset statistics."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from infraguard.data.mbdd import (
    AnnotationParseError,
    DatasetLayoutError,
    YoloAnnotationRow,
    parse_yolo_line,
)
from infraguard.data.validator import OOB_TOLERANCE, ValidationCode

_IMAGE_DIRECTORY_NAME = "JPEGImages"
_LABEL_DIRECTORY_NAME = "Labels"
_IMAGE_SUFFIXES = frozenset({".jpg"})


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """An immutable summary of a finite numeric distribution.

    Metrics are ``None`` when ``count`` is zero. This preserves the distinction
    between an unavailable statistic and a statistic whose value is zero.
    """

    count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    p25: float | None
    p75: float | None


@dataclass(frozen=True, slots=True)
class DatasetCounts:
    """Headline file, image, and usable-annotation counts."""

    image_count: int
    label_file_count: int
    usable_annotation_count: int


@dataclass(frozen=True, slots=True)
class ClassStatistics:
    """Image and annotation counts for one configured class."""

    class_id: int
    class_name: str
    image_count: int
    instance_count: int


@dataclass(frozen=True, slots=True)
class BoundingBoxStatistics:
    """Numeric summaries derived from included bounding boxes."""

    width: NumericSummary
    height: NumericSummary
    area: NumericSummary
    aspect_ratio: NumericSummary
    center_x: NumericSummary
    center_y: NumericSummary


@dataclass(frozen=True, slots=True)
class ImageStatistics:
    """Numeric summaries derived from readable images."""

    width: NumericSummary
    height: NumericSummary
    brightness: NumericSummary
    contrast: NumericSummary


@dataclass(frozen=True, slots=True)
class QualityAccounting:
    """Counts that make exclusions and source-quality findings explicit.

    Total rows counts non-blank YOLO rows. Each excluded row contributes to
    exactly one root-cause field; OOB is tracked separately as a usable subset.
    """

    total_annotation_row_count: int
    usable_annotation_count: int
    excluded_annotation_count: int
    malformed_annotation_count: int
    invalid_class_annotation_count: int
    invalid_coordinate_annotation_count: int
    negative_size_annotation_count: int
    zero_area_annotation_count: int
    out_of_bounds_annotation_count: int
    empty_label_count: int


@dataclass(frozen=True, slots=True)
class MBDD2025AnnotationStatistics:
    """Annotation-derived MBDD2025 statistics produced by Task 5.2."""

    counts: DatasetCounts
    classes: tuple[ClassStatistics, ...]
    objects_per_image: NumericSummary
    bounding_boxes: BoundingBoxStatistics
    quality: QualityAccounting


@dataclass(frozen=True, slots=True)
class MBDD2025Statistics:
    """Typed top-level skeleton for a future MBDD2025 statistics report.

    ``classes`` is expected to use deterministic class-ID order when populated.
    Task 5.1 defines this contract but does not calculate dataset statistics.
    """

    schema_version: int
    counts: DatasetCounts
    classes: tuple[ClassStatistics, ...]
    objects_per_image: NumericSummary
    bounding_boxes: BoundingBoxStatistics
    images: ImageStatistics
    quality: QualityAccounting


class DatasetStatisticsError(ValueError):
    """Raised when structural anomalies prevent meaningful statistics."""


def compute_mbdd2025_annotation_statistics(
    dataset_root: Path,
    *,
    class_names: Mapping[int, str],
) -> MBDD2025AnnotationStatistics:
    """Compute deterministic file, class, object, box, and quality statistics.

    A usable annotation parses as YOLO, belongs to ``class_names``, has finite
    coordinates, has centers within ``[0, 1]``, has sizes no greater than one,
    and has strictly positive width and height. Materially out-of-bounds XYXY
    boxes remain usable and retain their original YOLO values in all aggregates.

    Missing or orphan labels raise ``DatasetStatisticsError`` because their
    object count is unknown; they are never interpreted as empty labels.
    """
    image_paths, label_paths, image_label_pairs = _discover_dataset(dataset_root)
    ordered_classes = tuple(sorted(class_names.items()))
    valid_class_ids = frozenset(class_names)
    class_image_counts = {class_id: 0 for class_id, _ in ordered_classes}
    class_instance_counts = {class_id: 0 for class_id, _ in ordered_classes}

    objects_per_image: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []

    total_annotation_row_count = 0
    usable_annotation_count = 0
    malformed_annotation_count = 0
    invalid_class_annotation_count = 0
    invalid_coordinate_annotation_count = 0
    negative_size_annotation_count = 0
    zero_area_annotation_count = 0
    out_of_bounds_annotation_count = 0
    empty_label_count = 0

    for _, label_path in image_label_pairs:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise DatasetStatisticsError(
                f"Could not read YOLO label file: {label_path}"
            ) from error

        annotation_lines = tuple(
            (line_number, line)
            for line_number, line in enumerate(lines, start=1)
            if line.strip()
        )
        if not annotation_lines:
            empty_label_count += 1

        image_usable_count = 0
        image_class_ids: set[int] = set()
        for line_number, line in annotation_lines:
            total_annotation_row_count += 1
            try:
                row = parse_yolo_line(
                    line,
                    label_path=label_path,
                    line_number=line_number,
                )
            except AnnotationParseError:
                malformed_annotation_count += 1
                continue

            exclusion_code = _annotation_exclusion_code(row, valid_class_ids)
            if exclusion_code is not None:
                if exclusion_code is ValidationCode.INVALID_CLASS_ID:
                    invalid_class_annotation_count += 1
                elif exclusion_code is ValidationCode.INVALID_COORDINATE:
                    invalid_coordinate_annotation_count += 1
                elif exclusion_code is ValidationCode.NON_POSITIVE_BOX_SIZE:
                    negative_size_annotation_count += 1
                elif exclusion_code is ValidationCode.ZERO_AREA_BOX:
                    zero_area_annotation_count += 1
                continue

            usable_annotation_count += 1
            image_usable_count += 1
            image_class_ids.add(row.class_id)
            class_instance_counts[row.class_id] += 1
            widths.append(row.width)
            heights.append(row.height)
            areas.append(row.width * row.height)
            aspect_ratios.append(row.width / row.height)
            centers_x.append(row.x_center)
            centers_y.append(row.y_center)
            if _is_materially_out_of_bounds(row):
                out_of_bounds_annotation_count += 1

        objects_per_image.append(image_usable_count)
        for class_id in image_class_ids:
            class_image_counts[class_id] += 1

    classes = tuple(
        ClassStatistics(
            class_id=class_id,
            class_name=class_name,
            image_count=class_image_counts[class_id],
            instance_count=class_instance_counts[class_id],
        )
        for class_id, class_name in ordered_classes
    )
    excluded_annotation_count = (
        malformed_annotation_count
        + invalid_class_annotation_count
        + invalid_coordinate_annotation_count
        + negative_size_annotation_count
        + zero_area_annotation_count
    )
    quality = QualityAccounting(
        total_annotation_row_count=total_annotation_row_count,
        usable_annotation_count=usable_annotation_count,
        excluded_annotation_count=excluded_annotation_count,
        malformed_annotation_count=malformed_annotation_count,
        invalid_class_annotation_count=invalid_class_annotation_count,
        invalid_coordinate_annotation_count=invalid_coordinate_annotation_count,
        negative_size_annotation_count=negative_size_annotation_count,
        zero_area_annotation_count=zero_area_annotation_count,
        out_of_bounds_annotation_count=out_of_bounds_annotation_count,
        empty_label_count=empty_label_count,
    )
    return MBDD2025AnnotationStatistics(
        counts=DatasetCounts(
            image_count=len(image_paths),
            label_file_count=len(label_paths),
            usable_annotation_count=usable_annotation_count,
        ),
        classes=classes,
        objects_per_image=summarize_numeric(objects_per_image),
        bounding_boxes=BoundingBoxStatistics(
            width=summarize_numeric(widths),
            height=summarize_numeric(heights),
            area=summarize_numeric(areas),
            aspect_ratio=summarize_numeric(aspect_ratios),
            center_x=summarize_numeric(centers_x),
            center_y=summarize_numeric(centers_y),
        ),
        quality=quality,
    )


def _discover_dataset(
    dataset_root: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[tuple[Path, Path], ...]]:
    images_directory = dataset_root / _IMAGE_DIRECTORY_NAME
    labels_directory = dataset_root / _LABEL_DIRECTORY_NAME
    if not images_directory.is_dir():
        raise DatasetLayoutError(
            "MBDD2025 image directory does not exist or is not a directory: "
            f"{images_directory}"
        )
    if not labels_directory.is_dir():
        raise DatasetLayoutError(
            "MBDD2025 label directory does not exist or is not a directory: "
            f"{labels_directory}"
        )

    image_paths = tuple(
        sorted(
            (
                path
                for path in images_directory.iterdir()
                if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    label_paths = tuple(
        sorted(
            (
                path
                for path in labels_directory.iterdir()
                if path.is_file() and path.suffix == ".txt"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    label_paths_by_name = {path.name: path for path in label_paths}
    image_stems = {path.stem for path in image_paths}
    missing_label_names = tuple(
        f"{image_path.stem}.txt"
        for image_path in image_paths
        if f"{image_path.stem}.txt" not in label_paths_by_name
    )
    orphan_label_names = tuple(
        label_path.name
        for label_path in label_paths
        if label_path.stem not in image_stems
    )
    if missing_label_names or orphan_label_names:
        details: list[str] = []
        if missing_label_names:
            details.append(f"missing labels: {', '.join(missing_label_names)}")
        if orphan_label_names:
            details.append(f"orphan labels: {', '.join(orphan_label_names)}")
        raise DatasetStatisticsError("; ".join(details))

    image_label_pairs = tuple(
        (image_path, label_paths_by_name[f"{image_path.stem}.txt"])
        for image_path in image_paths
    )
    return image_paths, label_paths, image_label_pairs


def _annotation_exclusion_code(
    row: YoloAnnotationRow,
    valid_class_ids: frozenset[int],
) -> ValidationCode | None:
    if row.class_id not in valid_class_ids:
        return ValidationCode.INVALID_CLASS_ID

    coordinates = (row.x_center, row.y_center, row.width, row.height)
    if not all(math.isfinite(value) for value in coordinates):
        return ValidationCode.INVALID_COORDINATE
    if (
        not 0 <= row.x_center <= 1
        or not 0 <= row.y_center <= 1
        or row.width > 1
        or row.height > 1
    ):
        return ValidationCode.INVALID_COORDINATE
    if row.width < 0 or row.height < 0:
        return ValidationCode.NON_POSITIVE_BOX_SIZE
    if row.width == 0 or row.height == 0:
        return ValidationCode.ZERO_AREA_BOX
    return None


def _is_materially_out_of_bounds(row: YoloAnnotationRow) -> bool:
    box = row.to_bounding_box()
    return (
        box.xmin < -OOB_TOLERANCE
        or box.ymin < -OOB_TOLERANCE
        or box.xmax > 1.0 + OOB_TOLERANCE
        or box.ymax > 1.0 + OOB_TOLERANCE
    )


def summarize_numeric(values: Iterable[float]) -> NumericSummary:
    """Summarize finite values using linear-interpolated percentiles.

    For quantile ``q``, interpolation uses the zero-based fractional index
    ``(n - 1) * q`` in sorted values. Non-finite inputs raise ``ValueError``;
    an empty input returns a count of zero and ``None`` for every metric.
    """
    finite_values: list[float] = []
    for value in values:
        try:
            numeric_value = float(value)
        except OverflowError as error:
            raise ValueError("numeric summary values must be finite") from error
        if not math.isfinite(numeric_value):
            raise ValueError("numeric summary values must be finite")
        finite_values.append(numeric_value)

    if not finite_values:
        return NumericSummary(
            count=0,
            min=None,
            max=None,
            mean=None,
            median=None,
            p25=None,
            p75=None,
        )

    sorted_values = sorted(finite_values)
    summary = NumericSummary(
        count=len(sorted_values),
        min=sorted_values[0],
        max=sorted_values[-1],
        mean=_finite_mean(sorted_values),
        median=_linear_percentile(sorted_values, 0.5),
        p25=_linear_percentile(sorted_values, 0.25),
        p75=_linear_percentile(sorted_values, 0.75),
    )
    if not all(
        math.isfinite(metric)
        for metric in (
            summary.min,
            summary.max,
            summary.mean,
            summary.median,
            summary.p25,
            summary.p75,
        )
        if metric is not None
    ):
        raise ArithmeticError("numeric summary produced a non-finite metric")
    return summary


def _finite_mean(sorted_values: list[float]) -> float:
    scale = max(abs(sorted_values[0]), abs(sorted_values[-1]))
    if scale == 0.0:
        return 0.0
    normalized_mean = math.fsum(value / scale for value in sorted_values) / len(
        sorted_values
    )
    return normalized_mean * scale


def _linear_percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    fraction = position - lower_index
    scale = max(abs(lower), abs(upper))
    if scale == 0.0:
        return 0.0
    normalized = (1.0 - fraction) * (lower / scale) + fraction * (upper / scale)
    return normalized * scale
