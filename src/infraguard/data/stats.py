"""Typed schemas and numeric utilities for dataset statistics."""

import math
from collections.abc import Iterable
from dataclasses import dataclass


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
    """Headline file, image, and annotation counts."""

    image_count: int
    label_file_count: int
    annotation_count: int


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
    """Counts that make exclusions and source-quality findings explicit."""

    excluded_annotation_count: int
    out_of_bounds_annotation_count: int
    empty_label_count: int


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
