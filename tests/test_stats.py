"""Tests for dataset statistics schemas and numeric utilities."""

import json
import math
from dataclasses import FrozenInstanceError, asdict

import pytest
from infraguard.data.stats import (
    BoundingBoxStatistics,
    ClassStatistics,
    DatasetCounts,
    ImageStatistics,
    MBDD2025Statistics,
    NumericSummary,
    QualityAccounting,
    summarize_numeric,
)


def test_empty_distribution_uses_none_for_unavailable_metrics() -> None:
    """An empty distribution does not invent zero-valued statistics."""
    summary = summarize_numeric([])

    assert summary == NumericSummary(
        count=0,
        min=None,
        max=None,
        mean=None,
        median=None,
        p25=None,
        p75=None,
    )
    assert "NaN" not in json.dumps(asdict(summary), allow_nan=False)


def test_single_value_is_every_distribution_metric() -> None:
    """Every metric equals the sole value in a singleton distribution."""
    summary = summarize_numeric([5])

    assert summary == NumericSummary(
        count=1,
        min=5.0,
        max=5.0,
        mean=5.0,
        median=5.0,
        p25=5.0,
        p75=5.0,
    )


def test_multiple_values_use_linear_interpolated_percentiles() -> None:
    """Percentiles use fractional index (n - 1) * q and linear interpolation."""
    summary = summarize_numeric([1, 2, 3, 4])

    assert summary == NumericSummary(
        count=4,
        min=1.0,
        max=4.0,
        mean=2.5,
        median=2.5,
        p25=1.75,
        p75=3.25,
    )


def test_numeric_summary_is_deterministic() -> None:
    """Repeated and differently ordered equivalent inputs have equal results."""
    values = [4, 1, 3, 2]

    assert summarize_numeric(values) == summarize_numeric(values)
    assert summarize_numeric(values) == summarize_numeric(reversed(values))


def test_finite_inputs_produce_only_finite_metrics() -> None:
    """Large finite values cannot cause unsafe JSON numeric output."""
    summary = summarize_numeric([-1.0e308, 1.0e308, 1.0e308])
    metrics = (
        summary.min,
        summary.max,
        summary.mean,
        summary.median,
        summary.p25,
        summary.p75,
    )

    assert all(metric is not None and math.isfinite(metric) for metric in metrics)
    json.dumps(asdict(summary), allow_nan=False)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_input_is_rejected(value: float) -> None:
    """Non-finite source values raise instead of contaminating the summary."""
    with pytest.raises(ValueError, match="must be finite"):
        summarize_numeric([1.0, value])


def test_numeric_summary_is_immutable() -> None:
    """Callers cannot mutate a summary after it has been calculated."""
    summary = summarize_numeric([1.0])

    with pytest.raises(FrozenInstanceError):
        summary.count = 2  # type: ignore[misc]


def test_dataset_statistics_skeleton_is_immutable_and_json_compatible() -> None:
    """The report skeleton has typed sections suitable for later serialization."""
    empty = summarize_numeric([])
    statistics = MBDD2025Statistics(
        schema_version=1,
        counts=DatasetCounts(
            image_count=0,
            label_file_count=0,
            annotation_count=0,
        ),
        classes=(
            ClassStatistics(
                class_id=0,
                class_name="crack",
                image_count=0,
                instance_count=0,
            ),
        ),
        objects_per_image=empty,
        bounding_boxes=BoundingBoxStatistics(
            width=empty,
            height=empty,
            area=empty,
            aspect_ratio=empty,
            center_x=empty,
            center_y=empty,
        ),
        images=ImageStatistics(
            width=empty,
            height=empty,
            brightness=empty,
            contrast=empty,
        ),
        quality=QualityAccounting(
            excluded_annotation_count=0,
            out_of_bounds_annotation_count=0,
            empty_label_count=0,
        ),
    )

    json.dumps(asdict(statistics), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        statistics.schema_version = 2  # type: ignore[misc]
