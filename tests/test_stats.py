"""Tests for dataset statistics schemas and numeric utilities."""

import json
import math
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
from PIL import Image
from infraguard.data.mbdd import YoloAnnotationRow
from infraguard.data.stats import (
    BoundingBoxStatistics,
    ClassStatistics,
    DatasetCounts,
    DatasetStatisticsError,
    ImageResolutionCount,
    ImageStatistics,
    MBDD2025AnnotationStatistics,
    MBDD2025ImageStatistics,
    MBDD2025Statistics,
    NumericSummary,
    QualityAccounting,
    compute_mbdd2025_annotation_statistics,
    compute_mbdd2025_image_statistics,
    compute_mbdd2025_statistics,
    summarize_numeric,
)

_CLASS_NAMES = {0: "crack", 1: "leakage", 2: "bulge"}


def _write_synthetic_dataset(
    tmp_path: Path,
    labels: dict[str, str],
) -> Path:
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    for image_name, label_text in labels.items():
        (images_directory / f"{image_name}.jpg").touch()
        (labels_directory / f"{image_name}.txt").write_text(
            label_text,
            encoding="utf-8",
        )
    return dataset_root


def _assert_single_value_summary(
    summary: NumericSummary,
    expected: float,
) -> None:
    assert summary.count == 1
    for metric in (
        summary.min,
        summary.max,
        summary.mean,
        summary.median,
        summary.p25,
        summary.p75,
    ):
        assert metric == pytest.approx(expected)


def _write_constant_jpeg(
    dataset_root: Path,
    image_name: str,
    *,
    size: tuple[int, int],
    intensity: int,
    mode: str = "L",
) -> Path:
    images_directory = dataset_root / "JPEGImages"
    images_directory.mkdir(parents=True, exist_ok=True)
    color: int | tuple[int, int, int]
    if mode == "RGB":
        color = (intensity, intensity, intensity)
    else:
        color = intensity
    image_path = images_directory / f"{image_name}.jpg"
    with Image.new(mode, size, color=color) as image:
        image.save(image_path, format="JPEG", quality=100, subsampling=0)
    return image_path


def _write_two_level_jpeg(
    dataset_root: Path,
    image_name: str,
    *,
    size: tuple[int, int],
) -> Path:
    width, height = size
    assert width % 16 == 0
    images_directory = dataset_root / "JPEGImages"
    images_directory.mkdir(parents=True, exist_ok=True)
    image_path = images_directory / f"{image_name}.jpg"
    with Image.new("L", size, color=0) as image:
        image.paste(255, (width // 2, 0, width, height))
        image.save(image_path, format="JPEG", quality=100)
    return image_path


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
            usable_annotation_count=0,
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
        resolution_counts=(),
        quality=QualityAccounting(
            total_annotation_row_count=0,
            usable_annotation_count=0,
            excluded_annotation_count=0,
            malformed_annotation_count=0,
            invalid_class_annotation_count=0,
            invalid_coordinate_annotation_count=0,
            negative_size_annotation_count=0,
            zero_area_annotation_count=0,
            out_of_bounds_annotation_count=0,
            empty_label_count=0,
        ),
    )

    json.dumps(asdict(statistics), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        statistics.schema_version = 2  # type: ignore[misc]


def test_class_counts_keep_instances_images_and_zero_classes_distinct(
    tmp_path: Path,
) -> None:
    """Class counts preserve source instances while counting each image once."""
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {
            "multiple": (
                "0 0.5 0.5 0.2 0.2\n"
                "0 0.5 0.5 0.2 0.2\n"
                "0 0.5 0.5 0.2 0.2\n"
                "1 0.4 0.4 0.1 0.1\n"
            ),
            "empty": "",
        },
    )

    result = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    assert isinstance(result, MBDD2025AnnotationStatistics)
    assert result.counts == DatasetCounts(
        image_count=2,
        label_file_count=2,
        usable_annotation_count=4,
    )
    assert result.classes == (
        ClassStatistics(0, "crack", image_count=1, instance_count=3),
        ClassStatistics(1, "leakage", image_count=1, instance_count=1),
        ClassStatistics(2, "bulge", image_count=0, instance_count=0),
    )
    assert result.objects_per_image == NumericSummary(
        count=2,
        min=0.0,
        max=4.0,
        mean=2.0,
        median=2.0,
        p25=1.0,
        p75=3.0,
    )
    assert result.quality.empty_label_count == 1
    assert result.quality.total_annotation_row_count == 4
    assert result.quality.usable_annotation_count == 4
    assert result.quality.excluded_annotation_count == 0


def test_bounding_box_statistics_use_original_yolo_components(tmp_path: Path) -> None:
    """Box metrics are calculated directly from known normalized YOLO values."""
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {"box": "0 0.3 0.6 0.4 0.2\n"},
    )

    result = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    _assert_single_value_summary(result.bounding_boxes.width, 0.4)
    _assert_single_value_summary(result.bounding_boxes.height, 0.2)
    _assert_single_value_summary(result.bounding_boxes.area, 0.08)
    _assert_single_value_summary(result.bounding_boxes.aspect_ratio, 2.0)
    _assert_single_value_summary(result.bounding_boxes.center_x, 0.3)
    _assert_single_value_summary(result.bounding_boxes.center_y, 0.6)


def test_unusable_rows_have_one_exclusion_root_cause_each(tmp_path: Path) -> None:
    """Malformed and semantically unusable rows are excluded and accounted once."""
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {
            "quality": (
                "0 0.5 0.5 0.2\n"
                "99 0.5 0.5 0.2 0.2\n"
                "0 nan 0.5 0.2 0.2\n"
                "0 1.1 0.5 0.2 0.2\n"
                "0 0.5 0.5 -0.2 0.2\n"
                "0 0.5 0.5 0 0.2\n"
                "0 0.5 0.5 0.2 0.2\n"
            )
        },
    )

    result = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    assert result.quality == QualityAccounting(
        total_annotation_row_count=7,
        usable_annotation_count=1,
        excluded_annotation_count=6,
        malformed_annotation_count=1,
        invalid_class_annotation_count=1,
        invalid_coordinate_annotation_count=2,
        negative_size_annotation_count=1,
        zero_area_annotation_count=1,
        out_of_bounds_annotation_count=0,
        empty_label_count=0,
    )
    assert result.counts.usable_annotation_count == 1
    assert result.objects_per_image == summarize_numeric([1])
    assert result.bounding_boxes.width == summarize_numeric([0.2])


def test_positive_source_size_collapsing_in_xyxy_is_zero_area(tmp_path: Path) -> None:
    """A positive subnormal width is excluded when reconstructed XYXY collapses."""
    subnormal_width = float("5e-324")
    row = YoloAnnotationRow(
        class_id=0,
        x_center=0.5,
        y_center=0.5,
        width=subnormal_width,
        height=0.2,
    )
    box = row.to_bounding_box()
    assert row.width > 0
    assert row.height > 0
    assert box.xmax <= box.xmin

    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {"collapsed": f"0 0.5 0.5 {subnormal_width!r} 0.2\n"},
    )

    result = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    assert result.quality.zero_area_annotation_count == 1
    assert result.quality.excluded_annotation_count == 1
    assert result.quality.usable_annotation_count == 0
    assert result.quality.out_of_bounds_annotation_count == 0
    assert result.counts.usable_annotation_count == 0
    assert result.objects_per_image == summarize_numeric([0])
    assert all(class_stats.instance_count == 0 for class_stats in result.classes)
    assert all(class_stats.image_count == 0 for class_stats in result.classes)
    empty_summary = summarize_numeric([])
    assert result.bounding_boxes.width == empty_summary
    assert result.bounding_boxes.height == empty_summary
    assert result.bounding_boxes.area == empty_summary
    assert result.bounding_boxes.aspect_ratio == empty_summary
    assert result.bounding_boxes.center_x == empty_summary
    assert result.bounding_boxes.center_y == empty_summary


def test_materially_oob_annotation_remains_usable_and_unclamped(tmp_path: Path) -> None:
    """A source-valid OOB row contributes its unmodified YOLO values."""
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {"oob": "0 0.1 0.5 0.4 0.2\n"},
    )

    result = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    assert result.counts.usable_annotation_count == 1
    assert result.quality.usable_annotation_count == 1
    assert result.quality.excluded_annotation_count == 0
    assert result.quality.out_of_bounds_annotation_count == 1
    _assert_single_value_summary(result.bounding_boxes.width, 0.4)
    _assert_single_value_summary(result.bounding_boxes.height, 0.2)
    _assert_single_value_summary(result.bounding_boxes.area, 0.08)
    _assert_single_value_summary(result.bounding_boxes.aspect_ratio, 2.0)
    _assert_single_value_summary(result.bounding_boxes.center_x, 0.1)
    _assert_single_value_summary(result.bounding_boxes.center_y, 0.5)


def test_exact_duplicate_source_rows_are_not_deduplicated(tmp_path: Path) -> None:
    """Identical source rows each remain an annotation instance."""
    duplicate_row = "0 0.5 0.5 0.2 0.2\n"
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {"duplicates": duplicate_row * 2},
    )

    result = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    assert result.counts.usable_annotation_count == 2
    assert result.classes[0].image_count == 1
    assert result.classes[0].instance_count == 2
    assert result.bounding_boxes.width.count == 2


def test_discovery_and_results_are_deterministic(tmp_path: Path) -> None:
    """Filesystem creation order does not affect discovery or result ordering."""
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {
            "z-last": "1 0.5 0.5 0.2 0.2\n",
            "a-first": "0 0.5 0.5 0.1 0.1\n",
            "m-middle": "",
        },
    )
    (dataset_root / "JPEGImages" / "ignored.png").touch()
    (dataset_root / "Labels" / "ignored.TXT").touch()

    first = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names={2: "bulge", 0: "crack", 1: "leakage"},
    )
    second = compute_mbdd2025_annotation_statistics(
        dataset_root,
        class_names={1: "leakage", 2: "bulge", 0: "crack"},
    )

    assert first == second
    assert first.counts.image_count == 3
    assert first.counts.label_file_count == 3
    assert tuple(item.class_id for item in first.classes) == (0, 1, 2)


@pytest.mark.parametrize("anomaly", ["missing", "orphan"])
def test_structural_label_anomaly_is_not_treated_as_empty(
    tmp_path: Path,
    anomaly: str,
) -> None:
    """Missing and orphan labels stop statistics instead of inventing zeros."""
    dataset_root = _write_synthetic_dataset(tmp_path, {"paired": ""})
    if anomaly == "missing":
        (dataset_root / "JPEGImages" / "missing.jpg").touch()
    else:
        (dataset_root / "Labels" / "orphan.txt").touch()

    with pytest.raises(DatasetStatisticsError, match=anomaly):
        compute_mbdd2025_annotation_statistics(
            dataset_root,
            class_names=_CLASS_NAMES,
        )


def test_image_resolution_summaries_and_distribution_are_deterministic(
    tmp_path: Path,
) -> None:
    """Exact resolutions are counted in stable width-height order."""
    dataset_root = tmp_path / "MBDD2025"
    _write_constant_jpeg(
        dataset_root,
        "z-last",
        size=(10, 20),
        intensity=0,
    )
    _write_constant_jpeg(
        dataset_root,
        "a-first",
        size=(30, 40),
        intensity=64,
        mode="RGB",
    )
    _write_constant_jpeg(
        dataset_root,
        "m-middle",
        size=(10, 20),
        intensity=255,
    )

    first = compute_mbdd2025_image_statistics(dataset_root)
    second = compute_mbdd2025_image_statistics(dataset_root)

    assert first == second
    assert isinstance(first, MBDD2025ImageStatistics)
    assert first.summary.width == summarize_numeric([10, 30, 10])
    assert first.summary.height == summarize_numeric([20, 40, 20])
    assert first.resolution_counts == (
        ImageResolutionCount(width=10, height=20, image_count=2),
        ImageResolutionCount(width=30, height=40, image_count=1),
    )


def test_brightness_uses_one_grayscale_mean_per_image(tmp_path: Path) -> None:
    """Constant grayscale and RGB images yield controlled per-image means."""
    dataset_root = tmp_path / "MBDD2025"
    _write_constant_jpeg(
        dataset_root,
        "black",
        size=(8, 8),
        intensity=0,
    )
    _write_constant_jpeg(
        dataset_root,
        "middle",
        size=(16, 8),
        intensity=128,
        mode="RGB",
    )
    _write_constant_jpeg(
        dataset_root,
        "white",
        size=(8, 16),
        intensity=255,
    )

    result = compute_mbdd2025_image_statistics(dataset_root)

    assert result.summary.brightness == summarize_numeric([0, 128, 255])
    assert result.summary.contrast == summarize_numeric([0, 0, 0])


def test_contrast_is_population_standard_deviation(tmp_path: Path) -> None:
    """Equal black and white pixel groups have known population deviation."""
    dataset_root = tmp_path / "MBDD2025"
    _write_two_level_jpeg(
        dataset_root,
        "two-level",
        size=(16, 8),
    )

    result = compute_mbdd2025_image_statistics(dataset_root)

    _assert_single_value_summary(result.summary.brightness, 127.5)
    _assert_single_value_summary(result.summary.contrast, 127.5)


def test_brightness_and_contrast_weight_each_image_equally(tmp_path: Path) -> None:
    """A large image contributes one observation, not one per pixel."""
    dataset_root = tmp_path / "MBDD2025"
    _write_constant_jpeg(
        dataset_root,
        "small-constant",
        size=(8, 8),
        intensity=0,
    )
    _write_two_level_jpeg(
        dataset_root,
        "large-two-level",
        size=(160, 80),
    )

    result = compute_mbdd2025_image_statistics(dataset_root)

    assert result.summary.brightness.count == 2
    assert result.summary.brightness.mean == pytest.approx(63.75)
    assert result.summary.contrast.count == 2
    assert result.summary.contrast.mean == pytest.approx(63.75)


def test_unreadable_jpeg_raises_with_identifying_path(tmp_path: Path) -> None:
    """A discovered but undecodable JPEG fails explicitly."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    images_directory.mkdir(parents=True)
    bad_image_path = images_directory / "bad.jpg"
    bad_image_path.write_bytes(b"not a jpeg")

    with pytest.raises(DatasetStatisticsError, match=r"bad\.jpg"):
        compute_mbdd2025_image_statistics(dataset_root)


def test_full_statistics_composes_annotation_and_image_results(tmp_path: Path) -> None:
    """The top-level result combines approved annotation and image statistics."""
    dataset_root = _write_synthetic_dataset(
        tmp_path,
        {"combined": "0 0.5 0.5 0.2 0.2\n"},
    )
    _write_constant_jpeg(
        dataset_root,
        "combined",
        size=(12, 10),
        intensity=64,
        mode="RGB",
    )

    result = compute_mbdd2025_statistics(
        dataset_root,
        class_names=_CLASS_NAMES,
    )

    assert result.schema_version == 1
    assert result.counts == DatasetCounts(
        image_count=1,
        label_file_count=1,
        usable_annotation_count=1,
    )
    assert result.classes[0] == ClassStatistics(
        class_id=0,
        class_name="crack",
        image_count=1,
        instance_count=1,
    )
    _assert_single_value_summary(result.images.width, 12)
    _assert_single_value_summary(result.images.height, 10)
    _assert_single_value_summary(result.images.brightness, 64)
    _assert_single_value_summary(result.images.contrast, 0)
    assert result.resolution_counts == (
        ImageResolutionCount(width=12, height=10, image_count=1),
    )
