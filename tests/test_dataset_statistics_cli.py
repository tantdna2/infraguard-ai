"""Tests for deterministic statistics serialization and its CLI."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from infraguard.data.stats import (
    compute_mbdd2025_statistics,
    statistics_to_dict,
    statistics_to_json,
)
from scripts import dataset_statistics
from scripts.dataset_statistics import ConfigurationError, load_class_names, main


def test_cli_returns_zero_and_prints_headline_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid synthetic dataset produces a concise successful summary."""
    dataset_root = _write_statistics_dataset(tmp_path)
    config_path = _write_config(
        tmp_path,
        "classes:\n  1: leakage\n  0: crack\n",
    )

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Dataset: {dataset_root}" in captured.out
    assert "Images: 2" in captured.out
    assert "Labels: 2" in captured.out
    assert "Usable annotations: 1" in captured.out
    assert "Excluded annotations: 0" in captured.out
    assert "OOB annotations: 1" in captured.out
    assert "Empty labels: 1" in captured.out
    assert "Unique resolutions: 2" in captured.out
    assert captured.err == ""


def test_cli_uses_canonical_config_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting --config uses the repository MBDD2025 taxonomy."""
    dataset_root = _write_statistics_dataset(tmp_path)

    exit_code = main(["--dataset-root", str(dataset_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Usable annotations: 1" in captured.out
    assert captured.err == ""


def test_cli_writes_complete_strict_utf8_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A safe output path receives the full typed report as strict JSON."""
    dataset_root = _write_statistics_dataset(tmp_path)
    config_path = _write_config(
        tmp_path,
        "classes:\n  0: nứt\n  1: leakage\n",
    )
    output_path = tmp_path / "artifacts" / "statistics" / "mbdd2025.json"

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report_text = output_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert exit_code == 0
    assert captured.err == ""
    assert report_text.endswith("\n")
    assert output_path.read_bytes().endswith(b"\n")
    assert not output_path.read_bytes().endswith(b"\r\n")
    assert "nứt" in report_text
    assert report["schema_version"] == 1
    assert report["counts"] == {
        "image_count": 2,
        "label_file_count": 2,
        "usable_annotation_count": 1,
    }
    assert report["classes"] == [
        {
            "class_id": 0,
            "class_name": "nứt",
            "image_count": 1,
            "instance_count": 1,
        },
        {
            "class_id": 1,
            "class_name": "leakage",
            "image_count": 0,
            "instance_count": 0,
        },
    ]
    assert report["quality"]["out_of_bounds_annotation_count"] == 1
    assert report["quality"]["empty_label_count"] == 1
    assert "objects_per_image" in report
    assert "bounding_boxes" in report
    assert "images" in report
    assert report["resolution_counts"] == [
        {"height": 8, "image_count": 1, "width": 8},
        {"height": 8, "image_count": 1, "width": 16},
    ]


def test_statistics_serialization_is_deterministic_and_uses_json_arrays(
    tmp_path: Path,
) -> None:
    """Repeated official serialization produces byte-identical JSON text."""
    dataset_root = _write_statistics_dataset(tmp_path)
    statistics = compute_mbdd2025_statistics(
        dataset_root,
        class_names={1: "leakage", 0: "crack"},
    )

    first = statistics_to_json(statistics)
    second = statistics_to_json(statistics)
    report = statistics_to_dict(statistics)

    assert first == second
    assert first.endswith("\n")
    assert isinstance(report["classes"], list)
    assert isinstance(report["resolution_counts"], list)
    json.loads(first)
    json.dumps(report, allow_nan=False)


def test_statistics_json_rejects_non_finite_metrics(tmp_path: Path) -> None:
    """The official JSON path cannot emit NaN or Infinity tokens."""
    dataset_root = _write_statistics_dataset(tmp_path)
    statistics = compute_mbdd2025_statistics(
        dataset_root,
        class_names={0: "crack"},
    )
    unsafe_summary = replace(statistics.objects_per_image, mean=math.nan)
    unsafe_statistics = replace(
        statistics,
        objects_per_image=unsafe_summary,
    )

    with pytest.raises(ValueError, match="JSON compliant"):
        statistics_to_json(unsafe_statistics)


def test_load_class_names_accepts_and_orders_valid_taxonomy(tmp_path: Path) -> None:
    """Integer-like IDs map to non-empty names in class-ID order."""
    config_path = _write_config(
        tmp_path,
        "classes:\n  '2': bulge\n  0: crack\n  1: leakage\n",
    )

    assert load_class_names(config_path) == {
        0: "crack",
        1: "leakage",
        2: "bulge",
    }


@pytest.mark.parametrize(
    ("config_contents", "message"),
    [
        ("- not-a-mapping\n", "root must be"),
        ("name: MBDD2025\n", "'classes'"),
        ("classes: {}\n", "'classes'"),
        ("classes: [\n", "malformed YAML"),
        ("classes:\n  invalid: crack\n", "not an integer"),
        ("classes:\n  true: crack\n", "not booleans"),
        ("classes:\n  -1: crack\n", "must not be negative"),
        (
            "classes:\n  1: crack\n  '1': leakage\n",
            "duplicate class ID",
        ),
        ("classes:\n  0: [crack]\n", "non-empty string"),
        ("classes:\n  0: ''\n", "non-empty string"),
    ],
)
def test_load_class_names_rejects_invalid_config(
    tmp_path: Path,
    config_contents: str,
    message: str,
) -> None:
    """Malformed taxonomy shapes fail instead of being repaired."""
    config_path = _write_config(tmp_path, config_contents)

    with pytest.raises(ConfigurationError, match=message):
        load_class_names(config_path)


def test_cli_returns_two_for_invalid_config_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configuration failures stay at the CLI boundary."""
    config_path = _write_config(tmp_path, "name: MBDD2025\n")

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "MBDD2025"),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("Configuration error:")


def test_cli_returns_two_for_dataset_error_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dataset layout failures stay at the CLI boundary."""
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "missing"),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("Statistics error:")


def test_cli_rejects_output_inside_repository_raw_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reports cannot be written under the repository raw-data root."""
    dataset_root = _write_statistics_dataset(tmp_path)
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")
    raw_data_root = tmp_path / "repository" / "data" / "raw"
    output_path = raw_data_root / "statistics.json"
    monkeypatch.setattr(dataset_statistics, "_RAW_DATA_ROOT", raw_data_root)

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "outside data/raw" in captured.err
    assert not output_path.exists()


@pytest.mark.parametrize("output_location", ["root", "descendant"])
def test_cli_rejects_output_at_or_inside_dataset_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    output_location: str,
) -> None:
    """The dataset root and every descendant are forbidden output targets."""
    dataset_root = _write_statistics_dataset(tmp_path)
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")
    output_path = (
        dataset_root
        if output_location == "root"
        else dataset_root / "results" / "statistics.json"
    )

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "raw dataset root" in captured.err
    assert output_path == dataset_root or not output_path.exists()


def test_cli_returns_two_without_success_summary_on_output_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed atomic replacement reports an error without claiming success."""
    dataset_root = _write_statistics_dataset(tmp_path)
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")
    output_path = tmp_path / "existing-directory"
    output_path.mkdir()

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("Output error:")
    assert not tuple(tmp_path.glob(".existing-directory.*.tmp"))


def _write_statistics_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()

    with Image.new("L", (8, 8), color=0) as image:
        image.save(images_directory / "a.jpg", format="JPEG", quality=100)
    with Image.new("RGB", (16, 8), color=(255, 255, 255)) as image:
        image.save(images_directory / "b.jpg", format="JPEG", quality=100)

    (labels_directory / "a.txt").write_text(
        "0 0.1 0.5 0.4 0.2\n",
        encoding="utf-8",
    )
    (labels_directory / "b.txt").write_text("", encoding="utf-8")
    return dataset_root


def _write_config(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "statistics-dataset.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path
