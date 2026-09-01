"""Tests for the MBDD2025 dataset validation CLI."""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.validate_dataset import main


def test_cli_returns_zero_and_prints_summary_for_healthy_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A healthy synthetic dataset produces a compact successful summary."""
    dataset_root = _write_healthy_dataset(tmp_path, class_id=7)
    config_path = _write_config(tmp_path, "classes:\n  '7': custom\n")

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
    assert captured.out.splitlines() == [
        f"Dataset: {dataset_root}",
        "Valid: True",
        "Errors: 0",
        "Warnings: 0",
        "Info: 0",
    ]
    assert captured.err == ""


def test_cli_uses_mbdd2025_project_config_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting --config uses the repository's canonical MBDD2025 taxonomy."""
    dataset_root = _write_healthy_dataset(tmp_path, class_id=0)

    exit_code = main(["--dataset-root", str(dataset_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid: True" in captured.out
    assert captured.err == ""


def test_cli_returns_one_for_invalid_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation errors produce exit code one without dumping issue details."""
    dataset_root = tmp_path / "MBDD2025"
    dataset_root.mkdir()
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Valid: False" in captured.out
    assert "Errors: 2" in captured.out
    assert captured.err == ""


def test_cli_writes_parseable_json_and_creates_parent_directories(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The optional output path receives deterministic machine-readable JSON."""
    dataset_root = _write_healthy_dataset(tmp_path, class_id=0)
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")
    output_path = tmp_path / "artifacts" / "validation" / "mbdd2025.json"

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
    report_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert captured.out.startswith(f"Dataset: {dataset_root}\n")
    assert report_data == {
        "counts": {"ERROR": 0, "INFO": 0, "WARNING": 0},
        "is_valid": True,
        "issues": [],
        "schema_version": 1,
    }
    assert output_path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    "config_contents",
    [
        "name: MBDD2025\n",
        "classes: [\n",
    ],
)
def test_cli_returns_two_for_invalid_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_contents: str,
) -> None:
    """Missing or malformed class configuration produces a clear error."""
    dataset_root = tmp_path / "MBDD2025"
    config_path = _write_config(tmp_path, config_contents)

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("Configuration error:")


def test_cli_rejects_output_inside_dataset_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Derived reports cannot be written into the raw dataset tree."""
    dataset_root = _write_healthy_dataset(tmp_path, class_id=0)
    config_path = _write_config(tmp_path, "classes:\n  0: crack\n")
    output_path = dataset_root / "validation.json"

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
    assert not output_path.exists()
    assert "outside data/raw" in captured.err


def _write_healthy_dataset(tmp_path: Path, *, class_id: int) -> Path:
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()

    with Image.new("RGB", (4, 4), color=(32, 64, 96)) as image:
        image.save(images_directory / "sample.jpg", format="JPEG")
    (labels_directory / "sample.txt").write_text(
        f"{class_id} 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    return dataset_root


def _write_config(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path
