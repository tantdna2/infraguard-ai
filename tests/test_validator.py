"""Tests for dataset validation."""

import json
from pathlib import Path

import pytest
from PIL import Image
from infraguard.data import validator
from infraguard.data.validator import (
    ValidationCode,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    validate_mbdd2025,
    validation_report_to_dict,
)


def test_validator_module_imports() -> None:
    """The dataset validator module is importable."""
    assert validator.__name__ == "infraguard.data.validator"
    assert ValidationIssue.__annotations__["code"] is ValidationCode


def test_report_without_issues_is_valid() -> None:
    """An empty validation report contains no errors."""
    report = ValidationReport(issues=())

    assert report.is_valid
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.info_count == 0


def test_warning_and_info_issues_keep_report_valid() -> None:
    """Warnings and informational findings do not invalidate a report."""
    report = ValidationReport(
        issues=(
            ValidationIssue(
                code=ValidationCode.DUPLICATE_BOX,
                severity=ValidationSeverity.WARNING,
                message="Duplicate box.",
            ),
            ValidationIssue(
                code=ValidationCode.EMPTY_LABEL,
                severity=ValidationSeverity.INFO,
                message="Label file is empty.",
            ),
        )
    )

    assert report.is_valid
    assert report.error_count == 0
    assert report.warning_count == 1
    assert report.info_count == 1


def test_error_issue_invalidates_report() -> None:
    """At least one error finding makes a report invalid."""
    report = ValidationReport(
        issues=(
            ValidationIssue(
                code=ValidationCode.MISSING_LABEL,
                severity=ValidationSeverity.ERROR,
                message="Label file is missing.",
            ),
            ValidationIssue(
                code=ValidationCode.EMPTY_LABEL,
                severity=ValidationSeverity.INFO,
                message="Label file is empty.",
            ),
        )
    )

    assert not report.is_valid
    assert report.error_count == 1
    assert report.warning_count == 0
    assert report.info_count == 1


def test_report_serialization_is_deterministic_and_json_compatible() -> None:
    """Serialization has stable fields and converts enums and paths."""
    report = ValidationReport(
        issues=(
            ValidationIssue(
                code=ValidationCode.MALFORMED_ANNOTATION,
                severity=ValidationSeverity.ERROR,
                message="Expected five fields.",
                path=Path("Labels") / "example.txt",
                line_number=3,
                image_id="example",
            ),
            ValidationIssue(
                code=ValidationCode.EMPTY_LABEL,
                severity=ValidationSeverity.INFO,
                message="Label file is empty.",
            ),
        )
    )

    expected = {
        "schema_version": 1,
        "is_valid": False,
        "counts": {"ERROR": 1, "WARNING": 0, "INFO": 1},
        "issues": [
            {
                "code": "MALFORMED_ANNOTATION",
                "severity": "ERROR",
                "message": "Expected five fields.",
                "path": "Labels/example.txt",
                "line_number": 3,
                "image_id": "example",
            },
            {
                "code": "EMPTY_LABEL",
                "severity": "INFO",
                "message": "Label file is empty.",
                "path": None,
                "line_number": None,
                "image_id": None,
            },
        ],
    }

    serialized = validation_report_to_dict(report)

    assert serialized == expected
    assert json.dumps(serialized, sort_keys=True) == json.dumps(
        validation_report_to_dict(report), sort_keys=True
    )


def test_missing_image_directory_is_reported_without_raising(tmp_path: Path) -> None:
    """A missing JPEGImages directory is a structured validation error."""
    dataset_root = tmp_path / "MBDD2025"
    (dataset_root / "Labels").mkdir(parents=True)

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert tuple(issue.code for issue in report.issues) == ("MISSING_IMAGE_DIRECTORY",)
    assert report.issues[0].severity is ValidationSeverity.ERROR
    assert report.issues[0].path == Path("JPEGImages")


def test_missing_label_directory_is_reported_without_raising(tmp_path: Path) -> None:
    """A missing Labels directory is a structured validation error."""
    dataset_root = tmp_path / "MBDD2025"
    (dataset_root / "JPEGImages").mkdir(parents=True)

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert tuple(issue.code for issue in report.issues) == ("MISSING_LABEL_DIRECTORY",)
    assert report.issues[0].severity is ValidationSeverity.ERROR
    assert report.issues[0].path == Path("Labels")


def test_image_without_label_is_reported(tmp_path: Path) -> None:
    """An image without a stem-matched lowercase .txt label is an error."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    _write_valid_jpeg(images_directory / "a.jpg")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert len(report.issues) == 1
    assert report.issues[0].code == "MISSING_LABEL"
    assert report.issues[0].severity is ValidationSeverity.ERROR
    assert report.issues[0].path == Path("Labels/a.txt")
    assert report.issues[0].image_id == "a"


def test_orphan_label_is_reported_as_error(tmp_path: Path) -> None:
    """A label without a stem-matched JPEG image is an error."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    (labels_directory / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert len(report.issues) == 1
    assert report.issues[0].code == "ORPHAN_LABEL"
    assert report.issues[0].severity is ValidationSeverity.ERROR
    assert report.issues[0].path == Path("Labels/b.txt")
    assert report.issues[0].image_id == "b"


@pytest.mark.parametrize("label_contents", ["", " \n\t\n"])
def test_empty_or_whitespace_only_label_is_informational(
    tmp_path: Path, label_contents: str
) -> None:
    """An existing label without annotation rows is explicit but valid."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    _write_valid_jpeg(images_directory / "a.jpg")
    (labels_directory / "a.txt").write_text(label_contents, encoding="utf-8")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert report.is_valid
    assert len(report.issues) == 1
    assert report.issues[0].code == "EMPTY_LABEL"
    assert report.issues[0].severity is ValidationSeverity.INFO
    assert report.issues[0].path == Path("Labels/a.txt")
    assert report.issues[0].image_id == "a"


def test_healthy_image_and_label_pair_has_no_structural_issues(
    tmp_path: Path,
) -> None:
    """A matched image and non-empty label pass structural validation."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    _write_valid_jpeg(images_directory / "healthy.jpg")
    (labels_directory / "healthy.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert report == ValidationReport(issues=())
    assert report.is_valid


def test_structural_issue_order_is_deterministic(tmp_path: Path) -> None:
    """Discovery order does not affect canonical report ordering."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    _write_valid_jpeg(images_directory / "z.jpg")
    _write_valid_jpeg(images_directory / "a.JPG")
    (labels_directory / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    first_report = validate_mbdd2025(dataset_root, valid_class_ids={0})
    second_report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert tuple(issue.path for issue in first_report.issues) == (
        Path("Labels/a.txt"),
        Path("Labels/b.txt"),
        Path("Labels/z.txt"),
    )
    assert first_report == second_report


def test_uppercase_label_extension_does_not_satisfy_loader_pair(
    tmp_path: Path,
) -> None:
    """Only lowercase .txt labels satisfy the Day 3 loader convention."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    _write_valid_jpeg(images_directory / "a.jpg")
    (labels_directory / "a.TXT").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert tuple(issue.code for issue in report.issues) == ("MISSING_LABEL",)


def test_invalid_class_id_is_reported_from_caller_taxonomy(tmp_path: Path) -> None:
    """The validator rejects classes outside the caller-provided taxonomy."""
    dataset_root = _write_label_fixture(tmp_path, "4 0.5 0.5 0.2 0.2\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={7})

    assert _row_issue_codes(report) == ("INVALID_CLASS_ID",)
    assert report.issues[0].line_number == 1
    assert report.issues[0].path == Path("Labels/sample.txt")


@pytest.mark.parametrize(
    "row",
    [
        "0 -0.1 0.5 0.2 0.2",
        "0 1.1 0.5 0.2 0.2",
        "0 0.5 -0.1 0.2 0.2",
        "0 0.5 1.1 0.2 0.2",
        "0 0.5 0.5 1.1 0.2",
    ],
)
def test_out_of_range_normalized_value_is_invalid(tmp_path: Path, row: str) -> None:
    """YOLO centers and maximum sizes must remain in normalized range."""
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("INVALID_COORDINATE",)


@pytest.mark.parametrize(
    "row",
    [
        "0 nan 0.5 0.2 0.2",
        "0 0.5 inf 0.2 0.2",
        "0 0.5 0.5 -inf 0.2",
    ],
)
def test_non_finite_coordinate_is_invalid(tmp_path: Path, row: str) -> None:
    """NaN and infinite values are invalid even though float parses them."""
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("INVALID_COORDINATE",)


@pytest.mark.parametrize(
    ("row", "expected_code"),
    [
        ("0 0.5 0.5 -0.1 0.2", "NON_POSITIVE_BOX_SIZE"),
        ("0 0.5 0.5 0.2 -0.1", "NON_POSITIVE_BOX_SIZE"),
        ("0 0.5 0.5 0 0.2", "ZERO_AREA_BOX"),
        ("0 0.5 0.5 0.2 0", "ZERO_AREA_BOX"),
    ],
)
def test_invalid_box_size_uses_one_root_cause_code(
    tmp_path: Path, row: str, expected_code: str
) -> None:
    """Negative and zero sizes use distinct codes without bounds cascades."""
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == (expected_code,)


@pytest.mark.parametrize(
    "row",
    [
        "0 0.1 0.5 0.4 0.2",
        "0 0.9 0.5 0.4 0.2",
        "0 0.5 0.1 0.2 0.4",
        "0 0.5 0.9 0.2 0.4",
    ],
)
def test_box_crossing_image_boundary_is_reported(tmp_path: Path, row: str) -> None:
    """A positive normalized box may still cross an XYXY image boundary."""
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("OUT_OF_BOUNDS_BOX",)


@pytest.mark.parametrize(
    "row",
    [
        "0 0.9000000000000002 0.5 0.2 0.2",
        "0 0.0999999999999998 0.5 0.2 0.2",
    ],
)
def test_floating_point_boundary_noise_is_not_out_of_bounds(
    tmp_path: Path, row: str
) -> None:
    """Roughly 2e-16 XYXY boundary noise remains within the 1e-9 tolerance."""
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert "OUT_OF_BOUNDS_BOX" not in _row_issue_codes(report)
    assert report.is_valid


@pytest.mark.parametrize(
    "row",
    [
        "0 0.9000000011 0.5 0.2 0.2",
        "0 0.9001 0.5 0.2 0.2",
    ],
)
def test_box_beyond_oob_tolerance_is_reported(tmp_path: Path, row: str) -> None:
    """Both just-over-tolerance and representative material OOB remain errors."""
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("OUT_OF_BOUNDS_BOX",)


def test_exact_duplicate_valid_box_is_warning(tmp_path: Path) -> None:
    """Only the later occurrence of an exact valid duplicate is reported."""
    row = "0 0.5 0.5 0.2 0.2"
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert report.is_valid
    assert _row_issue_codes(report) == ("DUPLICATE_BOX",)
    assert report.issues[0].severity is ValidationSeverity.WARNING
    assert report.issues[0].line_number == 2


def test_invalid_rows_are_not_duplicate_candidates(tmp_path: Path) -> None:
    """Invalid rows do not create OOB or duplicate cascades."""
    row = "9 0.1 0.5 0.4 0.2"
    dataset_root = _write_label_fixture(tmp_path, f"{row}\n{row}\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == (
        "INVALID_CLASS_ID",
        "INVALID_CLASS_ID",
    )
    assert report.warning_count == 0


def test_valid_row_with_nonstandard_configured_class_has_no_issue(
    tmp_path: Path,
) -> None:
    """A valid row is accepted without a hardcoded MBDD2025 taxonomy."""
    dataset_root = _write_label_fixture(tmp_path, "7 0.5 0.5 0.2 0.2\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={7})

    assert report == ValidationReport(issues=())


def test_malformed_rows_do_not_stop_validation_of_later_rows(
    tmp_path: Path,
) -> None:
    """Each malformed row is reported and later rows are still validated."""
    dataset_root = _write_label_fixture(
        tmp_path,
        "0 0.5 0.5 0.2\n"
        "not-an-integer 0.5 0.5 0.2 0.2\n"
        "0 0.5 not-a-float 0.2 0.2\n"
        "9 0.5 0.5 0.2 0.2\n",
    )

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == (
        "MALFORMED_ANNOTATION",
        "MALFORMED_ANNOTATION",
        "MALFORMED_ANNOTATION",
        "INVALID_CLASS_ID",
    )
    assert tuple(issue.line_number for issue in report.issues) == (1, 2, 3, 4)


def test_valid_small_jpeg_has_no_unreadable_image_issue(tmp_path: Path) -> None:
    """A JPEG created by Pillow passes integrity verification and decoding."""
    dataset_root = _write_label_fixture(tmp_path, "0 0.5 0.5 0.2 0.2\n")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert "UNREADABLE_IMAGE" not in _row_issue_codes(report)
    assert report.is_valid


def test_random_bytes_with_jpg_extension_are_unreadable(tmp_path: Path) -> None:
    """A .jpg file containing arbitrary bytes is an image integrity error."""
    dataset_root = _write_label_fixture(tmp_path, "0 0.5 0.5 0.2 0.2\n")
    (dataset_root / "JPEGImages" / "sample.jpg").write_bytes(b"not a jpeg")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("UNREADABLE_IMAGE",)
    assert report.issues[0].severity is ValidationSeverity.ERROR
    assert report.issues[0].path == Path("JPEGImages/sample.jpg")
    assert report.issues[0].image_id == "sample"


def test_truncated_jpeg_is_unreadable(tmp_path: Path) -> None:
    """A JPEG that opens but cannot fully decode is an integrity error."""
    dataset_root = _write_label_fixture(tmp_path, "0 0.5 0.5 0.2 0.2\n")
    image_path = dataset_root / "JPEGImages" / "sample.jpg"
    image_bytes = image_path.read_bytes()
    image_path.write_bytes(image_bytes[: len(image_bytes) // 2])

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("UNREADABLE_IMAGE",)


def test_unreadable_label_is_reported_without_stopping_dataset(
    tmp_path: Path,
) -> None:
    """Invalid UTF-8 label data becomes an issue instead of escaping the API."""
    dataset_root = _write_label_fixture(tmp_path, "0 0.5 0.5 0.2 0.2\n")
    (dataset_root / "Labels" / "sample.txt").write_bytes(b"\xff\xfe")

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == ("UNREADABLE_LABEL",)
    assert report.issues[0].severity is ValidationSeverity.ERROR
    assert report.issues[0].path == Path("Labels/sample.txt")


def test_aggregate_validator_collects_multiple_dataset_problems(
    tmp_path: Path,
) -> None:
    """One run reports independent image, pairing, syntax, and semantic issues."""
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()

    (images_directory / "corrupt.jpg").write_bytes(b"not a jpeg")
    _write_valid_jpeg(images_directory / "missing.jpg")
    _write_valid_jpeg(images_directory / "mixed.jpg")

    (labels_directory / "corrupt.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    valid_row = "0 0.5 0.5 0.2 0.2"
    (labels_directory / "mixed.txt").write_text(
        f"0 0.5 0.5 0.2\n9 0.5 0.5 0.2 0.2\n{valid_row}\n{valid_row}\n",
        encoding="utf-8",
    )

    report = validate_mbdd2025(dataset_root, valid_class_ids={0})

    assert _row_issue_codes(report) == (
        "UNREADABLE_IMAGE",
        "MISSING_LABEL",
        "MALFORMED_ANNOTATION",
        "INVALID_CLASS_ID",
        "DUPLICATE_BOX",
    )
    assert report.error_count == 4
    assert report.warning_count == 1
    assert not report.is_valid


def _write_label_fixture(tmp_path: Path, label_contents: str) -> Path:
    dataset_root = tmp_path / "MBDD2025"
    images_directory = dataset_root / "JPEGImages"
    labels_directory = dataset_root / "Labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir()
    _write_valid_jpeg(images_directory / "sample.jpg")
    (labels_directory / "sample.txt").write_text(label_contents, encoding="utf-8")
    return dataset_root


def _write_valid_jpeg(image_path: Path) -> None:
    with Image.new("RGB", (4, 4), color=(32, 64, 96)) as image:
        image.save(image_path, format="JPEG")


def _row_issue_codes(report: ValidationReport) -> tuple[str, ...]:
    return tuple(issue.code.value for issue in report.issues)
