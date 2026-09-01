"""Dataset validation schemas and utilities."""

import math
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from infraguard.data.mbdd import (
    AnnotationParseError,
    YoloAnnotationRow,
    parse_yolo_line,
)

_IMAGE_DIRECTORY_NAME = "JPEGImages"
_LABEL_DIRECTORY_NAME = "Labels"
_IMAGE_SUFFIXES = frozenset({".jpg"})

# Suppress floating-point noise introduced while reconstructing XYXY coordinates.
# This is not a clipping tolerance and does not permit materially out-of-bounds boxes.
OOB_TOLERANCE = 1e-9


class ValidationCode(StrEnum):
    """Stable machine-readable code assigned to a validation issue."""

    MISSING_IMAGE_DIRECTORY = "MISSING_IMAGE_DIRECTORY"
    MISSING_LABEL_DIRECTORY = "MISSING_LABEL_DIRECTORY"
    MISSING_LABEL = "MISSING_LABEL"
    ORPHAN_LABEL = "ORPHAN_LABEL"
    EMPTY_LABEL = "EMPTY_LABEL"
    MALFORMED_ANNOTATION = "MALFORMED_ANNOTATION"
    UNREADABLE_LABEL = "UNREADABLE_LABEL"
    INVALID_CLASS_ID = "INVALID_CLASS_ID"
    INVALID_COORDINATE = "INVALID_COORDINATE"
    NON_POSITIVE_BOX_SIZE = "NON_POSITIVE_BOX_SIZE"
    ZERO_AREA_BOX = "ZERO_AREA_BOX"
    OUT_OF_BOUNDS_BOX = "OUT_OF_BOUNDS_BOX"
    DUPLICATE_BOX = "DUPLICATE_BOX"
    UNREADABLE_IMAGE = "UNREADABLE_IMAGE"


class ValidationSeverity(StrEnum):
    """Severity assigned to a dataset validation issue."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single structured dataset validation finding."""

    code: ValidationCode
    severity: ValidationSeverity
    message: str
    path: Path | None = None
    line_number: int | None = None
    image_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """An immutable collection of dataset validation findings."""

    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the report contains no errors."""
        return self.error_count == 0

    @property
    def error_count(self) -> int:
        """Return the number of error findings."""
        return self._count(ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Return the number of warning findings."""
        return self._count(ValidationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        """Return the number of informational findings."""
        return self._count(ValidationSeverity.INFO)

    def _count(self, severity: ValidationSeverity) -> int:
        return sum(issue.severity is severity for issue in self.issues)


def validate_mbdd2025(
    dataset_root: Path,
    *,
    valid_class_ids: Collection[int],
) -> ValidationReport:
    """Run all MBDD2025 dataset validation stages and aggregate their issues."""
    issues: list[ValidationIssue] = []
    valid_class_id_set = frozenset(valid_class_ids)
    images_directory = dataset_root / _IMAGE_DIRECTORY_NAME
    labels_directory = dataset_root / _LABEL_DIRECTORY_NAME

    if not images_directory.is_dir():
        issues.append(
            ValidationIssue(
                code=ValidationCode.MISSING_IMAGE_DIRECTORY,
                severity=ValidationSeverity.ERROR,
                message="Required image directory is missing.",
                path=Path(_IMAGE_DIRECTORY_NAME),
            )
        )

    if not labels_directory.is_dir():
        issues.append(
            ValidationIssue(
                code=ValidationCode.MISSING_LABEL_DIRECTORY,
                severity=ValidationSeverity.ERROR,
                message="Required label directory is missing.",
                path=Path(_LABEL_DIRECTORY_NAME),
            )
        )

    if issues:
        return _build_report(issues)

    image_paths = _discover_images(images_directory)
    label_paths = _discover_labels(labels_directory)
    issues.extend(_validate_image_label_matching(image_paths, label_paths))

    for label_path in label_paths:
        relative_label_path = Path(_LABEL_DIRECTORY_NAME) / label_path.name
        issues.extend(
            _validate_label_file(
                label_path,
                relative_label_path=relative_label_path,
                valid_class_ids=valid_class_id_set,
            )
        )

    for image_path in image_paths:
        image_issue = _validate_image_integrity(
            image_path,
            relative_image_path=Path(_IMAGE_DIRECTORY_NAME) / image_path.name,
        )
        if image_issue is not None:
            issues.append(image_issue)

    return _build_report(issues)


def _discover_images(images_directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in images_directory.iterdir()
                if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def _discover_labels(labels_directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in labels_directory.iterdir()
                if path.is_file() and path.suffix == ".txt"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def _validate_image_label_matching(
    image_paths: tuple[Path, ...],
    label_paths: tuple[Path, ...],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    label_names = {path.name for path in label_paths}
    image_stems = {path.stem for path in image_paths}

    for image_path in image_paths:
        expected_label_name = f"{image_path.stem}.txt"
        if expected_label_name not in label_names:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.MISSING_LABEL,
                    severity=ValidationSeverity.ERROR,
                    message="Image has no matching YOLO label file.",
                    path=Path(_LABEL_DIRECTORY_NAME) / expected_label_name,
                    image_id=image_path.stem,
                )
            )

    for label_path in label_paths:
        if label_path.stem not in image_stems:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.ORPHAN_LABEL,
                    severity=ValidationSeverity.ERROR,
                    message="YOLO label file has no matching JPEG image.",
                    path=Path(_LABEL_DIRECTORY_NAME) / label_path.name,
                    image_id=label_path.stem,
                )
            )

    return issues


def _validate_image_integrity(
    image_path: Path,
    *,
    relative_image_path: Path,
) -> ValidationIssue | None:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except UnidentifiedImageError:
        message = "Pillow could not identify the file as a readable image."
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        message = (
            f"Pillow could not verify or decode the image ({type(error).__name__})."
        )
    else:
        return None

    return ValidationIssue(
        code=ValidationCode.UNREADABLE_IMAGE,
        severity=ValidationSeverity.ERROR,
        message=message,
        path=relative_image_path,
        image_id=image_path.stem,
    )


def _validate_label_file(
    label_path: Path,
    *,
    relative_label_path: Path,
    valid_class_ids: frozenset[int],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    first_valid_row_lines: dict[YoloAnnotationRow, int] = {}
    annotation_row_count = 0

    try:
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return [
            ValidationIssue(
                code=ValidationCode.UNREADABLE_LABEL,
                severity=ValidationSeverity.ERROR,
                message=f"Could not read YOLO label file ({type(error).__name__}).",
                path=relative_label_path,
                image_id=label_path.stem,
            )
        ]

    for line_number, line in enumerate(label_lines, start=1):
        if not line.strip():
            continue

        annotation_row_count += 1
        try:
            row = parse_yolo_line(
                line,
                label_path=relative_label_path,
                line_number=line_number,
            )
        except AnnotationParseError as error:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.MALFORMED_ANNOTATION,
                    severity=ValidationSeverity.ERROR,
                    message=str(error),
                    path=relative_label_path,
                    line_number=line_number,
                    image_id=label_path.stem,
                )
            )
            continue

        row_issues = _validate_yolo_row(
            row,
            relative_label_path=relative_label_path,
            line_number=line_number,
            image_id=label_path.stem,
            valid_class_ids=valid_class_ids,
        )
        issues.extend(row_issues)
        if row_issues:
            continue

        first_line_number = first_valid_row_lines.get(row)
        if first_line_number is None:
            first_valid_row_lines[row] = line_number
        else:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.DUPLICATE_BOX,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "YOLO box exactly duplicates the valid box on "
                        f"line {first_line_number}."
                    ),
                    path=relative_label_path,
                    line_number=line_number,
                    image_id=label_path.stem,
                )
            )

    if annotation_row_count == 0:
        issues.append(
            ValidationIssue(
                code=ValidationCode.EMPTY_LABEL,
                severity=ValidationSeverity.INFO,
                message="YOLO label file contains no annotations.",
                path=relative_label_path,
                image_id=label_path.stem,
            )
        )

    return issues


def _validate_yolo_row(
    row: YoloAnnotationRow,
    *,
    relative_label_path: Path,
    line_number: int,
    image_id: str,
    valid_class_ids: frozenset[int],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if row.class_id not in valid_class_ids:
        issues.append(
            _row_issue(
                code=ValidationCode.INVALID_CLASS_ID,
                message=f"Class ID {row.class_id} is not in the configured taxonomy.",
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )
        return issues

    coordinate_values = (row.x_center, row.y_center, row.width, row.height)
    if not all(math.isfinite(value) for value in coordinate_values):
        issues.append(
            _row_issue(
                code=ValidationCode.INVALID_COORDINATE,
                message="YOLO coordinates must contain only finite values.",
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )
        return issues

    if (
        not 0 <= row.x_center <= 1
        or not 0 <= row.y_center <= 1
        or row.width > 1
        or row.height > 1
    ):
        issues.append(
            _row_issue(
                code=ValidationCode.INVALID_COORDINATE,
                message=(
                    "YOLO centers must be within [0, 1] and sizes must not exceed 1."
                ),
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )
        return issues

    if row.width < 0 or row.height < 0:
        issues.append(
            _row_issue(
                code=ValidationCode.NON_POSITIVE_BOX_SIZE,
                message="YOLO width and height must not be negative.",
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )
        return issues

    if row.width == 0 or row.height == 0:
        issues.append(
            _row_issue(
                code=ValidationCode.ZERO_AREA_BOX,
                message="YOLO width and height must be greater than zero.",
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )
        return issues

    box = row.to_bounding_box()
    if box.xmax <= box.xmin or box.ymax <= box.ymin:
        issues.append(
            _row_issue(
                code=ValidationCode.ZERO_AREA_BOX,
                message="Converted XYXY box has zero area.",
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )
        return issues

    if (
        box.xmin < -OOB_TOLERANCE
        or box.ymin < -OOB_TOLERANCE
        or box.xmax > 1.0 + OOB_TOLERANCE
        or box.ymax > 1.0 + OOB_TOLERANCE
    ):
        issues.append(
            _row_issue(
                code=ValidationCode.OUT_OF_BOUNDS_BOX,
                message="Converted XYXY box extends outside normalized image bounds.",
                path=relative_label_path,
                line_number=line_number,
                image_id=image_id,
            )
        )

    return issues


def _row_issue(
    *,
    code: ValidationCode,
    message: str,
    path: Path,
    line_number: int,
    image_id: str,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=ValidationSeverity.ERROR,
        message=message,
        path=path,
        line_number=line_number,
        image_id=image_id,
    )


def _build_report(issues: list[ValidationIssue]) -> ValidationReport:
    return ValidationReport(issues=tuple(sorted(issues, key=_issue_sort_key)))


def _issue_sort_key(issue: ValidationIssue) -> tuple[str, int, str, str, str]:
    return (
        issue.path.as_posix() if issue.path is not None else "",
        issue.line_number if issue.line_number is not None else 0,
        issue.code.value,
        issue.image_id or "",
        issue.message,
    )


def validation_report_to_dict(report: ValidationReport) -> dict[str, object]:
    """Convert a validation report to a deterministic JSON-compatible mapping."""
    return {
        "schema_version": 1,
        "is_valid": report.is_valid,
        "counts": {
            "ERROR": report.error_count,
            "WARNING": report.warning_count,
            "INFO": report.info_count,
        },
        "issues": [
            {
                "code": issue.code.value,
                "severity": issue.severity.value,
                "message": issue.message,
                "path": issue.path.as_posix() if issue.path is not None else None,
                "line_number": issue.line_number,
                "image_id": issue.image_id,
            }
            for issue in report.issues
        ],
    }
