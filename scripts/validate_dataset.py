"""Validate an extracted MBDD2025 dataset from the command line."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from infraguard.data.validator import (
    ValidationReport,
    validate_mbdd2025,
    validation_report_to_dict,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "datasets" / "mbdd2025.yaml"
_RAW_DATA_ROOT = _REPOSITORY_ROOT / "data" / "raw"


class ConfigurationError(ValueError):
    """Raised when a dataset configuration cannot provide a valid taxonomy."""


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset validation argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Path to the extracted MBDD2025 root directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Dataset YAML configuration path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation and return a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        valid_class_ids = load_valid_class_ids(args.config)
        if args.output is not None:
            _ensure_safe_output_path(args.output, dataset_root=args.dataset_root)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    report = validate_mbdd2025(
        args.dataset_root,
        valid_class_ids=valid_class_ids,
    )
    _print_summary(args.dataset_root, report)

    if args.output is not None:
        try:
            _write_json_report(args.output, report)
        except OSError as error:
            print(f"Output error: {error}", file=sys.stderr)
            return 2

    return 0 if report.is_valid else 1


def load_valid_class_ids(config_path: Path) -> frozenset[int]:
    """Load canonical class IDs from a dataset YAML configuration."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"could not read {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"malformed YAML in {config_path}: {error}") from error

    if not isinstance(config, Mapping):
        raise ConfigurationError("configuration root must be a YAML mapping")

    classes = config.get("classes")
    if not isinstance(classes, Mapping) or not classes:
        raise ConfigurationError("'classes' must be a non-empty YAML mapping")

    class_ids: set[int] = set()
    for raw_class_id in classes:
        class_id = _parse_class_id(raw_class_id)
        if class_id in class_ids:
            raise ConfigurationError(f"duplicate class ID after conversion: {class_id}")
        class_ids.add(class_id)

    return frozenset(class_ids)


def _parse_class_id(raw_class_id: object) -> int:
    if isinstance(raw_class_id, bool):
        raise ConfigurationError("class IDs must be integers, not booleans")

    if isinstance(raw_class_id, int):
        class_id = raw_class_id
    elif isinstance(raw_class_id, str):
        try:
            class_id = int(raw_class_id)
        except ValueError as error:
            raise ConfigurationError(
                f"class ID {raw_class_id!r} is not an integer"
            ) from error
    else:
        raise ConfigurationError(f"class ID {raw_class_id!r} is not an integer")

    if class_id < 0:
        raise ConfigurationError(f"class ID {class_id} must not be negative")

    return class_id


def _ensure_safe_output_path(output_path: Path, *, dataset_root: Path) -> None:
    resolved_output = output_path.resolve()
    forbidden_roots = (_RAW_DATA_ROOT.resolve(), dataset_root.resolve())
    if any(_is_within(resolved_output, root) for root in forbidden_roots):
        raise ConfigurationError(
            "--output must be outside data/raw and the raw dataset root"
        )


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _print_summary(dataset_root: Path, report: ValidationReport) -> None:
    print(f"Dataset: {dataset_root}")
    print(f"Valid: {report.is_valid}")
    print(f"Errors: {report.error_count}")
    print(f"Warnings: {report.warning_count}")
    print(f"Info: {report.info_count}")


def _write_json_report(output_path: Path, report: ValidationReport) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(
        validation_report_to_dict(report),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output_path.write_text(f"{report_json}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
