"""Compute deterministic statistics for an extracted MBDD2025 dataset."""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from infraguard.data.mbdd import DatasetLayoutError
from infraguard.data.stats import (
    DatasetStatisticsError,
    MBDD2025Statistics,
    compute_mbdd2025_statistics,
    statistics_to_json,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "datasets" / "mbdd2025.yaml"
_RAW_DATA_ROOT = _REPOSITORY_ROOT / "data" / "raw"
_SUCCESS_EXIT_CODE = 0
_OPERATIONAL_ERROR_EXIT_CODE = 2


class ConfigurationError(ValueError):
    """Raised when a dataset configuration cannot provide a valid taxonomy."""


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset statistics argument parser."""
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
        help="Optional path for the deterministic JSON statistics report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run statistics and return the explicit process exit code."""
    args = build_parser().parse_args(argv)

    try:
        class_names = load_class_names(args.config)
        if args.output is not None:
            _ensure_safe_output_path(args.output, dataset_root=args.dataset_root)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return _OPERATIONAL_ERROR_EXIT_CODE

    try:
        statistics = compute_mbdd2025_statistics(
            args.dataset_root,
            class_names=class_names,
        )
    except (DatasetLayoutError, DatasetStatisticsError) as error:
        print(f"Statistics error: {error}", file=sys.stderr)
        return _OPERATIONAL_ERROR_EXIT_CODE

    if args.output is not None:
        try:
            _write_json_report(args.output, statistics)
        except (OSError, ValueError) as error:
            print(f"Output error: {error}", file=sys.stderr)
            return _OPERATIONAL_ERROR_EXIT_CODE

    _print_summary(args.dataset_root, statistics)
    return _SUCCESS_EXIT_CODE


def load_class_names(config_path: Path) -> dict[int, str]:
    """Load integer IDs and non-empty string names from a YAML taxonomy."""
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

    class_names: dict[int, str] = {}
    for raw_class_id, raw_class_name in classes.items():
        class_id = _parse_class_id(raw_class_id)
        if class_id in class_names:
            raise ConfigurationError(f"duplicate class ID after conversion: {class_id}")
        if not isinstance(raw_class_name, str) or not raw_class_name.strip():
            raise ConfigurationError(
                f"class name for ID {class_id} must be a non-empty string"
            )
        class_names[class_id] = raw_class_name

    return dict(sorted(class_names.items()))


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


def _print_summary(dataset_root: Path, statistics: MBDD2025Statistics) -> None:
    print(f"Dataset: {dataset_root}")
    print(f"Images: {statistics.counts.image_count}")
    print(f"Labels: {statistics.counts.label_file_count}")
    print(f"Usable annotations: {statistics.counts.usable_annotation_count}")
    print(f"Excluded annotations: {statistics.quality.excluded_annotation_count}")
    print(f"OOB annotations: {statistics.quality.out_of_bounds_annotation_count}")
    print(f"Empty labels: {statistics.quality.empty_label_count}")
    print(f"Unique resolutions: {len(statistics.resolution_counts)}")


def _write_json_report(
    output_path: Path,
    statistics: MBDD2025Statistics,
) -> None:
    report_json = statistics_to_json(statistics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report_json)
        temporary_path.replace(output_path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
