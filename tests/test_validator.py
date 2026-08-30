"""Tests for dataset validation."""

from infraguard.data import validator


def test_validator_module_imports() -> None:
    """The dataset validator module is importable."""
    assert validator.__name__ == "infraguard.data.validator"
