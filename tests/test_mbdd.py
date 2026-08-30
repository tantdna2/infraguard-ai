"""Tests for the MBDD2025 dataset integration."""

import infraguard
from infraguard.data import mbdd


def test_infraguard_package_imports() -> None:
    """The top-level package is importable."""
    assert infraguard.__name__ == "infraguard"


def test_mbdd_module_imports() -> None:
    """The MBDD2025 integration module is importable."""
    assert mbdd.__name__ == "infraguard.data.mbdd"
