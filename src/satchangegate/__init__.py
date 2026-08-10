"""SatChangeGate: a cost-controlled review funnel for satellite change detection."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: read the installed distribution's version rather
    # than duplicating it here, where it silently drifted from pyproject.toml.
    __version__ = version("satchangegate")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
