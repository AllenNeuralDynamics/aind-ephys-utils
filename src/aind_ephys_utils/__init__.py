"""Helpful methods for exploring in vivo electrophysiology data."""

from __future__ import annotations

__version__ = "0.0.16"

# Important: import accessors so @register_* runs at import time
from . import accessors as _accessors  # noqa: E402,F401
from . import align  # noqa: E402,F401
from .adapters import from_dataframe  # noqa: E402,F401
from .config import configure_defaults  # noqa: E402,F401
from .ops.pseudopop import pseudopop  # noqa: E402,F401
from .standards.validate import validate  # noqa: E402,F401

__all__ = ["validate", "configure_defaults"]
