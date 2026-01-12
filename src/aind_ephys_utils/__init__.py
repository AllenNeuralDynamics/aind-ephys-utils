"""Helpful methods for exploring in vivo electrophysiology data."""

from __future__ import annotations

__version__ = "0.1.0"

# Important: import accessors so @register_* runs at import time
from . import accessors as _accessors  # noqa: F401
from . import align  # noqa: F401
from . import metrics  # noqa: F401
from . import sort  # noqa: F401
from .adapters import from_dataframe  # noqa: F401
from .core.validate import validate  # noqa: F401

__all__ = ["validate"]
