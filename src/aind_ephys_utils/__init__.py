"""Helpful methods for exploring in vivo electrophysiology data."""

from __future__ import annotations

# Important: import accessors so @register_* runs at import time
import matplotlib as mpl

__version__ = "0.1.0"

# Default plotting settings for exports.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"

from . import accessors as _accessors  # noqa: E402,F401
from . import align  # noqa: E402,F401
from .adapters import from_dataframe  # noqa: E402,F401
from .standards.validate import validate  # noqa: E402,F401

__all__ = ["validate"]
