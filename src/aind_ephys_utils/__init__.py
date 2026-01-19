"""Helpful methods for exploring in vivo electrophysiology data."""

from __future__ import annotations

import matplotlib as mpl

__version__ = "0.1.0"

# Default plotting settings for exports.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"

# Important: import accessors so @register_* runs at import time
from . import accessors as _accessors  # noqa: F401
from . import align  # noqa: F401

from .adapters import from_dataframe  # noqa: F401
from .core.validate import validate  # noqa: F401

__all__ = ["validate"]
