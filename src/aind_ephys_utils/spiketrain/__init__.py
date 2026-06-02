"""Raw-array spike-train transforms and utilities.

Functions in this subpackage operate directly on numpy arrays and lists
of spike-time arrays (no xarray coupling).
"""

from __future__ import annotations

from . import psth

__all__ = ["psth"]
