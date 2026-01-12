"""Plotting helpers for ephys analysis.

This package provides lightweight matplotlib-based plotting functions for
canonical spike representations.
"""

from .psth import psth
from .raster import raster

__all__ = ["raster", "psth"]
