"""Plotting helpers for ephys analysis.

This package provides lightweight matplotlib-based plotting functions for
canonical spike representations.
"""

from .annotations import annotate_events
from .colors import get_color_for_region
from .psth import psth
from .raster import raster
from .trajectory import trajectory

__all__ = [
    "annotate_events",
    "get_color_for_region",
    "raster",
    "psth",
    "trajectory",
]
