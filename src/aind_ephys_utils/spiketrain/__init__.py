"""Raw-array spike-train transforms and utilities.

Functions in this subpackage operate directly on numpy arrays and lists
of spike-time arrays (no xarray coupling).  The names re-exported here
are the main entry points; the full surface is reachable via the
submodules (e.g. ``spiketrain.psth``).
"""

from __future__ import annotations

from .psth import select_psth_bin_width, sliding_window_psth

__all__ = [
    "sliding_window_psth",
    "select_psth_bin_width",
]
