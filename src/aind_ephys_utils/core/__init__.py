"""Pure numpy core operations for electrophysiology analysis.

This package contains no xarray dependency. All functions operate on
plain numpy arrays with a fixed axis convention:

- **Dense data**: always ``(trial, unit, time)``
- **Ragged data**: ``spike_times[trial][unit]`` -> 1-D float array

No ``dim=`` parameters. Every function knows which axis is which.
"""

from .align import align, align_unit
from .baseline import baseline
from .bin import bin_spikes, make_bin_edges
from .normalize import normalize
from .psth import psth
from .restrict import restrict_dense, restrict_ragged
from .smooth import convolve_1d, infer_dt, make_kernel, smooth

__all__ = [
    "align",
    "align_unit",
    "baseline",
    "bin_spikes",
    "convolve_1d",
    "infer_dt",
    "make_bin_edges",
    "make_kernel",
    "normalize",
    "psth",
    "restrict_dense",
    "restrict_ragged",
    "smooth",
]
