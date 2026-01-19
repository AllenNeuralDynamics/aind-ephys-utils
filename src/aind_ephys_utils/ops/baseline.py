"""Baseline correction operations."""

from __future__ import annotations

from typing import Tuple

import xarray as xr

from ..core.conventions import C
from .utils import preserve_coords


def baseline(
    da: xr.DataArray,
    *,
    window: Tuple[float, float],
    dim: str = C.time,
    mode: str = "subtract",
) -> xr.DataArray:
    """
    Apply baseline correction over a window.

    Parameters
    ----------
    da:
        Input DataArray.
    window:
        (tmin, tmax) baseline window.
    dim:
        Dimension to baseline-correct.
    mode:
        Baseline mode ("subtract", "divide", "zscore").
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims.")

    tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(f"window must be (min,max) with min < max, got {window}.")

    baseline_da = da.sel({dim: slice(tmin, tmax)})
    if baseline_da.sizes.get(dim, 0) == 0:
        raise ValueError("Baseline window produced no samples.")

    mean = baseline_da.mean(dim=dim, keep_attrs=True)
    mode = mode.lower()

    if mode == "subtract":
        out = da - mean
    elif mode == "divide":
        out = da / mean
    elif mode == "zscore":
        std = baseline_da.std(dim=dim, keep_attrs=True)
        out = (da - mean) / std
    else:
        raise ValueError(f"Unknown baseline mode {mode!r}.")

    out.attrs = dict(da.attrs)
    out = preserve_coords(da, out)
    return out
