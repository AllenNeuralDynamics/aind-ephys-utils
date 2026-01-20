"""Binning operations for ragged spikes."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr

from ..core.conventions import C
from .utils import preserve_coords


def bin(  # noqa: C901
    da: xr.DataArray,
    dt: float,
    window: Optional[Tuple[float, float]] = None,
    output: str = "rate",
    time_unit: str = "s",
) -> xr.DataArray:
    """
    Bin ragged spikes into a dense representation.

    Parameters
    ----------
    da:
        Ragged spike DataArray (typically dims: trial, unit).
    dt:
        Bin width in seconds.
    window:
        Optional (tmin, tmax) window to bin.
    output:
        Output type, typically "rate" or "count".
    time_unit:
        Unit for time values, recorded in attrs.
    """
    if dt <= 0:
        raise ValueError("dt must be a positive number of seconds.")

    if da.dtype != object:
        raise ValueError("bin expects ragged spikes with dtype=object.")

    allowed_dims = {C.trial, C.unit}
    if not set(da.dims).issubset(allowed_dims):
        raise ValueError(
            "bin expects only 'trial'/'unit' dimensions for ragged spikes."
        )

    if window is None:
        window = da.attrs.get(C.attr_valid_intervals, None)
        if window is not None:
            tmin, tmax = window[0]  # Use first valid interval
        else:
            tmin, tmax = _infer_tlim(da)
    else:
        tmin, tmax = window

    if tmin >= tmax:
        raise ValueError(
            f"window must be (min,max) with min < max, got {window}."
        )

    ratio = (tmax - tmin) / dt
    if np.isclose(ratio, round(ratio)):
        n_bins = int(round(ratio))
    else:
        n_bins = int(np.ceil(ratio))
    if n_bins < 1:
        n_bins = 1
    edges = tmin + dt * np.arange(n_bins + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    output = output.lower()
    if output not in ("rate", "count"):
        raise ValueError("output must be 'rate' or 'count'.")

    has_trial = C.trial in da.dims
    has_unit = C.unit in da.dims

    if has_trial and has_unit:
        data = _bin_trial_unit(da, edges)
        if da.dims.index(C.trial) < da.dims.index(C.unit):
            dims = (C.trial, C.unit, C.time)
            coords = {
                C.trial: da[C.trial],
                C.unit: da[C.unit],
                C.time: centers,
            }
        else:
            data = np.transpose(data, (1, 0, 2))
            dims = (C.unit, C.trial, C.time)
            coords = {
                C.unit: da[C.unit],
                C.trial: da[C.trial],
                C.time: centers,
            }
    elif has_unit:
        data = _bin_unit(da, edges)
        dims = (C.unit, C.time)
        coords = {C.unit: da[C.unit], C.time: centers}
    elif has_trial:
        data = _bin_trial(da, edges)
        dims = (C.trial, C.time)
        coords = {C.trial: da[C.trial], C.time: centers}
    else:
        raise ValueError(
            "bin requires at least a 'trial' or 'unit' dimension."
        )

    if output == "rate":
        data = data / dt

    out = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        name=da.name,
        attrs=dict(da.attrs),
    )
    out = preserve_coords(da, out)
    out.attrs[C.attr_kind] = "binned"
    out.attrs[C.attr_time_unit] = time_unit
    if window is not None:
        out.attrs[C.attr_valid_intervals] = [window]
    return out


def _infer_tlim(da: xr.DataArray) -> Tuple[float, float]:
    """Infer (tmin, tmax) from ragged spikes."""
    tmin = np.inf
    tmax = -np.inf
    for x in da.data.ravel():
        if x is None:
            continue
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            continue
        tmin = min(tmin, float(arr.min()))
        tmax = max(tmax, float(arr.max()))
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        raise ValueError("Cannot infer tlim from empty spike arrays.")
    if tmin == tmax:
        tmax = tmin + 1e-6
    return tmin, tmax


def _bin_trial_unit(da: xr.DataArray, edges: np.ndarray) -> np.ndarray:
    """Bin spikes for (trial, unit) ragged input."""
    da_tu = da.transpose(C.trial, C.unit)
    n_trials, n_units = da_tu.shape
    n_bins = len(edges) - 1
    out = np.zeros((n_trials, n_units, n_bins), dtype=float)
    for i in range(n_trials):
        for j in range(n_units):
            arr = _ensure_1d_float_array(da_tu.data[i, j])
            if arr.size:
                out[i, j], _ = np.histogram(arr, bins=edges)
    return out


def _bin_unit(da: xr.DataArray, edges: np.ndarray) -> np.ndarray:
    """Bin spikes for (unit,) ragged input."""
    da_u = da.transpose(C.unit)
    n_units = da_u.shape[0]
    n_bins = len(edges) - 1
    out = np.zeros((n_units, n_bins), dtype=float)
    for j in range(n_units):
        arr = _ensure_1d_float_array(da_u.data[j])
        if arr.size:
            out[j], _ = np.histogram(arr, bins=edges)
    return out


def _bin_trial(da: xr.DataArray, edges: np.ndarray) -> np.ndarray:
    """Bin spikes for (trial,) ragged input."""
    da_t = da.transpose(C.trial)
    n_trials = da_t.shape[0]
    n_bins = len(edges) - 1
    out = np.zeros((n_trials, n_bins), dtype=float)
    for i in range(n_trials):
        arr = _ensure_1d_float_array(da_t.data[i])
        if arr.size:
            out[i], _ = np.histogram(arr, bins=edges)
    return out


def _ensure_1d_float_array(x: object) -> np.ndarray:
    """Coerce an entry to a 1D float array of spike times."""
    if x is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Ragged spikes entries must be 1D arrays.")
    return arr
