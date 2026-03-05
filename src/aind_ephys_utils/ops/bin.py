"""Binning operations for ragged spikes."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ..standards.conventions import C
from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)


def bin(  # noqa: C901
    data: DataInput,
    dt: float,
    window: Optional[Tuple[float, float]] = None,
    output: str = "rate",
    time_unit: str = "s",
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """
    Bin ragged spikes into a dense representation.

    Parameters
    ----------
    data:
        Ragged spike DataArray/list or object NumPy array.
    dt:
        Bin width in seconds.
    window:
        Optional (tmin, tmax) window to bin.
    output:
        Output type, typically "rate" or "count".
    time_unit:
        Unit for time values, recorded in attrs.
    dims:
        Optional dimension names used when ``data`` is a dense NumPy array.
        This is ignored for ragged list inputs.
    coords:
        Optional coordinate mapping used when constructing a DataArray from
        dense NumPy input.
    return_type:
        Output type policy: ``"auto"``, ``"xarray"``, or ``"numpy"``.
        ``"auto"`` mirrors the input style (xarray in/xarray out,
        list/NumPy ragged in/list or NumPy out).

    Returns
    -------
    xr.DataArray or object
        Binned dense output. For ``return_type="numpy"``, returns a NumPy
        array with the same dense shape as the xarray result values.
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

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
        # Preserve the original (trial, unit) vs (unit, trial) ordering.
        if da.dims.index(C.trial) < da.dims.index(C.unit):
            data = _bin_ragged(da, edges, (C.trial, C.unit))
            dims = (C.trial, C.unit, C.time)
            coords = {C.trial: da[C.trial], C.unit: da[C.unit], C.time: centers}
        else:
            data = _bin_ragged(da, edges, (C.unit, C.trial))
            dims = (C.unit, C.trial, C.time)
            coords = {C.unit: da[C.unit], C.trial: da[C.trial], C.time: centers}
    elif has_unit:
        data = _bin_ragged(da, edges, (C.unit,))
        dims = (C.unit, C.time)
        coords = {C.unit: da[C.unit], C.time: centers}
    elif has_trial:
        data = _bin_ragged(da, edges, (C.trial,))
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
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )


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


def _bin_ragged(
    da: xr.DataArray, edges: np.ndarray, ordered_dims: tuple
) -> np.ndarray:
    """Bin ragged spikes, iterating over all index combinations of ordered_dims.

    Parameters
    ----------
    da:
        Ragged spike DataArray.
    edges:
        Histogram bin edges.
    ordered_dims:
        Dimension names to iterate over, in the desired output order.
        The output shape is (*[da.sizes[d] for d in ordered_dims], n_bins).
    """
    da_ordered = da.transpose(*ordered_dims)
    shape = da_ordered.shape
    n_bins = len(edges) - 1
    out = np.zeros(shape + (n_bins,), dtype=float)
    for idx in np.ndindex(*shape):
        arr = _ensure_1d_float_array(da_ordered.data[idx])
        if arr.size:
            # np.searchsorted is faster than np.histogram here because the
            # edges are already known and spike times are sorted ascending.
            # np.diff(searchsorted(arr, edges)) gives the same bin counts
            # as np.histogram(arr, bins=edges) for sorted input.
            out[idx] = np.diff(np.searchsorted(arr, edges))
    return out


def _ensure_1d_float_array(x: object) -> np.ndarray:
    """Coerce an entry to a 1D float array of spike times."""
    if x is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Ragged spikes entries must be 1D arrays.")
    return arr
