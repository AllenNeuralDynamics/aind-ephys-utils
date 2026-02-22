"""Restrict operations for spike data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr

from ..standards.conventions import C
from ._internal.utils import preserve_coords


def restrict(
    da: xr.DataArray,
    *,
    window: Tuple[float, float],
    dim: str = C.time,
) -> xr.DataArray:
    """
    Restrict a DataArray to a time window.

    Parameters
    ----------
    da:
        Input DataArray, either binned/continuous or ragged spikes.
    window:
        (tmin, tmax) interval to keep.
    dim:
        Time dimension name for dense data.

    Returns
    -------
    xr.DataArray
        Cropped DataArray with data restricted to the window.
    """
    tmin, tmax = window
    if tmin > tmax:
        raise ValueError("window must be (tmin, tmax) with tmin <= tmax.")
    if dim in da.dims:
        out = da.sel({dim: slice(tmin, tmax)})
        out.attrs = dict(da.attrs)
        out.attrs[C.attr_valid_intervals] = [(float(tmin), float(tmax))]
        out = preserve_coords(da, out)
        return out
    if da.dtype != object:
        raise ValueError(
            f"restrict expects a '{dim}' dimension or ragged spikes."
        )

    data = np.empty(da.shape, dtype=object)
    it = np.nditer(
        np.empty(da.shape, dtype=int),
        flags=["multi_index", "refs_ok"],
    )
    for _ in it:
        idx = it.multi_index
        x = da.values[idx]
        if x is None:
            data[idx] = np.asarray([], dtype=float)
            continue
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1D spike arrays, got shape {arr.shape}."
            )
        left = np.searchsorted(arr, tmin, side="left")
        right = np.searchsorted(arr, tmax, side="right")
        data[idx] = arr[left:right]

    out = xr.DataArray(
        data,
        dims=da.dims,
        coords=da.coords,
        attrs=dict(da.attrs),
    )
    out.attrs[C.attr_valid_intervals] = [(float(tmin), float(tmax))]
    return out
