"""Restrict operations for spike data."""

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


def restrict(
    data: DataInput,
    *,
    window: Tuple[float, float],
    dim: str = C.time,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """
    Restrict input data to a time window.

    Parameters
    ----------
    data:
        Input DataArray/NumPy array or ragged spikes.
    window:
        (tmin, tmax) interval to keep.
    dim:
        Time dimension name for dense data.
    dims:
        Optional dimension names used when ``data`` is a dense NumPy array.
    coords:
        Optional coordinate mapping used when constructing a DataArray from
        dense NumPy input.
    return_type:
        Output type policy: ``"auto"``, ``"xarray"``, or ``"numpy"``.
        ``"auto"`` mirrors the input style.

    Returns
    -------
    xr.DataArray or object
        Cropped output with data restricted to ``window`` in the selected
        representation.
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    tmin, tmax = window
    if tmin > tmax:
        raise ValueError("window must be (tmin, tmax) with tmin <= tmax.")
    if dim in da.dims:
        out = da.sel({dim: slice(tmin, tmax)})
        out.attrs = dict(da.attrs)
        out.attrs[C.attr_valid_intervals] = [(float(tmin), float(tmax))]
        out = preserve_coords(da, out)
        return from_dataarray_output(
            out, input_kind=input_kind, return_type=return_type
        )
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
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )
