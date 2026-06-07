"""Peristimulus time histogram operations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..standards.conventions import C
from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)
from .bin import BinSize, bin
from .smooth import smooth


def psth(  # noqa: C901
    data: DataInput,
    *,
    dim: str = C.trial,
    method: str = "mean",
    group_by: Optional[Union[str, Sequence[str]]] = None,
    keep_trials: bool = False,
    bin_size: Optional[BinSize] = None,
    smooth_window: Optional[float] = None,
    window: Optional[Tuple[float, float]] = None,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """
    Reduce across trials to compute a PSTH-style summary.

    Parameters
    ----------
    data:
        Input DataArray or NumPy-like data (binned or continuous).
    dim:
        Dimension to reduce across.
    method:
        Reduction method (e.g. "mean", "median").
    group_by:
        Optional coord name(s) to group along ``dim`` before reducing.
    keep_trials:
        If True, keep per-trial data along with the summary.
    bin_size:
        For ragged spikes, bin width in seconds or ``"auto"`` for
        Shimazaki-Shinomoto optimal bin-width selection.
    smooth_window:
        Optional boxcar smoothing window in seconds, applied after reduction.
    window:
        Optional ``(tmin, tmax)`` window passed to ``bin`` for ragged input.
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
        PSTH summary (or summary plus trials when ``keep_trials=True``) in the
        selected output representation.
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    if bin_size is not None:
        if da.dtype != object:
            raise ValueError("bin_size is only supported for ragged spikes.")
        da = bin(
            da,
            bin_size=bin_size,
            window=window,
            return_type="xarray",
        )
    elif da.dtype == object:
        raise ValueError("psth requires bin_size for ragged spikes.")

    if dim not in da.dims:
        summary = da.copy()
        if smooth_window is not None:
            summary = smooth(
                summary,
                method="boxcar",
                window=smooth_window,
                return_type="xarray",
            )
        return from_dataarray_output(
            summary, input_kind=input_kind, return_type=return_type
        )

    if not isinstance(method, str):
        raise ValueError("method must be a non-empty string.")
    method_name = method.strip().lower()
    if method_name == "":
        raise ValueError("method must be a non-empty string.")

    if group_by is not None and keep_trials:
        raise ValueError(
            "keep_trials=True is not supported when group_by is set."
        )

    summary = _grouped_reduce(
        da, dim=dim, reduce=method_name, group_by=group_by
    )
    if smooth_window is not None:
        summary = smooth(
            summary,
            method="boxcar",
            window=smooth_window,
            return_type="xarray",
        )

    if not keep_trials:
        summary.attrs = dict(da.attrs)
        summary = preserve_coords(da, summary)
        return from_dataarray_output(
            summary, input_kind=input_kind, return_type=return_type
        )

    summary_exp = summary.expand_dims({dim: ["__summary__"]})
    if dim in da.coords:
        coord = np.asarray(da[dim].values, dtype=object)
        da = da.assign_coords({dim: coord})
    out = xr.concat([da, summary_exp], dim=dim)
    out.attrs = dict(da.attrs)
    out = preserve_coords(da, out)
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )


def _grouped_reduce(
    da: xr.DataArray,
    *,
    dim: str,
    reduce: str,
    group_by: Optional[Union[str, Sequence[str]]],
) -> xr.DataArray:
    """Apply optional groupby over ``dim`` and then reduce."""
    if group_by is None:
        if not hasattr(da, reduce):
            raise ValueError(f"Unknown method {reduce!r}.")
        return getattr(da, reduce)(dim=dim, keep_attrs=True)

    if isinstance(group_by, str):
        group_by = [group_by]
    for g in group_by:
        if g not in da.coords:
            raise ValueError(f"group_by coord {g!r} not found in DataArray.")
        if dim not in da[g].dims:
            raise ValueError(
                f"group_by coord {g!r} must be defined over {dim!r}."
            )

    if len(group_by) == 1:
        grouped = da.groupby(group_by[0])
        if not hasattr(grouped, reduce):
            raise ValueError(f"Unknown method {reduce!r}.")
        return getattr(grouped, reduce)(dim=dim, keep_attrs=True)

    cond_index = pd.MultiIndex.from_arrays(
        [np.asarray(da[g].values) for g in group_by], names=list(group_by)
    )
    da2 = da.assign_coords(_group=(dim, cond_index))
    grouped = da2.groupby("_group")
    if not hasattr(grouped, reduce):
        raise ValueError(f"Unknown method {reduce!r}.")
    out = getattr(grouped, reduce)(dim=dim, keep_attrs=True)
    return out.unstack("_group")
