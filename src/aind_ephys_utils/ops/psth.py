"""Peristimulus time histogram operations."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..standards.conventions import C
from .utils import preserve_coords


def psth(
    da: xr.DataArray,
    *,
    dim: str = C.trial,
    method: str = "mean",
    group_by: Optional[Union[str, Sequence[str]]] = None,
    keep_trials: bool = False,
) -> xr.DataArray:
    """
    Reduce across trials to compute a PSTH-style summary.

    Parameters
    ----------
    da:
        Input DataArray (binned or continuous).
    dim:
        Dimension to reduce across.
    method:
        Reduction method (e.g. "mean", "median").
    group_by:
        Optional coord name(s) to group along ``dim`` before reducing.
    keep_trials:
        If True, keep per-trial data along with the summary.
    """
    if dim not in da.dims:
        return da.copy()

    if method is not None:
        reduce = method
    reduce = reduce.lower()
    if group_by is not None and keep_trials:
        raise ValueError(
            "keep_trials=True is not supported when group_by is set."
        )

    summary = _grouped_reduce(da, dim=dim, reduce=reduce, group_by=group_by)

    if not keep_trials:
        summary.attrs = dict(da.attrs)
        summary = preserve_coords(da, summary)
        return summary

    summary_exp = summary.expand_dims({dim: ["__summary__"]})
    if dim in da.coords:
        coord = np.asarray(da[dim].values, dtype=object)
        da = da.assign_coords({dim: coord})
    out = xr.concat([da, summary_exp], dim=dim)
    out.attrs = dict(da.attrs)
    out = preserve_coords(da, out)
    return out


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
