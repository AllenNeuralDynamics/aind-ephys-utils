"""Peristimulus time histogram operations."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..core.conventions import C
from .utils import preserve_coords


def psth(
    da: xr.DataArray,
    *,
    dim: str = C.trial,
    reduce: str = "mean",
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
    reduce:
        Reduction method (e.g. "mean", "median").
    keep_trials:
        If True, keep per-trial data along with the summary.
    """
    if dim not in da.dims:
        return da.copy()

    reduce = reduce.lower()
    if not hasattr(da, reduce):
        raise ValueError(f"Unknown reduce method {reduce!r}.")

    reducer = getattr(da, reduce)
    summary = reducer(dim=dim, keep_attrs=True)

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
