"""Normalization operations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import xarray as xr

from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)


def normalize(
    data: DataInput,
    *,
    dim: Union[str, Tuple[str, ...]],
    method: str = "zscore",
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """
    Normalize data across one or more dimensions.

    Parameters
    ----------
    data:
        Input DataArray or NumPy-like data.
    dim:
        Dimension(s) to normalize across.
    method:
        Normalization method (e.g. "zscore", "minmax", "robust").
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    if isinstance(dim, str):
        dims = (dim,)
    else:
        dims = tuple(dim)

    for d in dims:
        if d not in da.dims:
            raise ValueError(f"dim {d!r} not found in DataArray dims.")

    method = method.lower()
    if method == "zscore":
        mean = da.mean(dim=dims, keep_attrs=True)
        std = da.std(dim=dims, keep_attrs=True)
        denom = std.where(std != 0)
        out = (da - mean) / denom
        out = out.where(std != 0, 0.0)
    elif method == "minmax":
        vmin = da.min(dim=dims, keep_attrs=True)
        vmax = da.max(dim=dims, keep_attrs=True)
        denom = (vmax - vmin).where(vmax != vmin)
        out = (da - vmin) / denom
        out = out.where(vmax != vmin, 0.0)
    elif method == "robust":
        qs = da.quantile([0.25, 0.75], dim=dims, keep_attrs=True)
        q25 = qs.sel(quantile=0.25)
        q75 = qs.sel(quantile=0.75)
        iqr = q75 - q25
        median = da.quantile(0.5, dim=dims, keep_attrs=True)
        denom = iqr.where(iqr != 0)
        out = (da - median) / denom
        out = out.where(iqr != 0, 0.0)
    else:
        raise ValueError(f"Unknown normalization method {method!r}.")

    out.attrs = dict(da.attrs)
    out = preserve_coords(da, out)
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )
