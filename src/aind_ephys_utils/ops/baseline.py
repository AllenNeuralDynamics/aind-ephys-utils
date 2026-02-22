"""Baseline correction operations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import xarray as xr

from ..standards.conventions import C
from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)


def baseline(
    data: DataInput,
    *,
    window: Tuple[float, float],
    dim: str = C.time,
    mode: str = "subtract",
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """
    Apply baseline correction over a window.

    Parameters
    ----------
    data:
        Input DataArray or NumPy-like data.
    window:
        (tmin, tmax) baseline window.
    dim:
        Dimension to baseline-correct.
    mode:
        Baseline mode ("subtract", "divide", "zscore").
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims.")

    tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(
            f"window must be (min,max) with min < max, got {window}."
        )

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
        denom = std.where(std != 0)
        out = (da - mean) / denom
        out = out.where(std != 0, 0.0)
    else:
        raise ValueError(f"Unknown baseline mode {mode!r}.")

    out.attrs = dict(da.attrs)
    out = preserve_coords(da, out)
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )
