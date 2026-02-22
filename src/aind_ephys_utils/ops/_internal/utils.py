"""Utility helpers for ops implementations."""

from __future__ import annotations

import xarray as xr


def preserve_coords(src: xr.DataArray, out: xr.DataArray) -> xr.DataArray:
    """
    Carry over coordinates whose dims are a subset of the output dims.

    This keeps trial/unit metadata available after ops transformations.
    """
    for name, coord in src.coords.items():
        if name in out.coords:
            continue
        if set(coord.dims).issubset(set(out.dims)):
            out = out.assign_coords({name: coord})
    return out
