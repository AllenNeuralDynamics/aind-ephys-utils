"""Dimensionality reduction operations.

This module will contain xarray-native reduction helpers (e.g., PCA) used by
the `.ephys.reduce` accessor.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

from .utils import preserve_coords

def reduce(
    da: xr.DataArray,
    *,
    method: str,
    dim: str,
    n: int,
    stack: Optional[Tuple[str, ...]] = None,
    unstack: bool = True,
) -> xr.DataArray:
    """
    Reduce data dimensionality in an xarray-friendly way.

    Parameters
    ----------
    da:
        Input DataArray.
    method:
        Reduction method (e.g. "pca").
    dim:
        Dimension to reduce across.
    n:
        Number of components to keep.
    stack:
        Optional dims to stack before reduction.
    unstack:
        If True, unstack stacked dims in the output.
    """
    method = method.lower()
    if method != "pca":
        raise ValueError(f"Unsupported reduction method {method!r}.")

    if stack is None:
        stack = tuple(d for d in da.dims if d != dim)
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims.")
    if dim in stack:
        raise ValueError("stack dims must not include the reduction dim.")
    if not stack:
        raise ValueError("Need at least one non-reduced dim to stack.")

    stacked_dim = "_stack"
    da_stack = da.stack({stacked_dim: stack})
    da_stack = da_stack.transpose(stacked_dim, dim)

    X = np.asarray(da_stack.data)
    if np.isnan(X).any():
        raise ValueError("PCA does not support NaN values.")

    model = PCA(n_components=n)
    scores = model.fit_transform(X)

    out = xr.DataArray(
        scores,
        dims=(stacked_dim, "component"),
        coords={stacked_dim: da_stack[stacked_dim], "component": np.arange(n)},
        name=da.name,
        attrs=dict(da.attrs),
    ).transpose("component", stacked_dim)

    if unstack:
        out = out.unstack(stacked_dim)

    out = preserve_coords(da, out)
    return out
