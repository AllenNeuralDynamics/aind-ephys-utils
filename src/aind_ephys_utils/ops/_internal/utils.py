"""Utility helpers for ops implementations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ...standards.conventions import C

RaggedSession = Sequence[object]
RaggedTrial = Sequence[Sequence[object]]
DataInput = Union[xr.DataArray, np.ndarray, RaggedSession, RaggedTrial]


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


def _coerce_spike_array(x: object) -> np.ndarray:
    """Coerce a ragged spike entry to a 1D float array."""
    if x is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"Ragged spike entries must be 1D arrays, got shape {arr.shape}."
        )
    return arr


def infer_dense_dims(ndim: int) -> Tuple[str, ...]:
    """Infer canonical dimension names for dense NumPy arrays."""
    if ndim == 1:
        return (C.time,)
    if ndim == 2:
        return (C.trial, C.time)
    if ndim == 3:
        return (C.trial, C.unit, C.time)
    return tuple(f"dim_{i}" for i in range(ndim))


def is_ragged_session_list(data: object) -> bool:
    """Return True if data is session-aligned ragged spikes (unit -> spikes)."""
    if isinstance(data, np.ndarray):
        return False
    if not isinstance(data, (list, tuple)):
        return False
    if len(data) == 0:
        return True
    first = data[0]
    if isinstance(first, (list, tuple)):
        return False
    for x in data:
        _ = _coerce_spike_array(x)
    return True


def is_ragged_trial_list(data: object) -> bool:
    """Return True if data is trial-aligned ragged spikes (trial -> unit)."""
    if isinstance(data, np.ndarray):
        return False
    if not isinstance(data, (list, tuple)):
        return False
    if len(data) == 0:
        return True
    first = data[0]
    if not isinstance(first, (list, tuple)):
        return False
    n_units: Optional[int] = None
    for trial in data:
        if not isinstance(trial, (list, tuple)):
            return False
        if n_units is None:
            n_units = len(trial)
        elif len(trial) != n_units:
            raise ValueError("All trials must have the same number of units.")
        for x in trial:
            _ = _coerce_spike_array(x)
    return True


def ragged_session_to_dataarray(
    data: RaggedSession,
    *,
    coords: Optional[Dict[str, object]] = None,
) -> xr.DataArray:
    """Convert session-aligned ragged list to DataArray with dims (trial, unit)."""
    n_units = len(data)
    arr = np.empty((1, n_units), dtype=object)
    for u in range(n_units):
        arr[0, u] = _coerce_spike_array(data[u])

    da_coords: Dict[str, object] = {
        C.trial: np.asarray([0]),
        C.unit: np.arange(n_units),
    }
    if coords:
        da_coords.update(coords)
    return xr.DataArray(arr, dims=(C.trial, C.unit), coords=da_coords)


def ragged_trial_to_dataarray(
    data: RaggedTrial,
    *,
    coords: Optional[Dict[str, object]] = None,
) -> xr.DataArray:
    """Convert trial-aligned ragged list to DataArray with dims (trial, unit)."""
    n_trials = len(data)
    n_units = len(data[0]) if n_trials > 0 else 0
    arr = np.empty((n_trials, n_units), dtype=object)
    for t in range(n_trials):
        trial = data[t]
        if len(trial) != n_units:
            raise ValueError("All trials must have the same number of units.")
        for u in range(n_units):
            arr[t, u] = _coerce_spike_array(trial[u])

    da_coords: Dict[str, object] = {
        C.trial: np.arange(n_trials),
        C.unit: np.arange(n_units),
    }
    if coords:
        da_coords.update(coords)
    return xr.DataArray(arr, dims=(C.trial, C.unit), coords=da_coords)


def ragged_dataarray_to_session_list(da: xr.DataArray) -> list[np.ndarray]:
    """Convert ragged DataArray to session-aligned ragged list."""
    da_u = da.transpose(C.trial, C.unit)
    if da_u.sizes.get(C.trial, 0) != 1:
        raise ValueError(
            "Expected trial dimension of size 1 for session ragged."
        )
    out: list[np.ndarray] = []
    for u in range(da_u.sizes[C.unit]):
        out.append(_coerce_spike_array(da_u.data[0, u]))
    return out


def ragged_dataarray_to_trial_list(da: xr.DataArray) -> list[list[np.ndarray]]:
    """Convert ragged DataArray to trial-aligned ragged list."""
    da_tu = da.transpose(C.trial, C.unit)
    out: list[list[np.ndarray]] = []
    for t in range(da_tu.sizes[C.trial]):
        row: list[np.ndarray] = []
        for u in range(da_tu.sizes[C.unit]):
            row.append(_coerce_spike_array(da_tu.data[t, u]))
        out.append(row)
    return out


def to_dataarray_input(
    data: DataInput,
    *,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
) -> Tuple[xr.DataArray, str]:
    """Normalize xarray/NumPy/ragged inputs to DataArray and remember input kind."""
    if isinstance(data, xr.DataArray):
        return data, "xarray"

    if is_ragged_trial_list(data):
        return ragged_trial_to_dataarray(data, coords=coords), "ragged_trial"

    if is_ragged_session_list(data):
        return (
            ragged_session_to_dataarray(data, coords=coords),
            "ragged_session",
        )

    arr = np.asarray(data)
    if arr.dtype == object and arr.ndim == 2:
        da_dims = tuple(dims) if dims is not None else (C.trial, C.unit)
        da_coords = dict(coords or {})
        if C.trial not in da_coords and C.trial in da_dims:
            da_coords[C.trial] = np.arange(arr.shape[da_dims.index(C.trial)])
        if C.unit not in da_coords and C.unit in da_dims:
            da_coords[C.unit] = np.arange(arr.shape[da_dims.index(C.unit)])
        return xr.DataArray(arr, dims=da_dims, coords=da_coords), "ndarray"

    da_dims = tuple(dims) if dims is not None else infer_dense_dims(arr.ndim)
    if len(da_dims) != arr.ndim:
        raise ValueError(
            f"dims length {len(da_dims)} does not match data ndim {arr.ndim}."
        )
    da_coords = dict(coords or {})
    for axis, d in enumerate(da_dims):
        if d not in da_coords:
            da_coords[d] = np.arange(arr.shape[axis])
    return xr.DataArray(arr, dims=da_dims, coords=da_coords), "ndarray"


def from_dataarray_output(  # noqa: C901
    out: xr.DataArray,
    *,
    input_kind: str,
    return_type: str = "auto",
) -> Union[xr.DataArray, np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    """Convert a DataArray output back to requested user-facing type."""
    if return_type not in ("auto", "xarray", "numpy"):
        raise ValueError(
            "return_type must be one of 'auto', 'xarray', 'numpy'."
        )

    if return_type == "xarray":
        return out

    if return_type == "numpy":
        if out.dtype == object:
            if out.sizes.get(C.trial, 0) == 1:
                return ragged_dataarray_to_session_list(out)
            return ragged_dataarray_to_trial_list(out)
        return np.asarray(out.values)

    # auto
    if input_kind == "xarray":
        return out
    if out.dtype == object:
        if input_kind == "ragged_session":
            if out.sizes.get(C.trial, 0) == 1:
                return ragged_dataarray_to_session_list(out)
            return ragged_dataarray_to_trial_list(out)
        if input_kind == "ragged_trial":
            return ragged_dataarray_to_trial_list(out)
        if out.sizes.get(C.trial, 0) == 1:
            return ragged_dataarray_to_session_list(out)
        return ragged_dataarray_to_trial_list(out)
    return np.asarray(out.values)
