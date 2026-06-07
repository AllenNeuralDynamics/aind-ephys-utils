"""Validation utilities for canonical ephys xarray objects.

`aind_ephys_utils` expects a small set of canonical DataArray "kinds" (ragged
spikes, binned spikes, continuous signals, events). This module provides helpers
to infer and validate those conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from .conventions import C


# ----------------------------
# Error types
# ----------------------------
class EphysValidationError(ValueError):
    """Raised when an object does not conform to ephysexplorer conventions."""


@dataclass(frozen=True)
class ValidationResult:
    """Structured result returned by `validate`.

    Attributes
    ----------
    ok:
        True if validation succeeded.
    kind:
        Inferred kind (e.g. "spikes_ragged", "binned", "continuous", "events").
    message:
        Human-readable status string.
    """

    ok: bool
    kind: (
        str  # "spikes_ragged" | "continuous" | "binned" | "events" | "unknown"
    )
    message: str = ""


# ----------------------------
# Predicates
# ----------------------------
def is_ragged_spikes(da: xr.DataArray) -> bool:
    """
    Ragged spikes convention:
      - dims include (trial, unit)
      - dtype is object
      - each element is a 1D array-like of spike times (seconds)
    """
    if da.dtype != object:
        return False
    if C.trial not in da.dims or C.unit not in da.dims:
        return False
    return True


def is_continuous(da: xr.DataArray) -> bool:
    """
    Continuous convention:
      - has a numeric dtype (int/float/complex)
      - has a time dimension
      - often has trial plus one of unit/channel/feature/etc.
    """
    if C.time not in da.dims:
        return False
    if da.dtype == object:
        return False
    # Allow integer/float/bool; exclude strings/objects
    try:
        np.asarray(da.data).dtype.kind
    except (TypeError, AttributeError):
        return False
    return True


def is_binned_spikes(da: xr.DataArray) -> bool:
    """
    Binned spikes convention:
      - numeric
      - dims include (trial, unit, time)
    """
    if da.dtype == object:
        return False
    required = {C.trial, C.unit, C.time}
    return required.issubset(set(da.dims))


def is_events(da: xr.DataArray) -> bool:
    """
    Events convention:
      - dims are (trial, event, bound)
      - dtype is numeric
      - bound coordinate has ["start", "end"]
      - each event has start/end times (instantaneous events have start==end)
    """
    if da.dtype == object:
        return False
    required = {C.trial, C.event, "bound"}
    if not required.issubset(set(da.dims)):
        return False
    # Check for bound coordinate with start/end
    if "bound" in da.coords:
        bounds = da.coords["bound"].values
        if len(bounds) == 2 and set(bounds) == {"start", "end"}:
            return True
    return False


def infer_kind(da: xr.DataArray) -> str:
    """Infer the canonical kind of a DataArray.

    Parameters
    ----------
    da:
        DataArray to classify.

    Returns
    -------
    str
        One of: "spikes_ragged", "binned", "continuous", "events", or "unknown".
    """
    if is_events(da):
        return "events"
    if is_ragged_spikes(da):
        return "spikes_ragged"
    if is_binned_spikes(da):
        return "binned"
    if is_continuous(da):
        return "continuous"
    return "unknown"


# ----------------------------
# Validation helpers
# ----------------------------
def _require_dims(
    da: xr.DataArray, dims: Sequence[str], *, context: str
) -> None:
    """Require that a DataArray contains specific dimensions."""
    missing = [d for d in dims if d not in da.dims]
    if missing:
        raise EphysValidationError(
            f"{context}: expected dims {tuple(dims)} but missing {missing}. "
            f"Got dims {da.dims}."
        )


def _require_time_coord_if_has_time_dim(da: xr.DataArray) -> None:
    """Require a numeric `time` coordinate when a `time` dimension is present."""
    if C.time in da.dims and C.time not in da.coords:
        # xarray can have a dim without an explicit coord; but analysis suffers.
        raise EphysValidationError(
            f"Expected a '{C.time}' coordinate when '{C.time}' is a dimension. "
            f"Tip: assign time values with da = da.assign_coords(time=...)."
        )


def _validate_ragged_contents(
    da: xr.DataArray, *, max_checks: int = 50
) -> None:
    """
    Lightweight sanity check: sample up to max_checks entries and ensure each is
    array-like 1D, numeric, monotonic optional.
    """
    flat = da.data.ravel()
    n = min(len(flat), max_checks)

    for i in range(n):
        x = flat[i]
        if x is None:
            continue
        arr = np.asarray(x)
        if arr.ndim != 1:
            msg = (
                "Ragged spikes entries must be 1D arrays of spike times. "
                f"Found entry with shape {arr.shape}."
            )
            raise EphysValidationError(msg)
        if arr.size == 0:
            continue
        if arr.dtype.kind not in ("i", "u", "f"):  # ints or floats
            raise EphysValidationError(
                "Ragged spikes entries must be numeric (int/float) spike times."
            )


# ----------------------------
# Public validate API
# ----------------------------
def validate(
    da: xr.DataArray,
    *,
    kind: Optional[str] = None,
    require_time_coord: bool = True,
    check_ragged_contents: bool = True,
) -> ValidationResult:
    """
    Validate that a DataArray conforms to the conventions expected by .ephys.

    Parameters
    ----------
    da:
        The DataArray to validate.
    kind:
        If provided, require that the DataArray matches this inferred kind.
        One of: "spikes_ragged", "continuous", "binned", "events".
    require_time_coord:
        If True, require a 'time' coordinate whenever there is a time dimension.
    check_ragged_contents:
        If True, sample-check ragged spike contents for basic shape/dtype sanity.

    Returns
    -------
    ValidationResult
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError(
            f"validate expects an xarray.DataArray, got {type(da)!r}."
        )

    inferred = infer_kind(da)

    if kind is not None and inferred != kind:
        raise EphysValidationError(
            f"Expected kind={kind!r} but inferred {inferred!r}. "
            f"dims={da.dims}, dtype={da.dtype!r}."
        )

    # Minimal structural checks
    if inferred == "events":
        _require_dims(da, (C.trial, C.event, "bound"), context="Events")
    elif inferred == "spikes_ragged":
        _require_dims(da, (C.trial, C.unit), context="Ragged spikes")
        if check_ragged_contents:
            _validate_ragged_contents(da)
    elif inferred == "binned":
        _require_dims(da, (C.trial, C.unit, C.time), context="Binned spikes")
        if require_time_coord:
            _require_time_coord_if_has_time_dim(da)
    elif inferred == "continuous":
        _require_dims(da, (C.time,), context="Continuous signal")
        if require_time_coord:
            _require_time_coord_if_has_time_dim(da)
    else:
        msg = (
            "Could not infer DataArray kind. "
            "Expected events (trial/event/bound), "
            "ragged spikes (object dtype with dims trial/unit), "
            "binned spikes (trial/unit/time), or continuous (numeric with time). "
            f"Got dims={da.dims}, dtype={da.dtype!r}."
        )
        raise EphysValidationError(msg)

    return ValidationResult(ok=True, kind=inferred, message="ok")
