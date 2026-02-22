"""Alignment operations.

This module contains the core implementation for aligning spikes/continuous
signals to event times.

The public entry point is `align`, which is used by the `.ephys.align` accessor.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ..standards.conventions import C
from ..standards.validate import infer_kind, validate
from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)


class EphysAlignError(ValueError):
    """Raised when alignment inputs are invalid or unsupported."""

    pass


def _normalize_events(
    events: Union[xr.DataArray, Sequence[float], np.ndarray],
    *,
    to: str = "event",
) -> xr.DataArray:
    """
    Accept:
      - DataArray of event times dims (trial, event)
      - DataArray of events with dims (trial, event, bound)
      - 1D array-like of event times (trial,)
      - scalar event time
    Return a DataArray with dims (trial, event).
    """

    if isinstance(events, xr.DataArray):
        if "bound" in events.dims:
            if "start" not in events.coords.get("bound", []):
                raise EphysAlignError(
                    "events['bound'] must include 'start' when using "
                    "(trial, event, bound) representation."
                )
            events = events.sel(bound="start")
        return events
    if np.isscalar(events):
        times = np.asarray([events], dtype=float)
    else:
        try:
            times = np.asarray(events, dtype=float)
        except Exception as e:
            raise TypeError(
                "events must be an xarray DataArray or array-like event times."
            ) from e

    if times.ndim != 1:
        raise EphysAlignError(
            "array-like events must be 1D event times with shape (trial,). "
            f"Got shape {times.shape}."
        )

    event_times = xr.DataArray(
        times[:, None],
        dims=(C.trial, C.event),
        coords={C.trial: np.arange(times.shape[0]), C.event: [to]},
    )
    return event_times


def _get_event_times(events: xr.DataArray, to: str) -> xr.DataArray:
    """Select per-trial event times for a specific event label.

    Parameters
    ----------
    events:
        Event-times DataArray with an `event` dimension.
    to:
        Event label to select.

    Returns
    -------
    xr.DataArray
        Per-trial event times with dims (trial,).
    """
    if C.event not in events.dims:
        raise EphysAlignError(
            f"events must have a '{C.event}' dimension. Got dims {events.dims}."
        )
    try:
        return events.sel({C.event: to})
    except KeyError as e:
        raise EphysAlignError(
            f"Could not find event label {to!r} in "
            "event coordinates."
        ) from e


def _finalize_output(
    da: xr.DataArray, out: xr.DataArray, window: Tuple[float, float]
) -> xr.DataArray:
    """Apply common output processing: attrs, coords, valid_intervals."""
    out.attrs = dict(da.attrs)
    out = preserve_coords(da, out)
    out.attrs[C.attr_valid_intervals] = [window]
    return out


def _align_continuous(
    da: xr.DataArray, t0: xr.DataArray, window: Tuple[float, float]
) -> xr.DataArray:
    """Align continuous/binned data to event times."""
    tmin, tmax = window

    if C.trial in da.dims:
        # Trial-wise data: assume time coords are already trial-relative.
        # Slice to window.
        time = da[C.time].values
        mask = (time >= tmin) & (time <= tmax)
        return da.sel({C.time: time[mask]})

    # No trial dim: single continuous trace with scalar event time
    if t0.size != 1:
        raise EphysAlignError(
            "Aligning a no-trial DataArray requires a single event time "
            "(trial size 1)."
        )
    t0_scalar = float(t0.values.flat[0])
    aligned_time = da[C.time].values - t0_scalar
    return da.assign_coords({C.time: aligned_time}).sel(
        {C.time: slice(tmin, tmax)}
    )


def _align_unit(
    all_spikes: object,
    t0_vals: np.ndarray,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    """Align spikes for a single unit across all trials.

    Returns an object array of shape (n_trials,) with aligned spike times.
    """
    n_trials = len(t0_vals)
    out = np.empty(n_trials, dtype=object)

    if all_spikes is None:
        for i in range(n_trials):
            out[i] = np.array([], dtype=float)
        return out

    arr = np.asarray(all_spikes, dtype=float)
    if arr.size == 0:
        for i in range(n_trials):
            out[i] = np.array([], dtype=float)
        return out

    # Vectorized searchsorted: find all boundaries at once
    lo_bounds = t0_vals + tmin
    hi_bounds = t0_vals + tmax
    all_lo = np.searchsorted(arr, lo_bounds)
    all_hi = np.searchsorted(arr, hi_bounds)

    # Slice and shift for each trial (can't vectorize due to ragged output)
    for i in range(n_trials):
        out[i] = arr[all_lo[i]: all_hi[i]] - t0_vals[i]  # fmt: skip

    return out


def _align_ragged(
    da: xr.DataArray,
    t0: xr.DataArray,
    window: Tuple[float, float],
) -> xr.DataArray:
    """Align ragged spike data to event times.

    Input da must be session-time ragged spikes with shape (1, n_units), where
    da[0, u] contains all spikes for unit u. Output has shape
    (n_trials, n_units) with spikes sliced around each event time.

    Uses vectorized searchsorted for efficient boundary finding across all
    trials simultaneously.
    """
    tmin, tmax = window
    t0_vals = t0.values.astype(float)
    n_input_trials = da.sizes[C.trial]
    if n_input_trials != 1:
        raise EphysAlignError(
            "Ragged spikes align expects session-time input with a single "
            f"'{C.trial}' entry (size=1). Got size={n_input_trials}. "
            "If spikes are already trialized, skip align and use restrict/bin/"
            "reduce operations directly."
        )
    n_units = da.sizes[C.unit]

    # Process each unit with vectorized searchsorted
    results = [
        _align_unit(
            da.isel({C.trial: 0, C.unit: u}).item(), t0_vals, tmin, tmax
        )
        for u in range(n_units)
    ]

    # Stack results: each is (n_trials,), stack to (n_trials, n_units)
    out_data = np.column_stack(results)

    return xr.DataArray(
        out_data,
        dims=(C.trial, C.unit),
        coords={C.trial: t0[C.trial], C.unit: da[C.unit]},
        name=da.name,
    )


def align(
    data: DataInput,
    *,
    events: Union[xr.DataArray, Sequence[float], np.ndarray],
    window: Tuple[float, float],
    to: Optional[str] = None,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """
    Align data to an event and extract a time window around it.

    - Continuous/binned: returns da with time shifted so event is at 0,
      sliced to window.
    - Ragged spikes: expects session-time ragged spikes with trial size 1;
      returns trialized spikes by subtracting each event time and filtering
      to window.

    Notes
    -----
    ``to`` is optional for array-like/ndarray ``events`` inputs. For xarray
    ``events`` objects, ``to`` is required to select an event label.
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    if to is None and isinstance(events, xr.DataArray):
        raise EphysAlignError(
            "to is required when events is an xarray DataArray. "
            "For array-like event times, to may be omitted."
        )
    to_use = to if to is not None else "__event__"

    validate(da)
    events = _normalize_events(events, to=to_use)
    t0 = _get_event_times(events, to=to_use)

    tmin, tmax = window
    if tmin >= tmax:
        raise EphysAlignError(
            f"window must be (min,max) with min < max, got {window}."
        )

    kind = infer_kind(da)

    if kind in ("continuous", "binned"):
        if C.time not in da.dims:
            raise EphysAlignError(
                "Continuous/binned alignment requires a time dimension."
            )
        out = _align_continuous(da, t0, window)

    elif kind == "spikes_ragged":
        if C.trial not in da.dims:
            raise EphysAlignError(
                "Ragged spikes alignment requires a 'trial' dimension."
            )
        out = _align_ragged(da, t0, window)

    else:
        raise EphysAlignError(f"Unsupported kind {kind!r} for align.")

    out = _finalize_output(da, out, window)
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )
