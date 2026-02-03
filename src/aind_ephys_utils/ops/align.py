"""Alignment operations.

This module contains the core implementation for aligning spikes/continuous
signals to event times.

The public entry point is `align`, which is used by the `.ephys.align` accessor.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import xarray as xr

from ..standards.conventions import C
from ..standards.validate import infer_kind, validate
from .utils import preserve_coords


class EphysAlignError(ValueError):
    """Raised when alignment inputs are invalid or unsupported."""

    pass


def _normalize_events(events: Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    """
    Accept:
      - Dataset with var "t" dims (trial, event)
      - DataArray of event times dims (trial, event)
      - DataArray of events with dims (trial, event, bound)
    Return a Dataset with events["t"] always present.
    """

    if isinstance(events, xr.DataArray):
        if "bound" in events.dims:
            if "start" not in events.coords.get("bound", []):
                raise EphysAlignError(
                    "events['bound'] must include 'start' when using "
                    "(trial, event, bound) representation."
                )
            events = events.sel(bound="start")
        return xr.Dataset({C.event_time_var: events})
    if isinstance(events, xr.Dataset):
        if C.event_time_var not in events:
            raise EphysAlignError(
                f"events must contain variable {C.event_time_var!r}."
            )
        return events
    raise TypeError(
        f"events must be an xarray Dataset or DataArray, got {type(events)!r}."
    )


def _get_event_times(events: xr.Dataset, to: str) -> xr.DataArray:
    """Select per-trial event times for a specific event label.

    Parameters
    ----------
    events:
        Events dataset containing `events[C.event_time_var]` with an `event`
        dimension.
    to:
        Event label to select.

    Returns
    -------
    xr.DataArray
        Per-trial event times with dims (trial,).
    """
    t = events[C.event_time_var]
    if C.event not in t.dims:
        raise EphysAlignError(
            f"events['{C.event_time_var}'] must have a '{C.event}' dimension. "
            f"Got dims {t.dims}."
        )
    try:
        return t.sel({C.event: to})
    except KeyError as e:
        raise EphysAlignError(
            f"Could not find event label {to!r} in "
            f"events['{C.event_time_var}']."
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
        out[i] = arr[all_lo[i]: all_hi[i]] - t0_vals[i]

    return out


def _align_ragged(
    da: xr.DataArray,
    t0: xr.DataArray,
    window: Tuple[float, float],
) -> xr.DataArray:
    """Align ragged spike data to event times.

    Input da has shape (1, n_units) where da[0, u] contains all spikes for
    unit u. Output has shape (n_trials, n_units) with spikes sliced around
    each event time.

    Uses vectorized searchsorted for efficient boundary finding across all
    trials simultaneously.
    """
    tmin, tmax = window
    t0_vals = t0.values.astype(float)
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
    da: xr.DataArray,
    *,
    events: Union[xr.Dataset, xr.DataArray],
    to: str,
    window: Tuple[float, float],
) -> xr.DataArray:
    """
    Align a DataArray to an event and extract a time window around it.

    - Continuous/binned: returns da with time shifted so event is at 0,
      sliced to window.
    - Ragged spikes: subtract event time per trial, filter to window.
    """
    validate(da)
    events = _normalize_events(events)
    t0 = _get_event_times(events, to=to)

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

    return _finalize_output(da, out, window)
