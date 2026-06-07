"""Binning operations for ragged spikes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from ..standards.conventions import C
from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BinSize = Union[float, str]


@dataclass(frozen=True)
class PSTHBinWidthResult:
    """Result of :func:`select_psth_bin_width`."""

    best_width_s: float
    best_cost: float
    candidate_widths_s: FloatArray
    candidate_costs: FloatArray
    best_multiplier: int
    refined: bool


def bin(  # noqa: C901
    data: DataInput,
    dt: Optional[BinSize] = None,
    window: Optional[Tuple[float, float]] = None,
    output: str = "rate",
    time_unit: str = "s",
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
    bin_size: Optional[BinSize] = None,
) -> Union[xr.DataArray, object]:
    """
    Bin ragged spikes into a dense representation.

    Parameters
    ----------
    data:
        Ragged spike DataArray/list or object NumPy array.
    dt:
        Bin width in seconds, or ``"auto"`` to select one with the
        Shimazaki-Shinomoto method.
    bin_size:
        Alias for ``dt``. Specify only one of ``dt`` or ``bin_size``.
    window:
        Optional (tmin, tmax) window to bin.
    output:
        Output type, typically "rate" or "count".
    time_unit:
        Unit for time values, recorded in attrs.
    dims:
        Optional dimension names used when ``data`` is a dense NumPy array.
        This is ignored for ragged list inputs.
    coords:
        Optional coordinate mapping used when constructing a DataArray from
        dense NumPy input.
    return_type:
        Output type policy: ``"auto"``, ``"xarray"``, or ``"numpy"``.
        ``"auto"`` mirrors the input style (xarray in/xarray out,
        list/NumPy ragged in/list or NumPy out).

    Returns
    -------
    xr.DataArray or object
        Binned dense output. For ``return_type="numpy"``, returns a NumPy
        array with the same dense shape as the xarray result values.
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    if da.dtype != object:
        raise ValueError("bin expects ragged spikes with dtype=object.")

    allowed_dims = {C.trial, C.unit}
    if not set(da.dims).issubset(allowed_dims):
        raise ValueError(
            "bin expects only 'trial'/'unit' dimensions for ragged spikes."
        )

    if window is None:
        window = da.attrs.get(C.attr_valid_intervals, None)
        if window is not None:
            tmin, tmax = window[0]  # Use first valid interval
        else:
            tmin, tmax = _infer_tlim(da)
    else:
        tmin, tmax = window

    if tmin >= tmax:
        raise ValueError(
            f"window must be (min,max) with min < max, got {window}."
        )

    dt = _resolve_bin_size(da, dt=dt, bin_size=bin_size, window=(tmin, tmax))

    ratio = (tmax - tmin) / dt
    if np.isclose(ratio, round(ratio)):
        n_bins = int(round(ratio))
    else:
        n_bins = int(np.ceil(ratio))
    if n_bins < 1:
        n_bins = 1
    edges = tmin + dt * np.arange(n_bins + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    output = output.lower()
    if output not in ("rate", "count"):
        raise ValueError("output must be 'rate' or 'count'.")

    has_trial = C.trial in da.dims
    has_unit = C.unit in da.dims

    if has_trial and has_unit:
        # Preserve the original (trial, unit) vs (unit, trial) ordering.
        if da.dims.index(C.trial) < da.dims.index(C.unit):
            data = _bin_ragged(da, edges, (C.trial, C.unit))
            dims = (C.trial, C.unit, C.time)
            coords = {
                C.trial: da[C.trial],
                C.unit: da[C.unit],
                C.time: centers,
            }
        else:
            data = _bin_ragged(da, edges, (C.unit, C.trial))
            dims = (C.unit, C.trial, C.time)
            coords = {
                C.unit: da[C.unit],
                C.trial: da[C.trial],
                C.time: centers,
            }
    elif has_unit:
        data = _bin_ragged(da, edges, (C.unit,))
        dims = (C.unit, C.time)
        coords = {C.unit: da[C.unit], C.time: centers}
    elif has_trial:
        data = _bin_ragged(da, edges, (C.trial,))
        dims = (C.trial, C.time)
        coords = {C.trial: da[C.trial], C.time: centers}
    else:
        raise ValueError(
            "bin requires at least a 'trial' or 'unit' dimension."
        )

    if output == "rate":
        data = data / dt

    out = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        name=da.name,
        attrs=dict(da.attrs),
    )
    out = preserve_coords(da, out)
    out.attrs[C.attr_kind] = "binned"
    out.attrs[C.attr_time_unit] = time_unit
    if window is not None:
        out.attrs[C.attr_valid_intervals] = [window]
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )


def _resolve_bin_size(
    da: xr.DataArray,
    *,
    dt: Optional[BinSize],
    bin_size: Optional[BinSize],
    window: Tuple[float, float],
) -> float:
    """Resolve ``dt``/``bin_size`` aliases and optional auto-selection."""
    if dt is not None and bin_size is not None:
        raise ValueError("Specify only one of dt or bin_size.")
    size = bin_size if bin_size is not None else dt
    if size is None:
        raise ValueError("dt or bin_size is required.")

    if isinstance(size, str):
        if size.lower() != "auto":
            raise ValueError("bin size string must be 'auto'.")
        trials = _flatten_ragged_entries(da)
        result = select_psth_bin_width(
            trials,
            t_start=float(window[0]),
            t_stop=float(window[1]),
        )
        return result.best_width_s

    size_float = float(size)
    if size_float <= 0 or not np.isfinite(size_float):
        raise ValueError("dt must be a positive number of seconds.")
    return size_float


def _flatten_ragged_entries(da: xr.DataArray) -> list[np.ndarray]:
    """Return all ragged spike entries as 1D float arrays."""
    return [_ensure_1d_float_array(x) for x in da.data.ravel()]


def _infer_tlim(da: xr.DataArray) -> Tuple[float, float]:
    """Infer (tmin, tmax) from ragged spikes."""
    tmin = np.inf
    tmax = -np.inf
    for x in da.data.ravel():
        if x is None:
            continue
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            continue
        tmin = min(tmin, float(arr.min()))
        tmax = max(tmax, float(arr.max()))
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        raise ValueError("Cannot infer tlim from empty spike arrays.")
    if tmin == tmax:
        tmax = tmin + 1e-6
    return tmin, tmax


def _bin_ragged(
    da: xr.DataArray, edges: np.ndarray, ordered_dims: tuple
) -> np.ndarray:
    """Bin ragged spikes, iterating over all index combinations of ordered_dims.

    Parameters
    ----------
    da:
        Ragged spike DataArray.
    edges:
        Histogram bin edges.
    ordered_dims:
        Dimension names to iterate over, in the desired output order.
        The output shape is (*[da.sizes[d] for d in ordered_dims], n_bins).
    """
    da_ordered = da.transpose(*ordered_dims)
    shape = da_ordered.shape
    n_bins = len(edges) - 1
    out = np.zeros(shape + (n_bins,), dtype=float)
    for idx in np.ndindex(*shape):
        arr = _ensure_1d_float_array(da_ordered.data[idx])
        if arr.size:
            # np.searchsorted is faster than np.histogram here because the
            # edges are already known and spike times are sorted ascending.
            # np.diff(searchsorted(arr, edges)) gives the same bin counts
            # as np.histogram(arr, bins=edges) for sorted input.
            out[idx] = np.diff(np.searchsorted(arr, edges))
    return out


def _ensure_1d_float_array(x: object) -> np.ndarray:
    """Coerce an entry to a 1D float array of spike times."""
    if x is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Ragged spikes entries must be 1D arrays.")
    return arr


def _as_float_array(
    x: Union[Sequence[float], NDArray[np.floating]],
) -> FloatArray:
    """Coerce input to a 1D float64 array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr


def _validate_trials(
    spike_times_by_trial: Sequence[
        Union[Sequence[float], NDArray[np.floating]]
    ],
    t_start: float,
    t_stop: float,
) -> list[FloatArray]:
    """Validate window bounds and clip each trial's spikes to it."""
    if not np.isfinite(t_start) or not np.isfinite(t_stop):
        raise ValueError("t_start and t_stop must be finite.")
    if t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start.")
    if len(spike_times_by_trial) == 0:
        raise ValueError("Need at least one trial.")

    trials: list[FloatArray] = []
    for st in spike_times_by_trial:
        arr = _as_float_array(st)
        arr = arr[(arr >= t_start) & (arr < t_stop)]
        trials.append(arr)

    return trials


def _build_base_counts(
    spike_times_by_trial: Sequence[FloatArray],
    t_start: float,
    t_stop: float,
    base_bin_width_s: float,
) -> IntArray:
    """Histogram each trial at the finest base bin width."""
    if not np.isfinite(base_bin_width_s) or base_bin_width_s <= 0:
        raise ValueError("base_bin_width_s must be positive and finite.")

    duration = t_stop - t_start
    n_base_bins = int(np.floor(duration / base_bin_width_s))
    if n_base_bins < 1:
        raise ValueError(
            "Base bin width is too large for the requested window."
        )

    effective_stop = t_start + n_base_bins * base_bin_width_s
    edges = (
        t_start
        + np.arange(n_base_bins + 1, dtype=np.float64) * base_bin_width_s
    )

    base_counts = np.zeros(
        (len(spike_times_by_trial), n_base_bins), dtype=np.int64
    )
    for i, st in enumerate(spike_times_by_trial):
        st = st[(st >= t_start) & (st < effective_stop)]
        counts, _ = np.histogram(st, bins=edges)
        base_counts[i, :] = counts.astype(np.int64, copy=False)

    return base_counts


def _candidate_multipliers(
    base_bin_width_s: float,
    min_width_s: float,
    max_width_s: float,
    *,
    n_grid: int = 60,
    log_spacing: bool = True,
) -> NDArray[np.int64]:
    """Build integer multipliers of the base width to evaluate."""
    if min_width_s < base_bin_width_s:
        raise ValueError("min_width_s must be >= base_bin_width_s.")
    if max_width_s < min_width_s:
        raise ValueError("max_width_s must be >= min_width_s.")
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2.")

    min_m = int(np.ceil(min_width_s / base_bin_width_s))
    max_m = int(np.floor(max_width_s / base_bin_width_s))
    if max_m < min_m:
        raise ValueError(
            "No candidate widths are compatible with base_bin_width_s."
        )

    if log_spacing:
        raw = np.geomspace(min_m, max_m, num=n_grid)
    else:
        raw = np.linspace(min_m, max_m, num=n_grid)

    m = np.unique(np.clip(np.rint(raw).astype(np.int64), min_m, max_m))
    if m.size == 0:
        raise ValueError("No candidate widths produced.")
    return m


def _aggregate_counts_for_multiplier(
    base_counts: IntArray, m: int
) -> IntArray:
    """Sum base-bin counts in non-overlapping groups of ``m`` bins."""
    if m < 1:
        raise ValueError("m must be >= 1.")

    n_trials, n_base_bins = base_counts.shape
    n_use = (n_base_bins // m) * m
    if n_use < m:
        raise ValueError(
            "Candidate width is too large for the available window."
        )

    reshaped = base_counts[:, :n_use].reshape(n_trials, n_use // m, m)
    result: IntArray = reshaped.sum(axis=2, dtype=np.int64)
    return result


def _shimazaki_shinomoto_cost(
    binned_counts_by_trial: IntArray,
    bin_width_s: float,
) -> float:
    """Compute the Shimazaki-Shinomoto SSMD cost for a bin width."""
    if bin_width_s <= 0 or not np.isfinite(bin_width_s):
        return float("inf")

    n_trials, n_bins = binned_counts_by_trial.shape
    if n_trials < 1 or n_bins < 1:
        return float("inf")

    k = binned_counts_by_trial.sum(axis=0, dtype=np.int64).astype(np.float64)
    kbar = float(k.mean())
    v = float(((k - kbar) ** 2).mean())

    return (2.0 * kbar - v) / ((n_trials * bin_width_s) ** 2)


def _evaluate_multipliers(
    base_counts: IntArray,
    base_bin_width_s: float,
    multipliers: NDArray[np.int64],
) -> FloatArray:
    """Compute the SS cost for every candidate bin-width multiplier."""
    costs = np.empty(multipliers.shape, dtype=np.float64)
    for i, m in enumerate(multipliers):
        agg = _aggregate_counts_for_multiplier(base_counts, int(m))
        costs[i] = _shimazaki_shinomoto_cost(agg, float(m) * base_bin_width_s)
    return costs


def select_psth_bin_width(
    spike_times_by_trial: Sequence[
        Union[Sequence[float], NDArray[np.floating]]
    ],
    *,
    t_start: float,
    t_stop: float,
    base_bin_width_s: float = 0.001,
    min_width_s: Optional[float] = None,
    max_width_s: Optional[float] = None,
    n_grid: int = 60,
    log_spacing: bool = True,
    refine: bool = True,
    local_refine_radius: int = 5,
) -> PSTHBinWidthResult:
    """Select PSTH bin width via the Shimazaki-Shinomoto method.

    Performs a fine-resolution prebinning pass, then searches over integer
    multiples of ``base_bin_width_s`` to minimise the SSMD cost.
    """
    trials = _validate_trials(spike_times_by_trial, t_start, t_stop)
    duration = t_stop - t_start

    if min_width_s is None:
        min_width_s = base_bin_width_s
    if max_width_s is None:
        max_width_s = min(0.25 * duration, 0.2)

    base_counts = _build_base_counts(trials, t_start, t_stop, base_bin_width_s)

    coarse_multipliers = _candidate_multipliers(
        base_bin_width_s=base_bin_width_s,
        min_width_s=min_width_s,
        max_width_s=max_width_s,
        n_grid=n_grid,
        log_spacing=log_spacing,
    )
    coarse_costs = _evaluate_multipliers(
        base_counts, base_bin_width_s, coarse_multipliers
    )

    best_idx = int(np.argmin(coarse_costs))
    best_multiplier = int(coarse_multipliers[best_idx])
    best_cost = float(coarse_costs[best_idx])
    refined = False

    if refine and local_refine_radius > 0:
        min_m = int(np.ceil(min_width_s / base_bin_width_s))
        max_m = int(np.floor(max_width_s / base_bin_width_s))

        local_multipliers = np.arange(
            max(min_m, best_multiplier - local_refine_radius),
            min(max_m, best_multiplier + local_refine_radius) + 1,
            dtype=np.int64,
        )

        local_costs = _evaluate_multipliers(
            base_counts, base_bin_width_s, local_multipliers
        )
        local_best_idx = int(np.argmin(local_costs))
        local_best_multiplier = int(local_multipliers[local_best_idx])
        local_best_cost = float(local_costs[local_best_idx])

        if local_best_cost < best_cost:
            best_multiplier = local_best_multiplier
            best_cost = local_best_cost
            refined = True

    return PSTHBinWidthResult(
        best_width_s=best_multiplier * base_bin_width_s,
        best_cost=best_cost,
        candidate_widths_s=coarse_multipliers.astype(np.float64)
        * base_bin_width_s,
        candidate_costs=coarse_costs,
        best_multiplier=best_multiplier,
        refined=refined,
    )
