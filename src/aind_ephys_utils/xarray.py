"""xarray-native wrappers for core operations.

DataArray in, DataArray out. Each function uses xarray's own operations
(broadcasting, ``.sel()``, ``.mean()``, ``apply_ufunc``) where possible,
falling back to :mod:`~aind_ephys_utils.core` only for operations without
xarray equivalents (binning, alignment, ragged restrict).

Fluent chaining works via xarray's built-in ``.pipe()``::

    from aind_ephys_utils.xarray import smooth, baseline, normalize, psth

    result = (
        da
        .pipe(align, events=events, to="stimulus", window=(-0.5, 1.0))
        .pipe(bin, dt=0.05)
        .pipe(smooth, sigma=0.03)
        .pipe(baseline, window=(-0.5, 0.0))
        .pipe(psth, labels=conditions)
    )
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from . import core
from .standards.conventions import C

__all__ = [
    "align",
    "baseline",
    "bin",
    "normalize",
    "psth",
    "restrict",
    "smooth",
]


# ------------------------------------------------------------------
# Internal helpers -- coord preservation and event handling
# ------------------------------------------------------------------


def _preserve_coords(src: xr.DataArray, out: xr.DataArray) -> xr.DataArray:
    """Carry over non-indexed coords whose dims are a subset of output dims.

    xarray drops non-indexed coordinates on arithmetic (e.g. ``da - mean``).
    This restores them after such operations.
    """
    for name, coord in src.coords.items():
        if name in out.coords:
            continue
        if set(coord.dims).issubset(set(out.dims)):
            out = out.assign_coords({name: coord})
    return out


def _normalize_events(events: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Normalize events input to a Dataset with a ``"t"`` variable.

    Accepts:
    - Dataset with var ``"t"``, dims ``(trial, event)``
    - DataArray of event times, dims ``(trial, event)``
    - DataArray with dims ``(trial, event, bound)`` -- selects ``"start"``
    """
    if isinstance(events, xr.DataArray):
        if "bound" in events.dims:
            if "start" not in events.coords.get("bound", []):
                raise ValueError(
                    "events['bound'] must include 'start' when using "
                    "(trial, event, bound) representation."
                )
            events = events.sel(bound="start")
        return xr.Dataset({C.event_time_var: events})
    if isinstance(events, xr.Dataset):
        if C.event_time_var not in events:
            raise ValueError(f"events must contain variable {C.event_time_var!r}.")
        return events
    raise TypeError(
        f"events must be an xarray Dataset or DataArray, got {type(events)!r}."
    )


def _get_event_times(events: xr.Dataset, to: str) -> xr.DataArray:
    """Select per-trial event times for a specific event label.

    Returns a DataArray with dims ``(trial,)``.
    """
    t = events[C.event_time_var]
    if C.event not in t.dims:
        raise ValueError(
            f"events['{C.event_time_var}'] must have a '{C.event}' dimension. "
            f"Got dims {t.dims}."
        )
    try:
        return t.sel({C.event: to})
    except KeyError as e:
        raise ValueError(
            f"Could not find event label {to!r} in events['{C.event_time_var}']."
        ) from e


# ------------------------------------------------------------------
# Internal helpers -- ragged operations
# ------------------------------------------------------------------


def _unpack_ragged(
    da: xr.DataArray,
) -> tuple[list[list[NDArray]], int, int, xr.DataArray]:
    """Extract ``list[list[ndarray]]`` from a ragged object DataArray.

    Transposes to ``(trial, unit)`` order.  Returns
    ``(spike_times, n_trials, n_units, da_transposed)``.
    """
    has_trial = C.trial in da.dims
    has_unit = C.unit in da.dims

    if has_trial and has_unit:
        da_work = da.transpose(C.trial, C.unit)
    elif has_trial:
        da_work = da.expand_dims(C.unit, axis=-1)
    elif has_unit:
        da_work = da.expand_dims(C.trial, axis=0)
    else:
        raise ValueError(
            "Ragged DataArray must have at least a 'trial' or 'unit' dimension."
        )

    n_trials, n_units = da_work.shape
    spike_times: list[list[NDArray]] = []
    for t in range(n_trials):
        trial_spikes: list[NDArray] = []
        for u in range(n_units):
            val = da_work.values[t, u]
            if val is None:
                trial_spikes.append(np.array([], dtype=np.float64))
            else:
                trial_spikes.append(np.asarray(val, dtype=np.float64))
        spike_times.append(trial_spikes)
    return spike_times, n_trials, n_units, da_work


def _repack_ragged(
    src_da: xr.DataArray,
    spike_times: list[list[NDArray]],
    n_trials: int,
    n_units: int,
    *,
    trial_coord: NDArray | xr.DataArray | None = None,
    unit_coord: NDArray | xr.DataArray | None = None,
) -> xr.DataArray:
    """Wrap ragged spike times back into an object-dtype DataArray."""
    out_data = np.empty((n_trials, n_units), dtype=object)
    for t in range(n_trials):
        for u in range(n_units):
            out_data[t, u] = spike_times[t][u]

    coords: dict = {}
    if trial_coord is not None:
        coords[C.trial] = trial_coord
    elif C.trial in src_da.coords:
        coords[C.trial] = src_da.coords[C.trial]
    if unit_coord is not None:
        coords[C.unit] = unit_coord
    elif C.unit in src_da.coords:
        coords[C.unit] = src_da.coords[C.unit]

    out = xr.DataArray(
        out_data,
        dims=(C.trial, C.unit),
        coords=coords,
        name=src_da.name,
        attrs=dict(src_da.attrs),
    )
    return _preserve_coords(src_da, out)


def _infer_tlim(da: xr.DataArray) -> tuple[float, float]:
    """Infer ``(tmin, tmax)`` from ragged spike data."""
    tmin = np.inf
    tmax = -np.inf
    for x in da.values.ravel():
        if x is None:
            continue
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            continue
        tmin = min(tmin, float(arr.min()))
        tmax = max(tmax, float(arr.max()))
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        raise ValueError("Cannot infer time limits from empty spike arrays.")
    if tmin == tmax:
        tmax = tmin + 1e-6
    return tmin, tmax


# ------------------------------------------------------------------
# Public wrappers -- xarray-native
# ------------------------------------------------------------------


def smooth(
    da: xr.DataArray,
    *,
    sigma: float | None = None,
    method: str = "gaussian",
    window: float | None = None,
    boundary: str = "reflect",
    dim: str = C.time,
) -> xr.DataArray:
    """Smooth a DataArray along a dimension.

    Uses ``xr.apply_ufunc`` to broadcast :func:`core.convolve_1d` over
    all non-core dimensions, preserving coordinates automatically.

    Parameters
    ----------
    da : xr.DataArray
        Dense DataArray.
    sigma : float, optional
        Gaussian kernel sigma in seconds.
    method : str
        ``"gaussian"`` or ``"boxcar"``.
    window : float, optional
        Boxcar window width in seconds.
    boundary : str
        Boundary padding mode for convolution.
    dim : str
        Dimension to smooth along (default ``"time"``).
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims {da.dims}.")

    time_vals = np.asarray(da.coords[dim].values, dtype=np.float64)
    dt = core.infer_dt(time_vals)

    if sigma is not None:
        method = "gaussian"
    elif method.lower() == "gaussian" and sigma is None and window is None:
        raise ValueError("Gaussian smoothing requires sigma parameter.")

    kernel = core.make_kernel(method, dt, sigma=sigma, window=window)
    if kernel.size == 1:
        return da.copy()

    causal = method.lower() in ("boxcar", "mean", "moving")

    out = xr.apply_ufunc(
        core.convolve_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        output_dtypes=[float],
        kwargs={"kernel": kernel, "boundary": boundary, "causal": causal},
    )
    out.attrs = dict(da.attrs)
    out = _preserve_coords(da, out)
    return out.transpose(*da.dims)


def baseline(
    da: xr.DataArray,
    *,
    window: tuple[float, float],
    dim: str = C.time,
    mode: str = "subtract",
) -> xr.DataArray:
    """Apply baseline correction over a time window.

    Uses xarray's ``.sel()`` and ``.mean()`` for dimension handling --
    no manual transposition or repacking needed.

    Parameters
    ----------
    da : xr.DataArray
        Dense DataArray.
    window : (float, float)
        ``(tmin, tmax)`` baseline window.
    dim : str
        Time dimension name.
    mode : str
        ``"subtract"``, ``"divide"``, or ``"zscore"``.
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims {da.dims}.")

    tmin, tmax = window
    bl = da.sel({dim: slice(tmin, tmax)})
    if bl.sizes[dim] == 0:
        raise ValueError("Baseline window produced no samples.")

    mean = bl.mean(dim=dim)

    mode = mode.lower()
    if mode == "subtract":
        out = da - mean
    elif mode == "divide":
        out = da / mean
    elif mode == "zscore":
        std = bl.std(dim=dim)
        safe_std = std.where(std != 0, 1.0)
        out = (da - mean) / safe_std
        out = out.where(std != 0, 0.0)
    else:
        raise ValueError(f"Unknown baseline mode {mode!r}.")

    out.attrs = dict(da.attrs)
    return _preserve_coords(da, out)


def normalize(
    da: xr.DataArray,
    *,
    dim: str | tuple[str, ...],
    method: str = "zscore",
) -> xr.DataArray:
    """Normalize across one or more dimensions.

    Uses xarray's broadcasting so no canonical dim ordering is required.

    Parameters
    ----------
    da : xr.DataArray
        Dense DataArray.
    dim : str or tuple of str
        Dimension(s) to normalize across.
    method : str
        ``"zscore"``, ``"minmax"``, or ``"robust"``.
    """
    if isinstance(dim, str):
        dims = (dim,)
    else:
        dims = tuple(dim)

    for d in dims:
        if d not in da.dims:
            raise ValueError(f"dim {d!r} not found in DataArray dims {da.dims}.")

    dims_list = list(dims)
    method = method.lower()

    if method == "zscore":
        mean = da.mean(dim=dims_list)
        std = da.std(dim=dims_list)
        safe_std = std.where(std != 0, 1.0)
        out = (da - mean) / safe_std
        out = out.where(std != 0, 0.0)
    elif method == "minmax":
        vmin = da.min(dim=dims_list)
        vmax = da.max(dim=dims_list)
        denom = vmax - vmin
        safe_denom = denom.where(denom != 0, 1.0)
        out = (da - vmin) / safe_denom
        out = out.where(denom != 0, 0.0)
    elif method == "robust":
        q25 = da.quantile(0.25, dim=dims_list).drop_vars("quantile")
        q75 = da.quantile(0.75, dim=dims_list).drop_vars("quantile")
        median = da.quantile(0.5, dim=dims_list).drop_vars("quantile")
        iqr = q75 - q25
        safe_iqr = iqr.where(iqr != 0, 1.0)
        out = (da - median) / safe_iqr
        out = out.where(iqr != 0, 0.0)
    else:
        raise ValueError(f"Unknown normalization method {method!r}.")

    out.attrs = dict(da.attrs)
    return _preserve_coords(da, out)


def psth(
    da: xr.DataArray,
    *,
    method: str = "mean",
    labels: NDArray | None = None,
    dim: str = C.trial,
) -> xr.DataArray:
    """Compute trial-averaged PSTH.

    Uses xarray's ``.mean()`` / ``.median()`` for unlabeled reduction,
    and ``xr.concat`` with per-group reduction for labeled PSTH.

    Parameters
    ----------
    da : xr.DataArray
        Dense DataArray.
    method : str
        ``"mean"`` or ``"median"``.
    labels : ndarray, optional
        Per-trial group labels. If provided, the reduced dimension is
        replaced by a group dimension with one entry per unique label.
    dim : str
        Dimension to reduce (default ``"trial"``).
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims {da.dims}.")

    method = method.lower()
    if method not in ("mean", "median"):
        raise ValueError(f"Unknown PSTH method {method!r}.")

    reduce_fn = method  # "mean" or "median" -- used as method name on DataArray

    if labels is None:
        out = getattr(da, reduce_fn)(dim=dim)
        out.attrs = dict(da.attrs)
        return _preserve_coords(da, out)

    # Grouped PSTH: reduce within each label group, then concat.
    labels = np.asarray(labels)
    unique = np.unique(labels)
    parts = []
    for lbl in unique:
        mask = labels == lbl
        subset = da.isel({dim: mask})
        parts.append(getattr(subset, reduce_fn)(dim=dim))

    out = xr.concat(parts, dim=dim)
    out[dim] = unique
    out.attrs = dict(da.attrs)
    return _preserve_coords(da, out)


def restrict(
    da: xr.DataArray,
    window: tuple[float, float],
    *,
    dim: str = C.time,
) -> xr.DataArray:
    """Restrict a DataArray to a time window.

    Dispatches between dense and ragged based on dtype.

    Parameters
    ----------
    da : xr.DataArray
        Dense or ragged DataArray.
    window : (float, float)
        ``(tmin, tmax)`` time window.
    dim : str
        Time dimension for dense data.
    """
    if da.dtype == object:
        return _restrict_ragged(da, window)
    return _restrict_dense(da, window, dim=dim)


def _restrict_dense(
    da: xr.DataArray,
    window: tuple[float, float],
    *,
    dim: str = C.time,
) -> xr.DataArray:
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims {da.dims}.")
    tmin, tmax = window
    if tmin > tmax:
        raise ValueError(f"window must have tmin <= tmax, got {window}.")
    return da.sel({dim: slice(tmin, tmax)})


def _restrict_ragged(
    da: xr.DataArray,
    window: tuple[float, float],
) -> xr.DataArray:
    spike_times, n_trials, n_units, _ = _unpack_ragged(da)
    restricted = core.restrict_ragged(spike_times, window)
    out = _repack_ragged(da, restricted, n_trials, n_units)
    out.attrs[C.attr_valid_intervals] = [window]
    return out


# ------------------------------------------------------------------
# Public wrappers -- core-backed (no xarray equivalent)
# ------------------------------------------------------------------


def bin(
    da: xr.DataArray,
    dt: float,
    *,
    window: tuple[float, float] | None = None,
    output: str = "rate",
) -> xr.DataArray:
    """Bin ragged spikes into a dense representation.

    Parameters
    ----------
    da : xr.DataArray
        Ragged spike DataArray (``dtype=object``).
    dt : float
        Bin width in seconds.
    window : (float, float), optional
        Time window. If ``None``, inferred from attrs or data.
    output : str
        ``"rate"`` (spikes/s) or ``"count"``.
    """
    if da.dtype != object:
        raise ValueError("bin expects ragged spikes with dtype=object.")

    has_trial = C.trial in da.dims
    has_unit = C.unit in da.dims
    if not has_trial and not has_unit:
        raise ValueError("bin requires at least a 'trial' or 'unit' dimension.")

    if window is None:
        intervals = da.attrs.get(C.attr_valid_intervals)
        if intervals:
            window = tuple(intervals[0])
        else:
            window = _infer_tlim(da)

    spike_times, n_trials, n_units, da_work = _unpack_ragged(da)
    data, centers = core.bin_spikes(spike_times, dt, window, output=output)

    # Build output dims and coords based on which dims were in the input.
    if has_trial and has_unit:
        out_data = data
        out_dims = (C.trial, C.unit, C.time)
        coords: dict = {
            C.trial: da_work.coords.get(C.trial, np.arange(n_trials)),
            C.unit: da_work.coords.get(C.unit, np.arange(n_units)),
            C.time: centers,
        }
    elif has_unit:
        out_data = data[0]  # squeeze synthetic trial dim
        out_dims = (C.unit, C.time)
        coords = {
            C.unit: da.coords.get(C.unit, np.arange(n_units)),
            C.time: centers,
        }
    else:  # has_trial only
        out_data = data[:, 0, :]  # squeeze synthetic unit dim
        out_dims = (C.trial, C.time)
        coords = {
            C.trial: da.coords.get(C.trial, np.arange(n_trials)),
            C.time: centers,
        }

    out = xr.DataArray(
        out_data,
        dims=out_dims,
        coords=coords,
        name=da.name,
        attrs=dict(da.attrs),
    )
    out = _preserve_coords(da, out)
    out.attrs[C.attr_valid_intervals] = [window]
    return out


def align(
    da: xr.DataArray,
    *,
    events: xr.Dataset | xr.DataArray,
    to: str,
    window: tuple[float, float],
) -> xr.DataArray:
    """Align ragged spikes to event times.

    Parameters
    ----------
    da : xr.DataArray
        Ragged spike DataArray (``dtype=object``). Expects a single
        "trial" containing all session spikes per unit, i.e. shape
        ``(1, n_units)`` or ``(n_units,)``.
    events : xr.Dataset or xr.DataArray
        Event times with a ``"trial"`` and ``"event"`` dimension.
    to : str
        Event label to align to.
    window : (float, float)
        ``(tmin, tmax)`` relative to event time.
    """
    tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(f"window must be (tmin, tmax) with tmin < tmax, got {window}.")

    events_ds = _normalize_events(events)
    t0 = _get_event_times(events_ds, to=to)
    anchor_times = t0.values.astype(np.float64)

    # Unpack the single-session ragged array and extract per-unit spike lists.
    ragged, _, n_units, _ = _unpack_ragged(da)
    spike_list = ragged[0]  # first (and only) "trial" = all session spikes

    aligned = core.align(spike_list, anchor_times, window)
    n_trials = len(anchor_times)

    out = _repack_ragged(
        da,
        aligned,
        n_trials,
        n_units,
        trial_coord=t0[C.trial] if C.trial in t0.coords else np.arange(n_trials),
    )
    out.attrs[C.attr_valid_intervals] = [window]
    return out
