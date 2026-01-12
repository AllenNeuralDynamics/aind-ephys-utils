"""PSTH computation and plotting.

The main entry point is `psth`, which takes a ragged spike DataArray and
computes a peristimulus time histogram (PSTH). Plotting is optional and uses
matplotlib.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class PSTHPlotError(ValueError):
    """Raised when PSTH inputs are invalid or inconsistent."""

    pass


def _as_list_of_1d_arrays(values: np.ndarray) -> list[np.ndarray]:
    """Convert an object array of spike arrays into a list of 1D float arrays."""
    out: list[np.ndarray] = []
    for x in values:
        if x is None:
            out.append(np.asarray([], dtype=float))
        else:
            a = np.asarray(x, dtype=float)
            if a.ndim != 1:
                raise PSTHPlotError(
                    f"Expected 1D spike arrays, got shape {a.shape}."
                )
            out.append(a)
    return out


def _make_time_centers(
    bin_size: float, window: Tuple[float, float]
) -> np.ndarray:
    """Create uniformly spaced bin centers for a PSTH."""
    tmin, tmax = window
    if tmin >= tmax:
        raise PSTHPlotError(f"Invalid window {window}: require tmin < tmax.")
    edges = np.arange(tmin, tmax + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if centers.size == 0:
        raise PSTHPlotError(
            f"Window {window} with bin_size={bin_size} produced 0 bins."
        )
    return centers


def _bin_edges_from_centers(centers: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute histogram edges (and dt) from uniformly spaced centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise PSTHPlotError("time must be a 1D array with at least 2 points.")
    dt = float(centers[1] - centers[0])
    if not np.allclose(np.diff(centers), dt):
        raise PSTHPlotError(
            "time vector must be uniformly spaced for binning."
        )
    edges = np.concatenate([centers - dt / 2, [centers[-1] + dt / 2]])
    return edges, dt


def _gaussian_kernel(sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    """Return a normalized 1D Gaussian smoothing kernel in bin units."""
    if sigma_bins <= 0:
        return np.asarray([1.0])
    half = int(np.ceil(truncate * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k


def _smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Smooth a 1D array by Gaussian convolution in bin units."""
    if sigma_bins <= 0:
        return y
    k = _gaussian_kernel(sigma_bins)
    return np.convolve(y, k, mode="same")


def _trialwise_binned_rate(
    ragged_trial_spikes: list[np.ndarray],
    *,
    edges: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Return (trial, time) rate matrix (Hz).
    """
    n_trials = len(ragged_trial_spikes)
    n_bins = len(edges) - 1
    out = np.zeros((n_trials, n_bins), dtype=float)
    for i, s in enumerate(ragged_trial_spikes):
        if s.size:
            out[i], _ = np.histogram(s, bins=edges)
    out /= dt
    return out


def psth(
    spikes: xr.DataArray,
    *,
    unit: Optional[Union[int, str]] = None,
    units: Optional[Sequence[Union[int, str]]] = None,
    trials: Optional[Sequence[Union[int, str]]] = None,
    by: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
    bin_size: float = 0.01,
    window: Tuple[float, float] = (-0.5, 1.0),
    time: Optional[np.ndarray] = None,
    smooth_sigma: float = 0.0,  # seconds; 0 disables
    error: Optional[Literal["sem"]] = "sem",
    ax: Optional[plt.Axes] = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    fill_alpha: float = 0.25,
    label: Optional[str] = None,
) -> Tuple[xr.DataArray, Optional[plt.Axes]]:
    """
    Compute (and optionally plot) a PSTH from a ragged spike DataArray.

    Input
    -----
    spikes:
        Ragged spikes DataArray with dtype=object.
        Expected dims: ("unit","trial") or pre-selected ("trial",).

    Output
    ------
    psth_da:
        If by is None:
          dims: ("time",) if one unit, else ("unit","time")
        If by is not None:
          dims: ("group","time") if one unit, else ("group","unit","time")

        Values are firing rate (Hz).

    ax:
        If ax was provided (or created), returns it; otherwise None.

    Notes
    -----
    - smooth_sigma is in *seconds* and applied after binning.
    - error="sem" shades ± SEM across trials (within each group).
    """
    if spikes.dtype != object:
        raise PSTHPlotError(
            f"psth expects a ragged spike DataArray with dtype=object; got {spikes.dtype!r}."
        )

    # Resolve units selection
    per_unit: list[Tuple[str, xr.DataArray]] = []
    if "unit" in spikes.dims:
        if units is not None:
            for u in units:
                da_u = spikes.sel(unit=u)
                if "trial" not in da_u.dims:
                    raise PSTHPlotError(
                        "Selected unit did not yield a ('trial',) DataArray."
                    )
                per_unit.append((str(u), da_u))
        else:
            if unit is None:
                if spikes.sizes.get("unit", 0) == 1:
                    u0 = spikes["unit"].values[0]
                    per_unit.append((str(u0), spikes.isel(unit=0)))
                else:
                    raise PSTHPlotError(
                        "Specify unit=... or units=[...] when spikes has multiple units."
                    )
            else:
                per_unit.append((str(unit), spikes.sel(unit=unit)))
    else:
        if "trial" not in spikes.dims:
            raise PSTHPlotError(
                "Expected spikes dims to include ('unit','trial') or be pre-selected to ('trial',)."
            )
        per_unit.append(("unit", spikes))

    # Time axis
    if time is None:
        centers = _make_time_centers(bin_size, window)
    else:
        centers = np.asarray(time, dtype=float)

    edges, dt = _bin_edges_from_centers(centers)
    sigma_bins = (
        float(smooth_sigma) / dt if smooth_sigma and smooth_sigma > 0 else 0.0
    )

    # Helper: apply trial selection / sorting consistently for each unit
    def _prep_trials(da: xr.DataArray) -> xr.DataArray:
        """Apply trial subsetting and sorting consistently for each unit."""
        out = da
        if trials is not None:
            out = out.sel(trial=list(trials))
        if sort_by is not None:
            if sort_by not in out.coords:
                raise PSTHPlotError(
                    f"sort_by={sort_by!r} not found in coords. Available: {list(out.coords)}"
                )
            order = np.argsort(np.asarray(out[sort_by].values))
            if not sort_ascending:
                order = order[::-1]
            out = out.isel(trial=order)
        return out

    # Grouping
    if by is not None:
        # Determine groups from the first unit after selection/sorting
        da0 = _prep_trials(per_unit[0][1])
        if by not in da0.coords:
            raise PSTHPlotError(
                f"by={by!r} not found in coords. Available: {list(da0.coords)}"
            )
        group_vals = np.asarray(da0[by].values)
        # Order-preserving unique groups
        seen = set()
        groups: List[Any] = []
        for v in group_vals.tolist():
            if v not in seen:
                groups.append(v)
                seen.add(v)
    else:
        groups = [None]

    # Compute
    # We compute trialwise binned rates, then average across trials (and keep SEM if requested)
    unit_labels = [u for (u, _) in per_unit]
    n_units = len(per_unit)
    n_groups = len(groups)

    mean_rate = np.zeros((n_groups, n_units, centers.size), dtype=float)
    sem_rate = (
        np.zeros((n_groups, n_units, centers.size), dtype=float)
        if error == "sem"
        else None
    )

    for ui, (_, da_u_raw) in enumerate(per_unit):
        da_u = _prep_trials(da_u_raw)

        if by is None:
            seq = _as_list_of_1d_arrays(np.asarray(da_u.values, dtype=object))
            trial_rate = _trialwise_binned_rate(
                seq, edges=edges, dt=dt
            )  # (trial, time)
            m = trial_rate.mean(axis=0)
            if sigma_bins > 0:
                m = _smooth_1d(m, sigma_bins)
            mean_rate[0, ui, :] = m

            if sem_rate is not None:
                s = trial_rate.std(axis=0, ddof=1) / np.sqrt(
                    max(trial_rate.shape[0], 1)
                )
                if sigma_bins > 0:
                    s = _smooth_1d(s, sigma_bins)
                sem_rate[0, ui, :] = s
        else:
            by_vals = np.asarray(da_u[by].values)
            for gi, g in enumerate(groups):
                idx = np.nonzero(by_vals == g)[0]
                if idx.size == 0:
                    mean_rate[gi, ui, :] = np.nan
                    if sem_rate is not None:
                        sem_rate[gi, ui, :] = np.nan
                    continue

                da_g = da_u.isel(trial=idx)
                seq = _as_list_of_1d_arrays(
                    np.asarray(da_g.values, dtype=object)
                )
                trial_rate = _trialwise_binned_rate(seq, edges=edges, dt=dt)
                m = trial_rate.mean(axis=0)
                if sigma_bins > 0:
                    m = _smooth_1d(m, sigma_bins)
                mean_rate[gi, ui, :] = m

                if sem_rate is not None:
                    s = trial_rate.std(axis=0, ddof=1) / np.sqrt(
                        max(trial_rate.shape[0], 1)
                    )
                    if sigma_bins > 0:
                        s = _smooth_1d(s, sigma_bins)
                    sem_rate[gi, ui, :] = s

    # Build DataArray
    coords: Dict[str, Any] = {"time": centers}
    dims: Tuple[str, ...]
    data: np.ndarray

    if by is None:
        if n_units == 1:
            data = mean_rate[0, 0, :]
            dims = ("time",)
        else:
            data = mean_rate[0, :, :]
            dims = ("unit", "time")
            coords["unit"] = np.asarray(unit_labels, dtype=object)
    else:
        coords["group"] = np.asarray(groups, dtype=object)
        if n_units == 1:
            data = mean_rate[:, 0, :]
            dims = ("group", "time")
        else:
            data = mean_rate
            dims = ("group", "unit", "time")
            coords["unit"] = np.asarray(unit_labels, dtype=object)

    psth_da = xr.DataArray(data, dims=dims, coords=coords, name="psth_rate_hz")
    psth_da.attrs["bin_size_s"] = float(dt)
    psth_da.attrs["window"] = tuple(window)
    psth_da.attrs["smooth_sigma_s"] = float(smooth_sigma)
    psth_da.attrs["units"] = "Hz"

    # Plot (optional)
    out_ax: Optional[plt.Axes] = None
    if ax is not None:
        out_ax = ax
    # If user passes ax=None, we don't auto-create to keep compute-only workflows clean.
    # If you want auto-create, pass ax=plt.gca() or create externally.

    if out_ax is not None:
        if by is None:
            if n_units == 1:
                y = mean_rate[0, 0, :]
                out_ax.plot(
                    centers, y, linewidth=linewidth, alpha=alpha, label=label
                )
                if sem_rate is not None:
                    s = sem_rate[0, 0, :]
                    out_ax.fill_between(
                        centers, y - s, y + s, alpha=fill_alpha
                    )
            else:
                # plot each unit (small multiples can be a separate function later)
                for ui, u in enumerate(unit_labels):
                    y = mean_rate[0, ui, :]
                    out_ax.plot(
                        centers,
                        y,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=str(u),
                    )
                    if sem_rate is not None:
                        s = sem_rate[0, ui, :]
                        out_ax.fill_between(
                            centers, y - s, y + s, alpha=fill_alpha
                        )
        else:
            # plot each group (and average across units if multiple units)
            if n_units == 1:
                for gi, g in enumerate(groups):
                    y = mean_rate[gi, 0, :]
                    out_ax.plot(
                        centers,
                        y,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=str(g),
                    )
                    if sem_rate is not None:
                        s = sem_rate[gi, 0, :]
                        out_ax.fill_between(
                            centers, y - s, y + s, alpha=fill_alpha
                        )
            else:
                # average across units for plotting
                y_mu = np.nanmean(mean_rate, axis=1)  # (group, time)
                y_se = None
                if sem_rate is not None:
                    # combine SEM across trials within each unit is tricky; for now shade unit-to-unit SEM
                    y_se = np.nanstd(mean_rate, axis=1, ddof=1) / np.sqrt(
                        n_units
                    )

                for gi, g in enumerate(groups):
                    y = y_mu[gi]
                    out_ax.plot(
                        centers,
                        y,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=str(g),
                    )
                    if y_se is not None:
                        s = y_se[gi]
                        out_ax.fill_between(
                            centers, y - s, y + s, alpha=fill_alpha
                        )

        out_ax.set_xlabel("time (s)")
        out_ax.set_ylabel("rate (Hz)")
        out_ax.axvline(0.0, linewidth=1.0, alpha=0.5)
        out_ax.set_xlim((centers[0], centers[-1]))
        out_ax.legend(frameon=False)

    return psth_da, out_ax


__all__ = ["psth", "PSTHPlotError"]
