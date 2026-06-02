"""Sliding window PSTH computation and optimal bin-width selection.

The bin-width selector implements the Shimazaki--Shinomoto method for
choosing a histogram bin width that minimises mean integrated squared
error (Shimazaki & Shinomoto, 2007).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _detect_per_trial_relative(spike_times: Any) -> bool:
    """Detect whether spike_times is per-trial relative (sequence of arrays)."""
    if (
        hasattr(spike_times, "__len__")
        and hasattr(spike_times, "__getitem__")
        and len(spike_times) > 0
    ):
        first = spike_times[0]
        if isinstance(first, (np.ndarray, list, tuple)):
            return True
    return False


def _compute_extended_bins(
    time_range: tuple[float, float],
    window_size: float,
    bin_size: float,
) -> tuple[int, float, float, float, np.ndarray, np.ndarray, int]:
    """Compute extended bin edges and centers for sliding window PSTH."""
    t0, t1 = time_range
    window_bins = max(1, int(round(window_size / bin_size)))
    effective_window_size = window_bins * bin_size

    pad = effective_window_size
    ext_start = t0 - pad
    ext_end = t1 + pad

    ext_duration = ext_end - ext_start
    n_bins_ext = int(np.floor(ext_duration / bin_size))
    if n_bins_ext < 1:
        raise ValueError("Extended range too small given bin_size")

    bins = ext_start + np.arange(n_bins_ext + 1) * bin_size
    bin_centers_ext = ext_start + (np.arange(n_bins_ext) + 0.5) * bin_size

    return (
        window_bins,
        pad,
        ext_start,
        ext_end,
        bins,
        bin_centers_ext,
        n_bins_ext,
    )


def _canonicalize_trials_and_supports(  # noqa: C901
    spike_times: np.ndarray | Sequence[np.ndarray] | Sequence[Sequence[float]],
    trial_starts: np.ndarray | Sequence[float] | None,
    trial_supports: (
        tuple[float, float] | Sequence[tuple[float, float] | None] | None
    ),
    per_trial_relative: bool,
    ext_start: float,
    ext_end: float,
) -> tuple[list[np.ndarray], list[list[tuple[float, float]]], int]:
    """Convert inputs into per-trial relative spikes and support intervals."""
    spike_times_by_trial: list[np.ndarray] = []
    supports_by_trial: list[list[tuple[float, float]]] = []

    if per_trial_relative:
        # Mode 2: per-trial relative arrays
        spike_times_seq = spike_times
        n_trials = len(spike_times_seq)

        for st in spike_times_seq:
            spike_times_by_trial.append(np.asarray(st, dtype=float))

        if trial_supports is None:
            # All trials valid over full extended range
            supports_by_trial = [
                [(ext_start, ext_end)] for _ in range(n_trials)
            ]
        else:
            ts_list = list(trial_supports)
            if len(ts_list) != n_trials:
                raise ValueError(
                    "In per-trial relative mode, trial_supports must have the same "
                    "length as spike_times (one (start, end) per trial)."
                )
            for supp in ts_list:
                if supp is None:
                    supports_by_trial.append([(ext_start, ext_end)])
                else:
                    if not (
                        isinstance(supp, (tuple, list, np.ndarray))
                        and len(supp) == 2
                    ):
                        raise ValueError(
                            "Each trial_supports entry must be None or a "
                            "(start, end) pair in relative time for mode 2."
                        )
                    s0, s1 = float(supp[0]), float(supp[1])
                    supports_by_trial.append([(s0, s1)])

    else:
        # Mode 1: absolute spike times + trial starts
        spikes_abs = np.asarray(spike_times, dtype=float)
        if spikes_abs.ndim != 1:
            raise ValueError("spike_times must be 1D in absolute-time mode")
        spikes_abs = np.sort(spikes_abs)

        if trial_starts is None:
            raise ValueError(
                "trial_starts must be provided when spike_times is a 1D "
                "array of absolute timestamps."
            )
        trial_starts_arr = np.asarray(trial_starts, dtype=float)
        if trial_starts_arr.ndim != 1:
            raise ValueError("trial_starts must be 1D")
        n_trials = len(trial_starts_arr)

        # Single global absolute support interval
        if trial_supports is None:
            g0, g1 = float("-inf"), float("inf")
        else:
            ts = trial_supports
            if not (
                isinstance(ts, (tuple, list, np.ndarray))
                and len(ts) == 2
                and np.isscalar(ts[0])
                and np.isscalar(ts[1])
            ):
                raise ValueError(
                    "trial_supports in absolute-time mode must be a single "
                    "(start, end) pair in absolute time."
                )
            g0, g1 = float(ts[0]), float(ts[1])

        for t_trial in trial_starts_arr:
            start_abs = t_trial + ext_start
            end_abs = t_trial + ext_end

            i0 = np.searchsorted(spikes_abs, start_abs, side="left")
            i1 = np.searchsorted(spikes_abs, end_abs, side="right")
            rel_spikes = spikes_abs[i0:i1] - t_trial
            spike_times_by_trial.append(rel_spikes)

            rel_start = g0 - t_trial
            rel_end = g1 - t_trial
            supports_by_trial.append([(rel_start, rel_end)])

    return spike_times_by_trial, supports_by_trial, n_trials


def _accumulate_counts_and_support(
    spike_times_by_trial: Sequence[np.ndarray],
    supports_by_trial: Sequence[Sequence[tuple[float, float]]],
    bins: np.ndarray,
    bin_centers_ext: np.ndarray,
    ext_start: float,
    ext_end: float,
    n_bins_ext: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate spike counts and trial support per fine bin."""
    counts = np.zeros(n_bins_ext, dtype=float)
    trials_per_bin = np.zeros(n_bins_ext, dtype=float)

    for rel_spikes, intervals in zip(spike_times_by_trial, supports_by_trial):
        rel_spikes = np.asarray(rel_spikes, dtype=float)

        for obs_start, obs_end in intervals:
            obs_start_eff = max(obs_start, ext_start)
            obs_end_eff = min(obs_end, ext_end)
            if obs_end_eff <= obs_start_eff:
                continue

            # Spike counts
            if rel_spikes.size > 0:
                mask_spk = (rel_spikes >= obs_start_eff) & (
                    rel_spikes <= obs_end_eff
                )
                rel_spikes_in = rel_spikes[mask_spk]
                if rel_spikes_in.size > 0:
                    trial_counts, _ = np.histogram(rel_spikes_in, bins=bins)
                    counts += trial_counts

            # Trial support
            mask_bins = (bin_centers_ext >= obs_start_eff) & (
                bin_centers_ext < obs_end_eff
            )
            trials_per_bin[mask_bins] += 1.0

    return counts, trials_per_bin


def _smooth_counts_and_support(
    counts: np.ndarray,
    trials_per_bin: np.ndarray,
    window_bins: int,
    causal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply boxcar smoothing to counts and trial support."""
    if causal:
        pad_front = window_bins - 1

        padded_counts = np.concatenate(
            [np.zeros(pad_front, dtype=float), counts]
        )
        cumsum_counts = np.cumsum(padded_counts)
        smoothed_counts = (
            cumsum_counts[window_bins:] - cumsum_counts[:-window_bins]
        ) / window_bins

        padded_trials = np.concatenate(
            [np.zeros(pad_front, dtype=float), trials_per_bin]
        )
        cumsum_trials = np.cumsum(padded_trials)
        smoothed_trials = (
            cumsum_trials[window_bins:] - cumsum_trials[:-window_bins]
        ) / window_bins

    else:
        half = window_bins // 2
        pad_left = half
        pad_right = window_bins - half

        padded_counts = np.concatenate(
            [
                np.zeros(pad_left, dtype=float),
                counts,
                np.zeros(pad_right, dtype=float),
            ]
        )
        cumsum_counts = np.cumsum(padded_counts)
        smoothed_counts = (
            cumsum_counts[window_bins:] - cumsum_counts[:-window_bins]
        ) / window_bins

        padded_trials = np.concatenate(
            [
                np.zeros(pad_left, dtype=float),
                trials_per_bin,
                np.zeros(pad_right, dtype=float),
            ]
        )
        cumsum_trials = np.cumsum(padded_trials)
        smoothed_trials = (
            cumsum_trials[window_bins:] - cumsum_trials[:-window_bins]
        ) / window_bins

    return smoothed_counts, smoothed_trials


def _trim_to_range(
    psth_full: np.ndarray,
    smoothed_trials: np.ndarray,
    time_range: tuple[float, float],
    bin_size: float,
    pad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim PSTH and trial support back to requested time_range."""
    t0, t1 = time_range
    pad_bins = int(round(pad / bin_size))
    n_bins_main = int(np.floor((t1 - t0) / bin_size))

    time_bins = t0 + (np.arange(n_bins_main) + 0.5) * bin_size

    start_idx = pad_bins
    end_idx = start_idx + n_bins_main
    if end_idx > len(psth_full):
        end_idx = len(psth_full)
        n_bins_main = end_idx - start_idx
        time_bins = t0 + (np.arange(n_bins_main) + 0.5) * bin_size

    psth = psth_full[start_idx:end_idx]
    n_trials_eff = smoothed_trials[start_idx:end_idx]
    return time_bins, psth, n_trials_eff


def sliding_window_psth(
    spike_times: np.ndarray | Sequence[np.ndarray] | Sequence[Sequence[float]],
    trial_starts: np.ndarray | Sequence[float] | None = None,
    *,
    window_size: float = 0.05,
    bin_size: float = 0.001,
    time_range: tuple[float, float] = (-0.5, 1.0),
    causal: bool = False,
    trial_supports: (
        tuple[float, float] | Sequence[tuple[float, float] | None] | None
    ) = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window PSTH using binning + cumsum boxcar smoothing.

    Modes
    -----
    1) Absolute spike times + trial starts
       - spike_times : 1D array of absolute timestamps (seconds)
       - trial_starts : 1D array of absolute trial onset timestamps (seconds)
       - trial_supports (optional):
           * None: entire recording is valid
           * (abs_start, abs_end): single global valid interval (absolute time)

    2) Per-trial relative spike times
       - spike_times : sequence (or numpy object array) of 1D arrays, each with
                       spike times already relative to that trial's event.
       - trial_supports (optional):
           * None: each trial valid over full extended range
           * sequence of (start, end) in relative time (one per trial)

    Returns
    -------
    time_bins : (n_bins,)
    psth : (n_bins,)
        Firing rate (spikes/s).
    n_trials_eff : (n_bins,)
        Effective number of trials contributing to each bin.
    """
    t0, t1 = time_range
    if t1 <= t0:
        raise ValueError("time_range must have end > start")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    per_trial_relative = _detect_per_trial_relative(spike_times)

    (
        window_bins,
        pad,
        ext_start,
        ext_end,
        bins,
        bin_centers_ext,
        n_bins_ext,
    ) = _compute_extended_bins(time_range, window_size, bin_size)

    (
        spike_times_by_trial,
        supports_by_trial,
        n_trials,
    ) = _canonicalize_trials_and_supports(
        spike_times,
        trial_starts,
        trial_supports,
        per_trial_relative,
        ext_start,
        ext_end,
    )

    if n_trials == 0:
        raise ValueError("No trials found / n_trials == 0")

    counts, trials_per_bin = _accumulate_counts_and_support(
        spike_times_by_trial,
        supports_by_trial,
        bins,
        bin_centers_ext,
        ext_start,
        ext_end,
        n_bins_ext,
    )

    smoothed_counts, smoothed_trials = _smooth_counts_and_support(
        counts, trials_per_bin, window_bins, causal
    )

    # Use np.divide with where= to avoid invalid divisions
    psth_full = np.full_like(smoothed_counts, np.nan, dtype=float)
    np.divide(
        smoothed_counts,
        smoothed_trials * bin_size,
        out=psth_full,
        where=smoothed_trials > 0,
    )

    time_bins, psth, n_trials_eff = _trim_to_range(
        psth_full,
        smoothed_trials,
        time_range,
        bin_size,
        pad,
    )
    return time_bins, psth, n_trials_eff


# ---------------------------------------------------------------------------
# Shimazaki–Shinomoto optimal bin-width selection
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PSTHBinWidthResult:
    """Result of :func:`select_psth_bin_width`."""

    best_width_s: float
    best_cost: float
    candidate_widths_s: FloatArray
    candidate_costs: FloatArray
    best_multiplier: int
    refined: bool


def _as_float_array(x: Sequence[float] | NDArray[np.floating]) -> FloatArray:
    """Coerce input to a 1D float64 array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr


def _validate_trials(
    spike_times_by_trial: Sequence[Sequence[float] | NDArray[np.floating]],
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
    """Histogram each trial at the finest (base) bin width."""
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
    """Sum base-bin counts in non-overlapping groups of *m* bins."""
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
    """Compute the Shimazaki-Shinomoto SSMD cost for a given bin width."""
    if bin_width_s <= 0 or not np.isfinite(bin_width_s):
        return float("inf")

    n_trials, n_bins = binned_counts_by_trial.shape
    if n_trials < 1 or n_bins < 1:
        return float("inf")

    k = binned_counts_by_trial.sum(axis=0, dtype=np.int64).astype(np.float64)
    kbar = float(k.mean())
    v = float(((k - kbar) ** 2).mean())

    cost: float = (2.0 * kbar - v) / ((n_trials * bin_width_s) ** 2)
    return cost


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
    spike_times_by_trial: Sequence[Sequence[float] | NDArray[np.floating]],
    *,
    t_start: float,
    t_stop: float,
    base_bin_width_s: float = 0.001,
    min_width_s: float | None = None,
    max_width_s: float | None = None,
    n_grid: int = 60,
    log_spacing: bool = True,
    refine: bool = True,
    local_refine_radius: int = 5,
) -> PSTHBinWidthResult:
    """Select an optimal PSTH bin width via the Shimazaki-Shinomoto method.

    Performs a single fine-resolution prebinning pass, then searches over
    integer multiples of *base_bin_width_s* to minimise the SSMD cost
    (mean integrated squared error).  An optional dense local refinement
    step improves precision around the coarse-grid winner.

    Parameters
    ----------
    spike_times_by_trial
        Spike times for one neuron, one array-like per trial, in seconds.
    t_start, t_stop
        PSTH window ``[t_start, t_stop)``.
    base_bin_width_s
        Finest time resolution for prebinning (default 1 ms).
    min_width_s, max_width_s
        Search range.  Defaults: ``base_bin_width_s`` and
        ``min(0.25 * window, 0.2)`` respectively.
    n_grid
        Coarse grid points before deduplication.
    log_spacing
        Use approximately log-spaced coarse candidates.
    refine
        Dense local refinement around the coarse winner.
    local_refine_radius
        Integer multipliers to search on each side of the coarse winner.

    Returns
    -------
    PSTHBinWidthResult
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
