"""Ripley's K and L functions for 1D point processes.

Estimates the K function and its variance-stabilised transform L for
temporal point processes (spike trains, event times).  Supports
multiple observation windows (trials) treated as replicates, with
proper pooled estimation and no cross-window pairs.

1D conventions
--------------
For a homogeneous Poisson process with intensity lambda:

* K(t) = 2t
* L(t) = K(t)/2 - t = 0  (L > 0 = clustering, L < 0 = regularity)

Edge correction
---------------
The default is **translation correction** (1D):  each pair (i, j) with
lag h = |xi - xj| is upweighted by T / (T - h), compensating for the
fact that fewer positions in [0, T] can support lag h.

For multiple windows with different durations, a **pooled** estimator
sums corrected pair contributions across windows with window-specific
normalisation.  Windows are treated as replicates: no cross-window
pairs are formed.

A **border** correction is also available: at scale t, only reference
points at least t from both window boundaries are used.  Simpler but
less efficient, especially for short windows or large t.

Caveats
-------
Edge correction does not fix nonstationarity.  If the event rate
changes strongly within a window or across trials, L > 0 may reflect
rate modulation rather than true clustering.  Consider time-rescaling
or an inhomogeneous K variant for non-stationary data.

Keep t_max well below the shortest window duration to avoid unstable
translation weights (T / (T - h) blows up as h -> T).

.. todo::
    Cross-Ripley's K/L (``cross_ripley_k``, ``cross_ripley_k_envelope``).
    Measures association between two point processes on the same window.
    Same translation correction, but pairs are cross-process (every i
    from process 1 with every j from process 2).  Denominator uses
    ``n_1m * n_2m / T_m**2`` instead of ``n_m * (n_m - 1) / T_m**2``.
    Preserves absolute distance only (no lag sign) — for directional
    analysis use a cross-correlogram / peri-event histogram instead.
    Main use case: testing whether a spike train is associated with
    behavioral events (licks, rewards) at multiple timescales.  Less
    useful than auto-L for artifact detection because it loses
    directionality; a peri-event histogram of contributing vs
    non-contributing spikes is more informative for that purpose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aind_ephys_utils._numba import njit, prange

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RipleyResult:
    """Result of :func:`ripley_k`."""

    r: NDArray[np.float64]
    K: NDArray[np.float64]
    L: NDArray[np.float64]
    L_uncentered: NDArray[np.float64]
    edge_correction: str
    intensity: float
    n_points: int
    observation_duration: float
    denominator: float


@dataclass(frozen=True)
class RipleyEnvelopeResult:
    """Result of :func:`ripley_k_envelope` with Monte Carlo envelopes."""

    r: NDArray[np.float64]
    K: NDArray[np.float64]
    L: NDArray[np.float64]
    L_uncentered: NDArray[np.float64]
    K_lo: NDArray[np.float64]
    K_hi: NDArray[np.float64]
    L_lo: NDArray[np.float64]
    L_hi: NDArray[np.float64]
    edge_correction: str
    intensity: float
    n_points: int
    observation_duration: float
    denominator: float
    n_simulations: int
    alpha: float


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _normalise_window_lengths(
    n_obs: int,
    window_lengths: float | Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Broadcast *window_lengths* to a ``(n_obs,)`` float64 array."""
    if np.isscalar(window_lengths):
        T = np.full(n_obs, float(window_lengths), dtype=np.float64)
    else:
        T = np.asarray(window_lengths, dtype=np.float64)
        if T.shape != (n_obs,):
            raise ValueError(
                "window_lengths must be a scalar or shape (len(observations),)"
            )
    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("All window lengths must be finite and > 0.")
    return T


def _prepare_trials(
    observations: Sequence[ArrayLike],
    window_lengths: NDArray[np.float64],
    assume_sorted: bool,
) -> list[NDArray[np.float64]]:
    """Validate and sort per-trial event times."""
    trials: list[NDArray[np.float64]] = []
    for obs, T in zip(observations, window_lengths, strict=True):
        x = np.asarray(obs, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(
                "Each observation must be a 1D array of event times."
            )
        if np.any(~np.isfinite(x)):
            raise ValueError("Event times must be finite.")
        if not assume_sorted and x.size > 1:
            x = np.sort(x)
        if x.size:
            if x[0] < 0 or x[-1] > T:
                raise ValueError(f"Event times must lie inside [0, {T}].")
        trials.append(x)
    return trials


def _flatten_trials(
    trials: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Flatten per-trial arrays into CSR-like (flat, offsets) layout."""
    offsets = np.empty(len(trials) + 1, dtype=np.int64)
    offsets[0] = 0
    for i, x in enumerate(trials):
        offsets[i + 1] = offsets[i] + x.size
    flat = np.empty(int(offsets[-1]), dtype=np.float64)
    for i, x in enumerate(trials):
        flat[offsets[i] : offsets[i + 1]] = x
    return flat, offsets


def _auto_radii(
    durations: NDArray[np.float64], n_r: int
) -> NDArray[np.float64]:
    """Default evaluation radii: 0 to min(T)/4."""
    return np.linspace(
        0.0, float(durations.min()) / 4.0, n_r, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Translation correction core
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _accumulate_translation(
    flat,
    offsets,
    window_lengths,
    radii,
):
    """Numba-accelerated translation-corrected accumulation.

    Uses a forward-scanning radii pointer instead of binary search.
    Outer loop over reference points is parallelised with prange.
    """
    R = radii.size
    r_max = radii[-1] if R > 0 else 0.0
    denom = 0.0

    # Process each window; collect (start, end, T, cutoff) first
    # so the parallel loop over i can be flat across all windows.
    # For simplicity, loop windows sequentially and parallelise i.
    diff = np.zeros(R + 1, np.float64)

    for m in range(window_lengths.size):
        T = window_lengths[m]
        start = offsets[m]
        end = offsets[m + 1]
        n = end - start
        if n < 2:
            continue
        denom += n * (n - 1) / (T * T)
        cutoff = min(r_max, T)

        # Per-reference-point accumulation (parallel)
        n_ref = end - 1 - start
        local = np.zeros((n_ref, R + 1), np.float64)
        for ii in prange(n_ref):
            i = start + ii
            xi = flat[i]
            k = 0
            for j in range(i + 1, end):
                d = flat[j] - xi
                if d > cutoff:
                    break
                w = 2.0 / (T - d)
                while k < R and radii[k] < d:
                    k += 1
                if k < R:
                    local[ii, k] += w

        # Reduce across threads
        for ii in range(n_ref):
            for rr in range(R + 1):
                diff[rr] += local[ii, rr]

    return np.cumsum(diff[:-1]), denom


def _k_translation(
    flat: NDArray[np.float64],
    offsets: NDArray[np.int64],
    window_lengths: NDArray[np.float64],
    radii: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Compute translation-corrected K estimator."""
    numerator, denom = _accumulate_translation(
        flat, offsets, window_lengths, radii
    )
    if denom <= 0:
        return np.full(radii.size, np.nan, dtype=np.float64), 0.0
    K: NDArray[np.float64] = numerator / denom
    return K, float(denom)


# ---------------------------------------------------------------------------
# Border correction core
# ---------------------------------------------------------------------------


def _k_border(
    flat: NDArray[np.float64],
    offsets: NDArray[np.int64],
    window_lengths: NDArray[np.float64],
    radii: NDArray[np.float64],
    total_n: int,
    total_dur: float,
) -> tuple[NDArray[np.float64], float]:
    """Pooled border-corrected K estimator."""
    n_r = radii.size
    K = np.empty(n_r, dtype=np.float64)
    lam = total_n / total_dur
    denom = lam  # border uses lambda as the denominator scaling

    for idx in range(n_r):
        r_val = radii[idx]
        if r_val <= 0:
            K[idx] = 0.0
            continue

        total_interior = 0
        total_neighbors = 0
        for m in range(len(window_lengths)):
            T = window_lengths[m]
            start = int(offsets[m])
            end = int(offsets[m + 1])
            trial = flat[start:end]
            n_m = len(trial)
            if n_m < 2:
                continue
            interior = (trial >= r_val) & (trial <= T - r_val)
            n_interior = int(interior.sum())
            total_interior += n_interior
            if n_interior == 0:
                continue
            interior_pts = trial[interior]
            right = np.searchsorted(trial, interior_pts + r_val, side="right")
            left = np.searchsorted(trial, interior_pts - r_val, side="left")
            total_neighbors += int((right - left - 1).sum())

        if total_interior == 0:
            K[idx] = np.nan
        else:
            K[idx] = float(total_neighbors) / (lam * total_interior)

    return K, float(denom)


# ---------------------------------------------------------------------------
# Monte Carlo envelope helpers
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _simulate_envelopes_translation(
    all_sims_flat,
    sim_offsets,
    window_lengths,
    radii,
    n_simulations,
):
    """Numba-accelerated MC envelopes for translation correction.

    Parameters
    ----------
    all_sims_flat : (n_simulations, total_n) float64
        Pre-generated and sorted random points for each simulation.
    sim_offsets : (n_windows + 1,) int64
        Window boundary offsets into each row of all_sims_flat.
    """
    n_r = radii.size
    K_sims = np.empty((n_simulations, n_r), np.float64)

    for s in prange(n_simulations):
        numerator, denom = _accumulate_translation(
            all_sims_flat[s],
            sim_offsets,
            window_lengths,
            radii,
        )
        if denom <= 0:
            K_sims[s, :] = np.nan
        else:
            K_sims[s, :] = numerator / denom

    return K_sims


def _simulate_envelopes(
    trial_sizes: list[int],
    window_lengths: NDArray[np.float64],
    radii: NDArray[np.float64],
    edge_correction: str,
    n_simulations: int,
    total_n: int,
    total_dur: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Monte Carlo K under CSR, shape ``(n_simulations, n_r)``."""
    n_r = radii.size
    offsets = np.zeros(len(trial_sizes) + 1, dtype=np.int64)
    for i, n_m in enumerate(trial_sizes):
        offsets[i + 1] = offsets[i] + n_m
    flat_n = int(offsets[-1])

    # Pre-generate all simulations: draw uniform, then sort per window
    all_flat = np.empty((n_simulations, flat_n), dtype=np.float64)
    for s in range(n_simulations):
        for m, (n_m, T) in enumerate(zip(trial_sizes, window_lengths)):
            start, end = int(offsets[m]), int(offsets[m + 1])
            all_flat[s, start:end] = np.sort(rng.uniform(0.0, T, size=n_m))

    if edge_correction == "translation":
        return _simulate_envelopes_translation(
            all_flat,
            offsets,
            window_lengths,
            radii,
            n_simulations,
        )

    # Border fallback (not parallelised)
    K_sims = np.empty((n_simulations, n_r), dtype=np.float64)
    for s in range(n_simulations):
        K_sims[s, :], _ = _k_border(
            all_flat[s],
            offsets,
            window_lengths,
            radii,
            total_n,
            total_dur,
        )
    return K_sims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ripley_k(
    observations: Sequence[ArrayLike],
    radii: ArrayLike | None = None,
    window_lengths: float | Sequence[float] | NDArray[np.float64] = 1.0,
    *,
    edge_correction: Literal["translation", "border"] = "translation",
    n_r: int = 200,
    assume_sorted: bool = False,
) -> RipleyResult:
    """Estimate Ripley's K and L for replicated 1D point processes.

    Parameters
    ----------
    observations
        List of per-trial 1D event-time arrays.  Times should be in
        ``[0, T]`` where *T* is the corresponding window length.
    radii
        Evaluation distances (sorted, non-negative).  If ``None``,
        *n_r* linearly spaced values from 0 to min(T)/4 are generated.
    window_lengths
        Scalar (same for all trials) or per-trial array of window
        durations.
    edge_correction
        ``"translation"`` (default, recommended) or ``"border"``.
    n_r
        Number of evaluation radii when *radii* is ``None``.
    assume_sorted
        Skip sorting of input event times.

    Returns
    -------
    RipleyResult
    """
    if len(observations) == 0:
        raise ValueError("observations must not be empty.")

    T = _normalise_window_lengths(len(observations), window_lengths)
    trials = _prepare_trials(observations, T, assume_sorted)
    flat, offsets = _flatten_trials(trials)
    total_n = int(offsets[-1])
    total_dur = float(T.sum())

    if total_n < 2:
        raise ValueError("Need at least 2 points across all trials.")

    if radii is None:
        r = _auto_radii(T, n_r)
    else:
        r = np.asarray(radii, dtype=np.float64).ravel()
        if r.size == 0:
            raise ValueError("radii must be non-empty.")

    if edge_correction == "translation":
        K, denom = _k_translation(flat, offsets, T, r)
    elif edge_correction == "border":
        K, denom = _k_border(flat, offsets, T, r, total_n, total_dur)
    else:
        raise ValueError(f"Unknown edge_correction: {edge_correction!r}")

    L_unc = K / 2.0
    L = L_unc - r

    return RipleyResult(
        r=r,
        K=K,
        L=L,
        L_uncentered=L_unc,
        edge_correction=edge_correction,
        intensity=total_n / total_dur,
        n_points=total_n,
        observation_duration=total_dur,
        denominator=denom,
    )


def ripley_k_envelope(
    observations: Sequence[ArrayLike],
    radii: ArrayLike | None = None,
    window_lengths: float | Sequence[float] | NDArray[np.float64] = 1.0,
    *,
    edge_correction: Literal["translation", "border"] = "translation",
    n_r: int = 200,
    n_simulations: int = 99,
    alpha: float = 0.05,
    assume_sorted: bool = False,
    rng: np.random.Generator | None = None,
) -> RipleyEnvelopeResult:
    """Estimate Ripley's K/L with Monte Carlo confidence envelopes.

    Envelopes are computed under the null of homogeneous Poisson
    processes with the same number of points per window.

    Parameters
    ----------
    observations
        List of per-trial 1D event-time arrays.
    radii
        Evaluation distances.
    window_lengths
        Scalar or per-trial window durations.
    edge_correction
        ``"translation"`` or ``"border"``.
    n_r
        Number of radii when *radii* is ``None``.
    n_simulations
        Monte Carlo simulations for envelopes.
    alpha
        Significance level (two-sided).
    assume_sorted
        Skip sorting of input event times.
    rng
        Random number generator.

    Returns
    -------
    RipleyEnvelopeResult
    """
    if rng is None:
        rng = np.random.default_rng()

    result = ripley_k(
        observations,
        radii=radii,
        window_lengths=window_lengths,
        edge_correction=edge_correction,
        n_r=n_r,
        assume_sorted=assume_sorted,
    )

    n_r_out = result.r.size
    if n_simulations <= 0:
        nan_arr = np.full(n_r_out, np.nan, dtype=np.float64)
        return RipleyEnvelopeResult(
            r=result.r,
            K=result.K,
            L=result.L,
            L_uncentered=result.L_uncentered,
            K_lo=nan_arr.copy(),
            K_hi=nan_arr.copy(),
            L_lo=nan_arr.copy(),
            L_hi=nan_arr.copy(),
            edge_correction=result.edge_correction,
            intensity=result.intensity,
            n_points=result.n_points,
            observation_duration=result.observation_duration,
            denominator=result.denominator,
            n_simulations=0,
            alpha=alpha,
        )

    T = _normalise_window_lengths(len(observations), window_lengths)
    trials = _prepare_trials(observations, T, assume_sorted)
    trial_sizes = [len(tr) for tr in trials]

    K_sims = _simulate_envelopes(
        trial_sizes,
        T,
        result.r,
        edge_correction,
        n_simulations,
        result.n_points,
        result.observation_duration,
        rng,
    )
    L_sims = K_sims / 2.0 - result.r[np.newaxis, :]

    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    return RipleyEnvelopeResult(
        r=result.r,
        K=result.K,
        L=result.L,
        L_uncentered=result.L_uncentered,
        K_lo=np.percentile(K_sims, lo_pct, axis=0),
        K_hi=np.percentile(K_sims, hi_pct, axis=0),
        L_lo=np.percentile(L_sims, lo_pct, axis=0),
        L_hi=np.percentile(L_sims, hi_pct, axis=0),
        edge_correction=result.edge_correction,
        intensity=result.intensity,
        n_points=result.n_points,
        observation_duration=result.observation_duration,
        denominator=result.denominator,
        n_simulations=n_simulations,
        alpha=alpha,
    )


def compare_ripley_l(
    processes: dict[str, Sequence[ArrayLike]],
    radii: ArrayLike | None = None,
    window_lengths: float | Sequence[float] | NDArray[np.float64] = 1.0,
    *,
    edge_correction: Literal["translation", "border"] = "translation",
    n_r: int = 200,
    n_simulations: int = 99,
    alpha: float = 0.05,
    assume_sorted: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, RipleyEnvelopeResult]:
    """Compare Ripley's L across multiple point processes.

    Computes :func:`ripley_k_envelope` for each named process using
    shared *radii* and *window_lengths*.

    Parameters
    ----------
    processes
        Dict mapping names to per-trial observation lists.  Example::

            {"contributing": [contrib_trial1, contrib_trial2, ...],
             "full": [full_trial1, full_trial2, ...],
             "licks": [lick_trial1, lick_trial2, ...]}

    radii, window_lengths, edge_correction, n_r, n_simulations,
    alpha, assume_sorted, rng
        Passed to :func:`ripley_k_envelope`.

    Returns
    -------
    dict[str, RipleyEnvelopeResult]
    """
    if rng is None:
        rng = np.random.default_rng()
    if not processes:
        raise ValueError("processes dict must not be empty.")

    # Resolve shared radii from the first process if not provided
    if radii is None:
        first_obs = next(iter(processes.values()))
        T = _normalise_window_lengths(len(first_obs), window_lengths)
        r = _auto_radii(T, n_r)
    else:
        r = np.asarray(radii, dtype=np.float64).ravel()

    results: dict[str, RipleyEnvelopeResult] = {}
    for name, obs in processes.items():
        results[name] = ripley_k_envelope(
            obs,
            radii=r,
            window_lengths=window_lengths,
            edge_correction=edge_correction,
            n_simulations=n_simulations,
            alpha=alpha,
            assume_sorted=assume_sorted,
            rng=rng,
        )

    return results
