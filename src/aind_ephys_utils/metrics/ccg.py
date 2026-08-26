"""Cross-correlogram (CCG) computation for spike trains.

Requires the ``numba`` optional dependency for the core kernel
(install with ``pip install aind-ephys-utils[numba]``).

Lag convention
--------------
The CCG ``C[i, j]`` is computed as the histogram of ``t_j - t_i``:

* **Positive lag** → unit *j* fires *after* unit *i* → *i* drives *j*.
* **Negative lag** → unit *j* fires *before* unit *i* → *j* drives *i*.

So a peak at positive lag in ``C[i, j]`` is evidence that *i* → *j*
(e.g., a monosynaptic excitatory connection from *i* to *j*).
``C[j, i]`` is the time-reverse: ``C[j, i, k] == C[i, j, -k]``.

The kernel uses left-closed bin assignment (``floor``) matching the convention
``np.histogram``.  The ``"corrcoef"`` normalization mode subtracts
edge-corrected expected counts and divides by ``sqrt(auto_u * auto_v)`` to
produce a proper correlation coefficient, matching the original Julia
implementation of these functions.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_prominences, peak_widths

from aind_ephys_utils._numba import njit, prange
from aind_ephys_utils.standards.conventions import C
from aind_ephys_utils.standards.validate import validate

# ---------------------------------------------------------------------------
# Xarray-compatible CCG interface
# ---------------------------------------------------------------------------


def ccg(  # noqa: C901
    spikes: xr.DataArray,
    *,
    source_units: Iterable[Any] | None = None,
    target_units: Iterable[Any] | None = None,
    pairs: Iterable[tuple[Any, Any]] | np.ndarray | None = None,
    pairing: np.ndarray | Iterable[int] | None = None,
    window: tuple[float, float] | None = None,
    bin_size: float = 0.001,
    max_lag: float = 0.1,
    normalize: str = "none",
    exclude_zero_lag_autocorr: bool = True,
    include_autocorr: bool = True,
) -> tuple[np.ndarray, xr.DataArray]:
    """Compute CCGs directly from a canonical ragged spikes DataArray.

    Single-trial ragged spikes dispatch to the session-wide CCG helpers.
    Multi-trial ragged spikes are packed internally and dispatched to the
    trial-paired CCG helper. Unit selectors are resolved by coordinate label
    when possible and by integer position otherwise.
    """
    spikes_ut = _normalize_ragged_spikes(spikes)
    labels = _unit_labels(spikes_ut)
    n_trials = spikes_ut.sizes[C.trial]
    window_use = _resolve_window(spikes_ut, window)

    if pairs is not None and (
        source_units is not None or target_units is not None
    ):
        raise ValueError(
            "Use either pairs or source_units/target_units, not both."
        )

    attrs = {
        "ephys.metric": "ccg",
        "bin_size": float(bin_size),
        "max_lag": float(max_lag),
        "normalize": normalize,
        "window": window_use,
    }

    if pairs is not None:
        pair_idx, pair_src_labels, pair_tgt_labels = _resolve_unit_pairs(
            spikes_ut, pairs
        )
    else:
        source_idx = _resolve_unit_indices(spikes_ut, source_units)
        target_idx = _resolve_unit_indices(spikes_ut, target_units)
        pair_idx = np.array(
            [(i, j) for i in source_idx for j in target_idx],
            dtype=np.int64,
        )

    selected = (
        pairs is not None
        or source_units is not None
        or target_units is not None
    )

    if n_trials == 1:
        if pairs is not None:
            unique_src = np.unique(pair_idx[:, 0])
            unique_tgt = np.unique(pair_idx[:, 1])
            src_lookup = {int(idx): pos for pos, idx in enumerate(unique_src)}
            tgt_lookup = {int(idx): pos for pos, idx in enumerate(unique_tgt)}
            lags, dense = ccg_between_sets_sparse(
                _spike_list_for_units(
                    spikes_ut, unique_src, window=window_use
                ),
                _spike_list_for_units(
                    spikes_ut, unique_tgt, window=window_use
                ),
                bin_size=bin_size,
                max_lag=max_lag,
                normalize=normalize,
                observation_window=window_use,
            )
            out = np.zeros(
                (pair_idx.shape[0], dense.shape[-1]), dtype=dense.dtype
            )
            for p, (i, j) in enumerate(pair_idx):
                out[p] = dense[src_lookup[int(i)], tgt_lookup[int(j)]]
            return lags, _wrap_pair_ccg(
                out, lags, pair_src_labels, pair_tgt_labels, attrs
            )

        if not selected:
            all_idx = np.arange(spikes_ut.sizes[C.unit])
            lags, out = ccg_allpairs_sparse(
                _spike_list_for_units(spikes_ut, all_idx, window=window_use),
                bin_size=bin_size,
                max_lag=max_lag,
                normalize=normalize,
                exclude_zero_lag_autocorr=exclude_zero_lag_autocorr,
                include_autocorr=include_autocorr,
                observation_window=window_use,
            )
            return lags, _wrap_dense_ccg(out, lags, labels, labels, attrs)

        lags, out = ccg_between_sets_sparse(
            _spike_list_for_units(spikes_ut, source_idx, window=window_use),
            _spike_list_for_units(spikes_ut, target_idx, window=window_use),
            bin_size=bin_size,
            max_lag=max_lag,
            normalize=normalize,
            observation_window=window_use,
        )
        return lags, _wrap_dense_ccg(
            out, lags, labels[source_idx], labels[target_idx], attrs
        )

    if pairing is None:
        trial_pairing = np.arange(n_trials, dtype=np.int64)
    else:
        trial_pairing = np.asarray(list(pairing), dtype=np.int64)
    if trial_pairing.shape != (n_trials,):
        raise ValueError(
            f"pairing must have shape ({n_trials},), got {trial_pairing.shape}."
        )

    trial_segments = _trial_segments_from_ragged_da(
        spikes_ut, window=window_use
    )

    if pairs is not None:
        lags, out = ccg_trial_paired(
            trial_segments,
            trial_pairing,
            bin_size=bin_size,
            max_lag=max_lag,
            normalize=normalize,
            exclude_zero_lag_autocorr=exclude_zero_lag_autocorr,
            include_autocorr=include_autocorr,
            pairs=pair_idx,
        )
        return lags, _wrap_pair_ccg(
            out, lags, pair_src_labels, pair_tgt_labels, attrs
        )

    if not selected:
        lags, out = ccg_trial_paired(
            trial_segments,
            trial_pairing,
            bin_size=bin_size,
            max_lag=max_lag,
            normalize=normalize,
            exclude_zero_lag_autocorr=exclude_zero_lag_autocorr,
            include_autocorr=include_autocorr,
        )
        return lags, _wrap_dense_ccg(out, lags, labels, labels, attrs)

    lags, out = ccg_trial_paired(
        trial_segments,
        trial_pairing,
        bin_size=bin_size,
        max_lag=max_lag,
        normalize=normalize,
        exclude_zero_lag_autocorr=exclude_zero_lag_autocorr,
        include_autocorr=include_autocorr,
        pairs=pair_idx,
    )
    return lags, _wrap_pair_ccg(
        out, lags, labels[pair_idx[:, 0]], labels[pair_idx[:, 1]], attrs
    )


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def _ccg_two_pointer(t1, t2, bin_size, nbins):
    """Numba-accelerated two-pointer cross-correlogram kernel.

    Uses left-closed bins: bin *k* covers ``[k*bin_size - bin_size/2,
    k*bin_size + bin_size/2)`` relative to zero lag.
    """
    h = np.zeros(nbins, dtype=np.int64)
    _ccg_two_pointer_accum(t1, t2, bin_size, nbins, h)
    return h


@njit(cache=True, fastmath=True)
def _ccg_two_pointer_accum(t1, t2, bin_size, nbins, out):
    """Accumulate cross-correlogram counts into *out* (no allocation)."""
    half = nbins // 2
    inv = 1.0 / bin_size
    # Derived from the bins actually returned, not from max_lag: when
    # max_lag / bin_size is not an integer, half rounds up and the outer bin
    # extends past max_lag + bin_size/2, so that bound would end the sweep
    # early and undercount the outermost bins.
    window_margin = (half + 0.5) * bin_size

    j_start = 0
    n2 = t2.size
    for i in range(t1.size):
        ti = t1[i]
        while j_start < n2 and t2[j_start] < ti - window_margin:
            j_start += 1
        j = j_start
        while j < n2 and t2[j] < ti + window_margin:
            dt = t2[j] - ti
            b = int(np.floor(dt * inv + 0.5))
            if -half <= b <= half:
                out[b + half] += 1
            j += 1


# ---------------------------------------------------------------------------
# Edge-corrected normalization helpers (ported from SpikeAnalysis.jl by Galen
# Lynch)
#
# The corrcoef normalization produces a unitless correlation coefficient in
# [-1, 1] that is zero when the two spike trains are independent and
# conditionally uniform within each trial.
#
# For a full derivation see the docstring of ``xcorr_discrete_normed`` in
# SpikeAnalysis.jl (sp_corrs.jl).  In brief:
#
#   C[b] = (h[b] - E[b]) / sqrt(A_u * A_v)
#
# where h[b] is the raw coincidence count accumulated across all trials,
# E[b] is the edge-corrected expected count under within-trial uniformity,
# and A_u, A_v are edge-corrected auto-correlation terms summed across
# trials.
#
# The expected count for a lag bin of width *w* whose far edge is at
# distance *binstop* from zero lag, given n_u reference and n_v target
# spikes in a trial of duration *d*, is:
#
#   E_trial[b] = scale[b] * w[b] * (d - binstop[b] + w[b]/2) / d^2
#                * n_u * n_v
#
# The (d - binstop + w/2) factor is the triangle / edge correction: at
# larger lags fewer reference-spike positions allow the target spike to
# land in the bin without exceeding the observation window.
#
# For the centre bin (lag ~ 0) the scale factor is 2 because coincidences
# arrive from both the positive and negative lag sides, and the effective
# bin width is halved (halfbin = bin_size / 2).
#
# Summing n_u * n_v per trial (rather than using total counts) respects
# trial-by-trial rate variation: trials where both neurons fire more
# contribute proportionally more expected coincidences.
# ---------------------------------------------------------------------------


def _expected_count(nu: int, nv: int, bin_size: float, dur: float) -> float:
    """Compute expected coincidence count under independence (no edge correction).

    Return ``bin_size * n_u * n_v / dur`` — the fraction of the observation
    window covered by one bin, times the number of spike pairs.
    """
    return bin_size * nu * nv / dur


def _expected_count_edge_corrected(
    binstop: float, nu: int, nv: int, bin_size: float, dur: float
) -> float:
    """Compute expected coincidence count with edge (triangle) correction.

    Parameters
    ----------
    binstop
        Distance from lag 0 to the far edge of the bin.
    nu, nv
        Number of reference and target spikes.
    bin_size
        Width of the lag bin.
    dur
        Observation window duration.

    The numerator ``bin_size * (dur - binstop + bin_size/2)`` is the area
    of valid reference positions times bin width.  Dividing by ``dur**2``
    gives the probability that a uniformly-drawn ``(u, v)`` pair lands in
    this bin.  Multiplying by ``nu * nv`` gives the expected count.
    """
    return (nu * nv * bin_size * (dur - binstop + bin_size / 2)) / dur**2


@njit(cache=True, fastmath=False)
def _count_auto_first(u, halfbin):
    """Count spike pairs within *halfbin* of each other (including self-pairs).

    Uses a forward pointer sweep (O(n)), matching Julia's ``count_auto_first``.
    """
    nu = u.size
    cnt = nu  # each spike pairs with itself
    j = 1
    for i in range(nu):
        if j <= i:
            j = i + 1
        while j < nu and u[j] < u[i] + halfbin:
            j += 1
        cnt += j - (i + 1)
    return cnt


def _corrected_auto_counts(
    u: np.ndarray, bin_size: float, dur: float, edgecorrect: bool = True
) -> float:
    """Auto-correlation normalization term for one spike train.

    Counts spike pairs within *bin_size* of each other (the "raw" auto
    term), then subtracts the expected count under uniformity.  The
    coefficient of 2 on the expected count (rather than 1) matches the
    cross-correlation context: the denominator ``sqrt(A_u * A_v)``
    must normalize the *two-sided* cross-histogram at lag 0, which
    receives contributions from both positive and negative lags.
    """
    basecount = _count_auto_first(u, bin_size)
    coeff = 2  # cross-correlation coefficient (not auto)
    if edgecorrect:
        return float(
            basecount
            - coeff
            * _expected_count_edge_corrected(
                bin_size, u.size, u.size, bin_size, dur
            )
        )
    return float(
        basecount - coeff * _expected_count(u.size, u.size, bin_size, dur)
    )


def _lag_geometry(
    nbins: int, bin_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin lag geometry: far edge, width, and centre-bin scale.

    The centre bin collects coincidences of both lag signs, so it is half as
    wide as a side bin and counted twice.
    """
    n_sidebins = (nbins - 1) // 2
    halfbin = bin_size / 2
    binstops = halfbin + np.abs(np.arange(nbins) - n_sidebins) * bin_size
    bin_widths = np.full(nbins, bin_size, dtype=np.float64)
    bin_widths[n_sidebins] = halfbin
    center_scale = np.ones(nbins, dtype=np.float64)
    center_scale[n_sidebins] = 2.0
    return binstops, bin_widths, center_scale


def _lag_exposure(nbins: int, bin_size: float, dur: float) -> np.ndarray:
    """Lag exposure ``Q_b`` over a trial of length *dur*, in time squared.

    ``Q_b`` is the area of the band of ``[0, dur]**2`` whose lag falls in bin
    *b* -- the triangle (edge) correction before any division by ``dur**2``.
    Naming it separately from that division is what lets trials of differing
    duration share one geometry.
    """
    binstops, bin_widths, center_scale = _lag_geometry(nbins, bin_size)
    return center_scale * bin_widths * (dur - binstops + bin_widths / 2)


def _expected_counts_shape(
    nbins: int, bin_size: float, dur: float
) -> np.ndarray:
    """Per-bin expected-count shape ``Q_b / dur**2``.

    The expected count for bin *k* given spike counts ``nu, nv`` is
    ``shape[k] * nu * nv``.  Independent of the spike counts, so it is
    computed once per duration and reused across all pairs.
    """
    return _lag_exposure(nbins, bin_size, dur) / dur**2


# Two supports count as matching within this many seconds.  Far below any
# spike-time resolution (30 kHz sampling is 33 us) and far above float64
# rounding on session-scale timestamps, so it admits arithmetic noise without
# admitting a real mismatch.
_SUPPORT_ATOL = 1e-9


def _is_involution(pairing: np.ndarray) -> bool:
    """True when ``sigma == sigma^-1``, i.e. ``sigma(sigma(k)) == k``.

    The dense mirror ``C[j, i] = reverse(C[i, j])`` and the diagonal's
    ``0.5 * (h + reverse(h))`` both compute ``h^{sigma^-1}`` where they mean
    ``h^sigma``; they agree only for an involution.  The identity is the case
    that made this look correct.
    """
    return bool(np.array_equal(pairing[pairing], np.arange(pairing.size)))


def _validate_pairing_support(
    ts: "TrialSegments", pairing: np.ndarray
) -> None:
    """Require each paired trial to share the partner trial's support.

    Different pairs may have different support; the two members of one pair
    may not.  Under this contract each pair's overlap equals its own support,
    so the overlap clip is a no-op and every trial's exposure is fixed by that
    trial alone -- which is what makes summed exposure invariant across
    pairings, and therefore raw-count surrogate comparison valid.

    Support equality is the condition, not duration equality: ``(-0.2, 0.8)``
    and ``(-0.8, 0.2)`` both last 1 s and share only 0.4 s.
    """
    bad = ~(
        np.isclose(ts.pre, ts.pre[pairing], rtol=0.0, atol=_SUPPORT_ATOL)
        & np.isclose(ts.post, ts.post[pairing], rtol=0.0, atol=_SUPPORT_ATOL)
    )
    if not bad.any():
        return
    k = int(np.flatnonzero(bad)[0])
    partner = int(pairing[k])
    raise ValueError(
        f"Paired trials must share the same alignment-relative support: "
        f"trial {k} spans ({-ts.pre[k]}, {ts.post[k]}) but its partner "
        f"{partner} spans ({-ts.pre[partner]}, {ts.post[partner]}). "
        f"{int(bad.sum())} of {bad.size} trials violate this. "
        "Call clip_to_window first to put every trial on a common support."
    )


def _uniform_support(ts: "TrialSegments") -> bool:
    """True when every trial spans the same alignment-relative interval.

    Clipping each pair to its trials' overlap is a no-op only under this
    condition.  Equal *durations* are not enough: ``(-0.2, 0.8)`` and
    ``(-0.8, 0.2)`` both last 1 s but overlap in only 0.4 s, so a pairing
    that maps one onto the other must clip.
    """
    return bool(np.all(ts.pre == ts.pre[0]) and np.all(ts.post == ts.post[0]))


_TRIAL_NORMALIZE_MODES = frozenset({"none", "corrcoef"})


def _validate_trial_normalize(normalize: str) -> None:
    """Reject normalizations the trial-paired paths do not implement.

    The session-wide modes (``counts``/``rate``/``conditional``/``unbiased``)
    assume one global duration and are not defined per trial pairing; they
    previously fell through and returned raw counts.
    """
    if normalize not in _TRIAL_NORMALIZE_MODES:
        raise ValueError(
            f"Unknown or unsupported normalize mode for the trial-paired "
            f"path: {normalize!r}. Supported: "
            f"{sorted(_TRIAL_NORMALIZE_MODES)}."
        )


def _validate_binning(bin_size: float, max_lag: float) -> None:
    """Reject bin geometries that make the lag index undefined."""
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive and finite, got {bin_size!r}."
        )
    if not np.isfinite(max_lag) or max_lag < 0:
        raise ValueError(
            f"max_lag must be non-negative and finite, got {max_lag!r}."
        )


def _validate_spike_trains(trains: list[np.ndarray]) -> None:
    """Require sorted, finite spike times.

    Unsorted input breaks the two-pointer sweep's monotonicity assumption and
    non-finite times propagate into the bin index as a silent miscount rather
    than an error.
    """
    for train in trains:
        s = np.asarray(train)
        if s.size == 0:
            continue
        if not np.all(np.isfinite(s)):
            raise ValueError("Spike times must be finite.")
        if np.any(np.diff(s) < 0):
            raise ValueError("Spike times must be sorted.")


# ---------------------------------------------------------------------------
# Between-set CCG
# ---------------------------------------------------------------------------


def ccg_between_sets_sparse(  # noqa: C901
    spike_times_set1: list[np.ndarray],
    spike_times_set2: list[np.ndarray],
    bin_size: float = 0.001,
    max_lag: float = 0.1,
    normalize: str = "none",
    observation_window: float | tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-lean CCGs between two sets of spike trains.

    Parameters
    ----------
    spike_times_set1
        M spike trains (reference/source units).
    spike_times_set2
        N spike trains (target units).
    bin_size
        Histogram bin width in seconds.
    max_lag
        Maximum time lag (symmetric around 0).
    normalize
        ``"none"``, ``"counts"``, ``"rate"``, ``"conditional"``, ``"unbiased"``, or
        ``"corrcoef"`` (edge-corrected correlation coefficient matching
        ``SpikeAnalysis.jl``).
    observation_window
        Total observation time.  Float for duration, tuple ``(start, end)``
        for explicit bounds, or ``None`` to infer from data (not recommended).

    Returns
    -------
    lags : (B,) float64
    C : (M, N, B) float32  (float64 when ``normalize="corrcoef"``)
    """
    M = len(spike_times_set1)
    N = len(spike_times_set2)

    _validate_binning(bin_size, max_lag)
    _validate_spike_trains(spike_times_set1 + spike_times_set2)

    T = _resolve_observation_window(
        observation_window, spike_times_set1 + spike_times_set2
    )

    half = int(round(max_lag / bin_size))
    B = 2 * half + 1
    lags = (np.arange(-half, half + 1) * bin_size).astype(np.float64)

    S1 = [s.astype(np.float64) for s in spike_times_set1]
    S2 = [s.astype(np.float64) for s in spike_times_set2]

    pair_list = np.array(
        [(i, j) for i in range(M) for j in range(N)], dtype=np.int64
    )
    out_buf = np.zeros((pair_list.shape[0], B), dtype=np.float64)

    @njit(parallel=True, cache=True, fastmath=False)
    def _compute_pairs(pair_idx, S1, S2, half, bin_size, out_buf):
        """Numba kernel: accumulate CCGs for a chunk of set1×set2 pairs."""
        nbins = 2 * half + 1
        for k in prange(pair_idx.shape[0]):
            i = pair_idx[k, 0]
            j = pair_idx[k, 1]
            if S1[i].size == 0 or S2[j].size == 0:
                continue
            out_buf[k, :] = _ccg_two_pointer(S1[i], S2[j], bin_size, nbins)

    _compute_pairs(pair_list, S1, S2, half, bin_size, out_buf)

    if normalize == "corrcoef":
        ec_shape = _expected_counts_shape(B, bin_size, T)
        auto_s1 = np.array(
            [_corrected_auto_counts(s, bin_size, T) for s in S1]
        )
        auto_s2 = np.array(
            [_corrected_auto_counts(s, bin_size, T) for s in S2]
        )
        nspikes_s1 = np.array([s.size for s in S1])
        nspikes_s2 = np.array([s.size for s in S2])
        C = np.zeros((M, N, B), dtype=np.float64)
        p = 0
        for i in range(M):
            for j in range(N):
                h = out_buf[p, :] - ec_shape * nspikes_s1[i] * nspikes_s2[j]
                denom = np.sqrt(abs(auto_s1[i] * auto_s2[j]))
                C[i, j, :] = h / denom if denom > 0 else 0.0
                p += 1
    else:
        C = np.zeros((M, N, B), dtype=np.float32)
        p = 0
        for i in range(M):
            for j in range(N):
                C[i, j, :] = out_buf[p, :].astype(np.float32)
                p += 1
        _apply_normalization(C, lags, normalize, T, S1, bin_size)

    return lags, C


# ---------------------------------------------------------------------------
# All-pairs CCG
# ---------------------------------------------------------------------------


def _scatter_and_normalize(
    out_buf: np.ndarray,
    pair_list_arr: np.ndarray,
    N: int,
    B: int,
    half: int,
    bin_size: float,
    T: float,
    normalize: str,
    exclude_zero_lag_autocorr: bool,
    include_autocorr: bool,
    S: list[np.ndarray],
) -> np.ndarray:
    """Scatter pair CCG histograms into (N, N, B) and apply normalization."""
    use_corrcoef = normalize == "corrcoef"
    ec_shape: np.ndarray | None = None
    auto_terms: np.ndarray | None = None
    nspikes: np.ndarray | None = None
    if use_corrcoef:
        ec_shape = _expected_counts_shape(B, bin_size, T)
        auto_terms = np.array(
            [_corrected_auto_counts(s, bin_size, T) for s in S]
        )
        nspikes = np.array([s.size for s in S])

    C = np.zeros((N, N, B), dtype=np.float64 if use_corrcoef else np.float32)

    p = 0
    for i in range(N):
        j_start = i if include_autocorr else i + 1
        for j in range(j_start, N):
            if use_corrcoef and ec_shape is not None and nspikes is not None:
                h = out_buf[p, :] - ec_shape * nspikes[i] * nspikes[j]
            else:
                h = out_buf[p, :].astype(np.float32)
            if use_corrcoef and auto_terms is not None:
                denom = np.sqrt(abs(auto_terms[i] * auto_terms[j]))
                if denom > 0:
                    h = h / denom

            # After normalization, matching the trial paths: one meaning,
            # one place.
            if i == j and exclude_zero_lag_autocorr:
                h[half] = 0.0

            C[i, j, :] = h
            if i != j:
                # No trial pairing here, so sigma is the identity and the
                # mirror is exact.  The diagonal is already symmetric.
                C[j, i, :] = h[::-1]
            p += 1

    return C


def ccg_allpairs_sparse(
    spike_times_by_unit: list[np.ndarray],
    bin_size: float = 0.001,
    max_lag: float = 0.1,
    normalize: str = "none",
    exclude_zero_lag_autocorr: bool = True,
    include_autocorr: bool = True,
    observation_window: float | tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-lean all-pairs CCGs using event-driven sweep.

    Parameters
    ----------
    spike_times_by_unit
        List of sorted spike time arrays, one per unit.
    bin_size
        Histogram bin width in seconds.
    max_lag
        Maximum time lag (symmetric around 0).
    normalize
        ``"none"``, ``"counts"``, ``"rate"``, ``"conditional"``, ``"unbiased"``, or
        ``"corrcoef"`` (edge-corrected correlation coefficient).
    exclude_zero_lag_autocorr
        Zero out the central bin for autocorrelograms.
    include_autocorr
        Compute autocorrelograms on the diagonal.
    observation_window
        Total observation time.

    Returns
    -------
    lags : (B,) float64
    C : (N, N, B) float32  (float64 when ``normalize="corrcoef"``)
        ``C[i,j](τ) = C[j,i](-τ)``
    """
    N = len(spike_times_by_unit)
    _validate_binning(bin_size, max_lag)
    _validate_spike_trains(spike_times_by_unit)

    T = _resolve_observation_window(observation_window, spike_times_by_unit)

    half = int(round(max_lag / bin_size))
    B = 2 * half + 1
    lags = (np.arange(-half, half + 1) * bin_size).astype(np.float64)

    S = [s.astype(np.float64) for s in spike_times_by_unit]

    pair_list = []
    for i in range(N):
        j_start = i if include_autocorr else i + 1
        for j in range(j_start, N):
            pair_list.append((i, j))
    pair_list_arr = np.array(pair_list, dtype=np.int64)

    out_buf = np.zeros((pair_list_arr.shape[0], B), dtype=np.float64)

    @njit(parallel=True, cache=True, fastmath=False)
    def _compute_pairs(pair_idx, S, half, bin_size, out_buf):
        """Numba kernel: accumulate CCGs for a chunk of all-pairs."""
        nbins = 2 * half + 1
        for k in prange(pair_idx.shape[0]):
            i = pair_idx[k, 0]
            j = pair_idx[k, 1]
            if S[i].size == 0 or S[j].size == 0:
                continue
            out_buf[k, :] = _ccg_two_pointer(S[i], S[j], bin_size, nbins)

    _compute_pairs(pair_list_arr, S, half, bin_size, out_buf)

    C = _scatter_and_normalize(
        out_buf,
        pair_list_arr,
        N,
        B,
        half,
        bin_size,
        T,
        normalize,
        exclude_zero_lag_autocorr,
        include_autocorr,
        S,
    )

    if normalize != "corrcoef":
        _apply_normalization(C, lags, normalize, T, S, bin_size)

    return lags, C


def _coerce_spike_vector(x: object) -> np.ndarray:
    """Coerce one ragged spike entry to a sorted 1D float array."""
    if x is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Ragged spike entries must be 1D, got {arr.shape}.")
    if arr.size and np.any(np.diff(arr) < 0):
        raise ValueError(
            "Spike times must be sorted within each ragged entry."
        )
    return arr


def _normalize_ragged_spikes(spikes: xr.DataArray) -> xr.DataArray:
    """Validate and transpose ragged spikes to (unit, trial)."""
    if not isinstance(spikes, xr.DataArray):
        raise TypeError("ccg expects an xarray.DataArray of ragged spikes.")
    validate(spikes, kind="spikes_ragged")
    return spikes.transpose(C.unit, C.trial)


def _resolve_window(
    spikes: xr.DataArray, window: tuple[float, float] | None
) -> tuple[float, float]:
    """Resolve a relative-time observation window."""
    if window is not None:
        left, right = window
    else:
        valid = spikes.attrs.get(C.attr_valid_intervals)
        if valid:
            left, right = valid[0]
        else:
            mins: list[float] = []
            maxs: list[float] = []
            for x in spikes.data.ravel():
                arr = _coerce_spike_vector(x)
                if arr.size:
                    mins.append(float(arr.min()))
                    maxs.append(float(arr.max()))
            if not mins:
                raise ValueError(
                    "window is required when ragged spikes are empty and "
                    "ephys.valid_intervals is absent."
                )
            left, right = min(mins), max(maxs)
    left = float(left)
    right = float(right)
    if left >= right:
        raise ValueError(
            f"window must be (left, right) with left < right, got {window}."
        )
    return left, right


def _unit_labels(spikes_ut: xr.DataArray) -> np.ndarray:
    """Return unit coordinate labels."""
    if C.unit in spikes_ut.coords:
        return np.asarray(spikes_ut[C.unit].values)
    return np.arange(spikes_ut.sizes[C.unit])


def _resolve_unit_indices(
    spikes_ut: xr.DataArray,
    units: Iterable[Any] | None,
) -> np.ndarray:
    """Resolve unit selectors by coordinate label when possible, else position."""
    labels = _unit_labels(spikes_ut)
    if units is None:
        return np.arange(spikes_ut.sizes[C.unit], dtype=np.int64)

    resolved: list[int] = []
    for unit in units:
        matches = np.flatnonzero(labels == unit)
        if matches.size:
            resolved.append(int(matches[0]))
            continue
        if isinstance(unit, (int, np.integer)):
            idx = int(unit)
            if idx < 0 or idx >= spikes_ut.sizes[C.unit]:
                raise IndexError(f"unit index {idx} is out of bounds.")
            resolved.append(idx)
            continue
        raise KeyError(f"Could not find unit label {unit!r}.")
    return np.asarray(resolved, dtype=np.int64)


def _resolve_unit_pairs(
    spikes_ut: xr.DataArray,
    pairs: Iterable[tuple[Any, Any]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve explicit unit pairs to integer positions and labels."""
    labels = _unit_labels(spikes_ut)
    pair_values = list(pairs)
    pair_idx = np.empty((len(pair_values), 2), dtype=np.int64)
    for p, (src, tgt) in enumerate(pair_values):
        pair_idx[p, 0] = _resolve_unit_indices(spikes_ut, [src])[0]
        pair_idx[p, 1] = _resolve_unit_indices(spikes_ut, [tgt])[0]
    return pair_idx, labels[pair_idx[:, 0]], labels[pair_idx[:, 1]]


def _spike_list_for_units(
    spikes_ut: xr.DataArray,
    unit_indices: np.ndarray,
    trial_index: int = 0,
    window: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Extract one spike train per selected unit for one trial.

    *window* is applied here rather than left to the observation-duration
    argument downstream: the sparse kernels use it only to size the edge
    correction, so spikes outside it would otherwise still be counted.
    """
    trains = [
        _coerce_spike_vector(spikes_ut.data[int(i), trial_index])
        for i in unit_indices
    ]
    if window is None:
        return trains
    left, right = window
    return [s[(s >= left) & (s < right)] for s in trains]


def _trial_segments_from_ragged_da(
    spikes_ut: xr.DataArray,
    *,
    window: tuple[float, float],
) -> TrialSegments:
    """Pack ragged (unit, trial) spikes into the current CCG kernel format."""
    left, right = window
    n_units = spikes_ut.sizes[C.unit]
    n_trials = spikes_ut.sizes[C.trial]
    segments: list[list[np.ndarray]] = []
    for i in range(n_units):
        unit_segments: list[np.ndarray] = []
        for k in range(n_trials):
            arr = _coerce_spike_vector(spikes_ut.data[i, k])
            lo = np.searchsorted(arr, left, side="left")
            hi = np.searchsorted(arr, right, side="left")
            unit_segments.append(arr[lo:hi].astype(np.float64, copy=False))
        segments.append(unit_segments)

    all_spikes, offsets, counts = _build_csr(segments, n_units, n_trials)
    durations = np.full(n_trials, right - left, dtype=np.float64)
    pre = np.full(n_trials, -left, dtype=np.float64)
    post = np.full(n_trials, right, dtype=np.float64)
    return TrialSegments(
        segments, durations, pre, post, all_spikes, offsets, counts
    )


def _wrap_dense_ccg(
    values: np.ndarray,
    lags: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    attrs: dict[str, Any],
) -> xr.DataArray:
    """Wrap dense source x target CCG output as xarray."""
    return xr.DataArray(
        values,
        dims=("source_unit", "target_unit", "lag"),
        coords={
            "source_unit": source_labels,
            "target_unit": target_labels,
            "lag": lags,
        },
        attrs=attrs,
        name="ccg",
    )


def _wrap_pair_ccg(
    values: np.ndarray,
    lags: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    attrs: dict[str, Any],
) -> xr.DataArray:
    """Wrap sparse pair x lag CCG output as xarray."""
    n_pairs = values.shape[0]
    return xr.DataArray(
        values,
        dims=("pair", "lag"),
        coords={
            "pair": np.arange(n_pairs),
            "source_unit": ("pair", source_labels),
            "target_unit": ("pair", target_labels),
            "lag": lags,
        },
        attrs=attrs,
        name="ccg",
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def smooth_ccgs(
    C: np.ndarray, bin_size: float, kernel_width: float = 0.005
) -> np.ndarray:
    """Smooth correlograms with a Gaussian kernel.

    Parameters
    ----------
    C : (N, N, B)
        Raw correlogram counts.
    bin_size
        Bin width in seconds.
    kernel_width
        Gaussian kernel std in seconds.
    """
    sigma_bins = kernel_width / bin_size
    return gaussian_filter1d(C, sigma=sigma_bins, axis=-1, mode="nearest")


def rescale_ccgs(
    C: np.ndarray, axis: int = -1, eps: float = 1e-12
) -> np.ndarray:
    """Rescale correlograms so each (i,j) slice has min=0 and max=1."""
    minv = C.min(axis=axis, keepdims=True)
    maxv = C.max(axis=axis, keepdims=True)
    rng = np.maximum(maxv - minv, eps)
    return (C - minv) / rng


# ---------------------------------------------------------------------------
# Trial-structured CCG
# ---------------------------------------------------------------------------


@njit(cache=True)
def _searchsorted_left(a, v):
    """Binary search: index of first element in sorted *a* that is >= *v*."""
    lo, hi = 0, a.size
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(parallel=True, cache=True)
def _ccg_all_pairs_trials(
    all_spikes,
    offsets,
    counts,
    overlap_left,
    overlap_right,
    pair_list,
    pairing,
    bin_size,
    nbins,
    out_hist,
    out_dur,
    skip_clipping,
):
    """Numba kernel: all pairs × all trials, prange over pairs.

    When *skip_clipping* is True, segments are used as-is (no searchsorted).
    Valid only when every trial shares the same alignment-relative support,
    so each pair's overlap equals that support.
    """
    n_trials = overlap_left.shape[0]
    # Every valid overlap contributes its duration to every pair -- silence
    # is a valid rate=0 observation -- so the total is pair-independent.
    total_od = 0.0
    for k in range(n_trials):
        od = overlap_right[k] - overlap_left[k]
        if od > 0.0:
            total_od += od
    for p in range(pair_list.shape[0]):
        out_dur[p] = total_od

    for p in prange(pair_list.shape[0]):
        i = pair_list[p, 0]
        j = pair_list[p, 1]
        for k in range(n_trials):
            pk = pairing[k]
            od = overlap_right[k] - overlap_left[k]
            if od <= 0.0:
                continue
            off_i = offsets[i, k]
            cnt_i = counts[i, k]
            off_j = offsets[j, pk]
            cnt_j = counts[j, pk]
            if skip_clipping:
                ni = cnt_i
                nj = cnt_j
                si = all_spikes[off_i : off_i + ni]
                sj = all_spikes[off_j : off_j + nj]
            else:
                ol = overlap_left[k]
                orr = overlap_right[k]
                si = all_spikes[off_i : off_i + cnt_i]
                lo_i = _searchsorted_left(si, ol)
                hi_i = _searchsorted_left(si, orr)
                ni = hi_i - lo_i
                sj = all_spikes[off_j : off_j + cnt_j]
                lo_j = _searchsorted_left(sj, ol)
                hi_j = _searchsorted_left(sj, orr)
                nj = hi_j - lo_j
                si = si[lo_i:hi_i]
                sj = sj[lo_j:hi_j]
            if ni == 0 or nj == 0:
                continue
            _ccg_two_pointer_accum(si, sj, bin_size, nbins, out_hist[p, :])


@njit(parallel=True, cache=True)
def _auto_terms_clipped(
    all_spikes,
    offsets,
    counts,
    overlap_left,
    overlap_right,
    bin_size,
    out_auto,
):
    """Compute per-unit, per-trial auto terms on overlap-clipped segments.

    Uses the same clipping bounds as ``_ccg_all_pairs_trials`` so the auto
    terms are consistent with the actual spike counts in the CCG.
    """
    N = offsets.shape[0]
    n_trials = offsets.shape[1]
    for idx in prange(N * n_trials):
        i = idx // n_trials
        k = idx % n_trials
        ol = overlap_left[k]
        orr = overlap_right[k]
        od = orr - ol
        if od <= 0.0:
            continue
        off = offsets[i, k]
        cnt = counts[i, k]
        si = all_spikes[off : off + cnt]
        lo = _searchsorted_left(si, ol)
        hi = _searchsorted_left(si, orr)
        n_clipped = hi - lo
        if n_clipped == 0:
            continue
        si_clip = si[lo:hi]
        basecount = _count_auto_first(si_clip, bin_size)
        ec = (
            n_clipped * n_clipped * bin_size * (od - bin_size + bin_size / 2.0)
        ) / (od * od)
        out_auto[i, k] = basecount - 2.0 * ec


class TrialSegments:
    """CSR-like storage for spike segments clipped to trials.

    Spike times are stored relative to each trial's alignment time
    (t=0 is the alignment point).  For a go-cue-aligned trial with
    ``window=(-1, 3)``, spikes range from ``-1`` to ``3``.

    Attributes
    ----------
    segments
        ``segments[unit][trial]`` — ragged list (for Python access).
    durations
        ``(n_trials,)`` trial durations (stop - start).
    pre
        ``(n_trials,)`` time before alignment point (align - start, >= 0).
    post
        ``(n_trials,)`` time after alignment point (stop - align, >= 0).
    all_spikes
        ``(total_spikes,)`` flat contiguous array (alignment-relative).
    offsets
        ``(n_units, n_trials)`` start index into *all_spikes*.
    counts
        ``(n_units, n_trials)`` spike count per segment.
    """

    __slots__ = (
        "segments",
        "durations",
        "pre",
        "post",
        "all_spikes",
        "offsets",
        "counts",
    )

    def __init__(
        self,
        segments: list[list[np.ndarray]],
        durations: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        all_spikes: np.ndarray,
        offsets: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        """Store CSR-style trial-segment arrays and metadata."""
        self.segments = segments
        self.durations = durations
        self.pre = pre
        self.post = post
        self.all_spikes = all_spikes
        self.offsets = offsets
        self.counts = counts


def _build_csr(
    segments: list[list[np.ndarray]], N: int, n_trials: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack ragged segments into flat CSR arrays."""
    total = sum(seg.size for unit_segs in segments for seg in unit_segs)
    all_spikes = np.empty(total, dtype=np.float64)
    offsets_arr = np.zeros((N, n_trials), dtype=np.int64)
    counts_arr = np.zeros((N, n_trials), dtype=np.int64)
    pos = 0
    for i in range(N):
        for k in range(n_trials):
            seg = segments[i][k]
            n = seg.size
            offsets_arr[i, k] = pos
            counts_arr[i, k] = n
            if n > 0:
                all_spikes[pos : pos + n] = seg
                pos += n
    return all_spikes, offsets_arr, counts_arr


def clip_spikes_to_trials(
    spike_times_by_unit: list[np.ndarray],
    trial_epochs: np.ndarray,
    align_times: np.ndarray | None = None,
) -> TrialSegments:
    """Clip each unit's spikes into per-trial segments, relative to an alignment time.

    Parameters
    ----------
    spike_times_by_unit
        Sorted spike time arrays, one per unit.
    trial_epochs
        ``(n_trials, 2)`` array of ``[start, stop]`` times.
    align_times
        ``(n_trials,)`` alignment times (t=0 in the output).
        Defaults to trial starts (left edge).
    """
    _validate_spike_trains(list(spike_times_by_unit))
    trial_epochs = np.asarray(trial_epochs, dtype=np.float64)
    starts = trial_epochs[:, 0]
    stops = trial_epochs[:, 1]
    durations = stops - starts
    n_trials = trial_epochs.shape[0]
    N = len(spike_times_by_unit)

    if align_times is None:
        align = starts.copy()
    else:
        align = np.asarray(align_times, dtype=np.float64)

    pre = align - starts  # time before alignment (>= 0)
    post = stops - align  # time after alignment (>= 0)

    # Build ragged segments relative to alignment time
    segments: list[list[np.ndarray]] = []
    for st in spike_times_by_unit:
        unit_segs: list[np.ndarray] = []
        for k in range(n_trials):
            lo = np.searchsorted(st, starts[k], side="left")
            hi = np.searchsorted(st, stops[k], side="left")
            seg = (st[lo:hi] - align[k]).astype(np.float64)
            unit_segs.append(seg)
        segments.append(unit_segs)

    all_spikes, offsets_arr, counts_arr = _build_csr(segments, N, n_trials)
    return TrialSegments(
        segments, durations, pre, post, all_spikes, offsets_arr, counts_arr
    )


def clip_to_window(
    ts: TrialSegments, window: tuple[float, float]
) -> TrialSegments:
    """Re-clip a :class:`TrialSegments` to a symmetric window around alignment.

    Parameters
    ----------
    ts
        Output of :func:`clip_spikes_to_trials`.
    window
        ``(left, right)`` in alignment-relative time, e.g. ``(-1.0, 2.0)``.
        Typically choose ``left = -min(ts.pre)`` and
        ``right = min(ts.post)`` for the maximal common window.

    Returns
    -------
    TrialSegments
        New instance with segments and CSR clipped to *window*.
        All trials have the same effective duration ``right - left``.
    """
    left, right = window
    if right <= left:
        raise ValueError(f"window must satisfy left < right, got {window!r}.")
    N = len(ts.segments)
    n_trials = len(ts.durations)
    new_dur = right - left

    # Re-slicing cannot recover spikes an earlier clip already dropped, so a
    # window wider than any trial's support would claim exposure that was
    # never observed.
    if left < -ts.pre.min() or right > ts.post.min():
        raise ValueError(
            f"window {window!r} exceeds the common support "
            f"({-ts.pre.min()}, {ts.post.min()}); clip_to_window can only "
            "narrow."
        )

    # Re-slice the existing CSR — no new spike array needed
    new_offsets = np.empty_like(ts.offsets)
    new_counts = np.empty_like(ts.counts)
    new_segments: list[list[np.ndarray]] = []
    for i in range(N):
        unit_segs: list[np.ndarray] = []
        for k in range(n_trials):
            off = ts.offsets[i, k]
            cnt = ts.counts[i, k]
            seg = ts.all_spikes[off : off + cnt]
            lo = np.searchsorted(seg, left, side="left")
            hi = np.searchsorted(seg, right, side="left")
            new_offsets[i, k] = off + lo
            new_counts[i, k] = hi - lo
            unit_segs.append(seg[lo:hi])
        new_segments.append(unit_segs)

    durations = np.full(n_trials, new_dur, dtype=np.float64)
    pre = np.full(n_trials, -left, dtype=np.float64)
    post = np.full(n_trials, right, dtype=np.float64)
    return TrialSegments(
        new_segments,
        durations,
        pre,
        post,
        ts.all_spikes,
        new_offsets,
        new_counts,
    )


class _CCGBuffers:
    """Pre-allocated output buffers for :func:`ccg_trial_paired`.

    Create once with :meth:`for_segments`, reuse across surrogate calls
    to avoid repeated 100+ MB allocations.
    """

    __slots__ = (
        "pair_arr",
        "out_hist",
        "out_dur",
        "auto_per_trial",
    )

    def __init__(
        self,
        n_pairs: int,
        nbins: int,
        n_units: int,
        n_trials: int,
        need_auto: bool = True,
    ) -> None:
        """Allocate reusable output buffers for trial-paired CCGs.

        *need_auto* is false for every normalization but ``corrcoef``, which
        is the only consumer of the per-trial auto terms.
        """
        self.pair_arr: np.ndarray | None = None
        self.out_hist = np.zeros((n_pairs, nbins), dtype=np.int64)
        self.out_dur = np.zeros(n_pairs, dtype=np.float64)
        self.auto_per_trial = np.zeros(
            (n_units, n_trials) if need_auto else (0, 0), dtype=np.float64
        )

    def zero(self) -> None:
        """Reset all buffers to zero (avoids reallocation)."""
        self.out_hist[:] = 0
        self.out_dur[:] = 0
        self.auto_per_trial[:] = 0

    @staticmethod
    def for_segments(
        ts: TrialSegments,
        bin_size: float,
        max_lag: float,
        include_autocorr: bool = True,
        pairs: np.ndarray | None = None,
        need_auto: bool = True,
    ) -> _CCGBuffers:
        """Create buffers sized for the given segments and parameters.

        Parameters
        ----------
        pairs
            Optional ``(n_pairs, 2)`` int64 array of ``(i, j)`` unit index
            pairs.  If ``None``, all upper-triangle pairs are used.
        need_auto
            Allocate the per-trial auto-term buffer.  Only ``corrcoef``
            reads it.
        """
        N = len(ts.segments)
        n_trials = len(ts.durations)
        half = int(round(max_lag / bin_size))
        B = 2 * half + 1
        if pairs is not None:
            pair_arr = np.asarray(pairs, dtype=np.int64)
        else:
            pair_list = []
            for i in range(N):
                j_start = i if include_autocorr else i + 1
                for j in range(j_start, N):
                    pair_list.append((i, j))
            pair_arr = (
                np.array(pair_list, dtype=np.int64)
                if pair_list
                else np.empty((0, 2), dtype=np.int64)
            )
        bufs = _CCGBuffers(pair_arr.shape[0], B, N, n_trials, need_auto)
        bufs.pair_arr = pair_arr
        return bufs


class CCGCounts:
    """A pair-major CCG result and the metadata needed to interpret it.

    Compute and transforms stay pair-major end to end; the dense
    ``(N, N, B)`` layout is an explicit projection via :func:`to_dense`,
    never something a compute function does inline.  That projection is
    ~2 GB at ``N=499, B=1001``, so it has to be asked for.

    Attributes
    ----------
    counts
        ``(n_pairs, B)`` per-pair histogram in ``pairs`` row order -- raw
        kernel counts, or any pair-major transform of them.
    lags
        ``(B,)`` lag-bin centres.
    pairs
        ``(n_pairs, 2)`` array of ``(i, j)`` unit indices.
    n_units
        ``N``: the side of the square a dense projection would fill.
    bin_size
        Lag bin width, in the timebase of *lags*.
    pairing
        Trial pairing ``sigma`` the result was computed under, or ``None``
        for a session-wide result.  :func:`to_dense` reads it to decide
        whether the mirror is defined at all.
    exposure
        ``(B,)`` lag exposure ``Q_b``, when one geometry describes every
        trial; ``None`` when trials differ in duration and none does.
    expected
        ``(n_pairs, B)`` expected counts under independence, when computed.
    """

    __slots__ = (
        "counts",
        "lags",
        "pairs",
        "n_units",
        "bin_size",
        "pairing",
        "exposure",
        "expected",
        "expected_factors",
        "expected_trial",
        "n_spikes",
        "durations",
        "auto_direct",
        "auto_paired",
    )

    def __init__(
        self,
        counts: np.ndarray,
        lags: np.ndarray,
        pairs: np.ndarray,
        n_units: int,
        bin_size: float,
        pairing: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        expected: np.ndarray | None = None,
        expected_factors: tuple[np.ndarray, np.ndarray] | None = None,
        expected_trial: (
            tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ) = None,
        n_spikes: np.ndarray | None = None,
        durations: np.ndarray | None = None,
        auto_direct: np.ndarray | None = None,
        auto_paired: np.ndarray | None = None,
    ) -> None:
        """Store a pair-major CCG result and its interpretation metadata."""
        self.counts = counts
        self.lags = lags
        self.pairs = pairs
        self.n_units = n_units
        self.bin_size = bin_size
        self.pairing = pairing
        self.exposure = exposure
        self.expected = expected
        self.expected_factors = expected_factors
        self.expected_trial = expected_trial
        self.n_spikes = n_spikes
        self.durations = durations
        self.auto_direct = auto_direct
        self.auto_paired = auto_paired

    def expected_array(self) -> np.ndarray:
        """Expected counts as a full ``(n_pairs, B)`` array.

        Stored factored as ``shape[b] * scale[p]`` whenever one lag geometry
        serves every trial, which is the common case and keeps the memory
        linear in pairs; this materializes it.
        """
        if self.expected is not None:
            return self.expected
        if self.expected_factors is not None:
            shape, scale = self.expected_factors
            return shape[np.newaxis, :] * scale[:, np.newaxis]
        if self.expected_trial is not None:
            return np.stack(
                [self.expected_row(p) for p in range(self.n_pairs)]
            )
        raise ValueError(
            "this result carries no expected counts; compute it with "
            "compute_ccg_counts, or pass an explicit baseline."
        )

    def expected_row(self, p: int) -> np.ndarray:
        """Expected counts for pair *p* alone, as a ``(B,)`` lag profile.

        Transforms use this rather than :meth:`expected_array` so their
        temporaries stay one lag profile instead of one per pair: the full
        array is ~1 GB at ``N=499, B=1001``.
        """
        if self.expected is not None:
            return self.expected[p]
        if self.expected_factors is not None:
            shape, scale = self.expected_factors
            return shape * scale[p]
        if self.expected_trial is not None:
            # Trials differ in duration, so E is not rank-1 in (lag, pair);
            # it stays factored through the per-trial spike counts.
            ec_weighted, trial_counts, pairing = self.expected_trial
            i, j = int(self.pairs[p, 0]), int(self.pairs[p, 1])
            cross = trial_counts[i, :] * trial_counts[j, pairing]
            return np.asarray(ec_weighted.T @ cross)
        raise ValueError(
            "this result carries no expected counts; compute it with "
            "compute_ccg_counts, or pass an explicit baseline."
        )

    def rates(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-pair firing rates ``(lambda_i, lambda_j)``, each ``(n_pairs,)``.

        Rates are ``n_spikes / duration`` over the observed support, so they
        are the rates the expected-count term is built from rather than
        whole-session rates.
        """
        if self.n_spikes is None or self.durations is None:
            raise ValueError(
                "rates need `n_spikes` and `durations`; this result was "
                "built without observation metadata."
            )
        dur = np.asarray(self.durations, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_i = self.n_spikes[self.pairs[:, 0]] / dur
            lam_j = self.n_spikes[self.pairs[:, 1]] / dur
        return lam_i, lam_j

    def _exposure(self) -> np.ndarray:
        """Lag exposure, or a clear error when no single geometry applies."""
        if self.exposure is None:
            raise ValueError(
                "this result has no single lag exposure: its trials differ "
                "in duration, so no one Q_b describes it.  Density "
                "normalizations require a common geometry (see "
                "clip_to_window)."
            )
        return self.exposure

    @property
    def n_pairs(self) -> int:
        """Number of pairs carried by this result."""
        return int(self.pairs.shape[0])

    def mirror_is_defined(self) -> bool:
        """Whether ``C[j, i] == reverse(C[i, j])`` holds for this result.

        Reversing a histogram computes ``h^{sigma^-1}``, while the
        transposed cell needs ``h^sigma``.  Those coincide only when
        ``sigma`` is an involution -- which the identity satisfies, and
        which a session-wide result (no pairing at all) satisfies
        vacuously.
        """
        return self.pairing is None or _is_involution(self.pairing)


def _alloc(counts: CCGCounts, out: np.ndarray | None) -> np.ndarray:
    """Return *out*, or a fresh ``(n_pairs, B)`` float64 buffer."""
    if out is not None:
        return out
    return np.empty(counts.counts.shape, dtype=np.float64)


def _baseline_row(
    counts: CCGCounts, baseline: np.ndarray | None, p: int
) -> np.ndarray:
    """Baseline lag profile for pair *p*, defaulting to expected counts."""
    return counts.expected_row(p) if baseline is None else baseline[p]


def cross_intensity(
    counts: CCGCounts, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Joint event intensity ``rho_ij(tau) = H_b / Q_b``, in Hz^2.

    The raw second-order product density: how often the pair fires at a
    given lag, per unit of the time-squared exposure that lag actually had.
    Its baseline under independence is ``lambda_i lambda_j``, not zero.
    """
    exposure = counts._exposure()
    res = _alloc(counts, out)
    for p in range(counts.n_pairs):
        res[p, :] = counts.counts[p] / exposure
    return res


def excess_density(
    counts: CCGCounts,
    baseline: np.ndarray | None = None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Excess joint intensity ``(H_b - B_b) / Q_b`` over a baseline, in Hz^2.

    *baseline* defaults to the expected counts carried on the result -- the
    independence baseline -- which makes this the covariance density.  Pass
    a shift-predictor or surrogate-mean baseline for the general form.
    """
    exposure = counts._exposure()
    res = _alloc(counts, out)
    for p in range(counts.n_pairs):
        res[p, :] = (
            counts.counts[p] - _baseline_row(counts, baseline, p)
        ) / exposure
    return res


def covariance_density(
    counts: CCGCounts, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Covariance density ``c_ij(tau) = rho_ij - lambda_i lambda_j``, in Hz^2.

    :func:`excess_density` against the independence baseline, and the most
    literal answer to "how much more often than chance".  Zero under
    independence.
    """
    return excess_density(counts, None, out=out)


def normalized_covariance(
    counts: CCGCounts, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Rate-normalized covariance ``c_ij / sqrt(lambda_i lambda_j)``, in Hz.

    The recommended pair-comparable statistic: symmetric between the two
    units, free of the leading firing-rate dependence of sampling noise,
    independent of recording duration, and a density -- so it does not move
    with the display bin width.  It is *not* bounded to ``[-1, 1]``.
    """
    lam_i, lam_j = counts.rates()
    denom = np.sqrt(lam_i * lam_j)
    exposure = counts._exposure()
    res = _alloc(counts, out)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in range(counts.n_pairs):
            row = (counts.counts[p] - counts.expected_row(p)) / exposure
            res[p, :] = row / denom[p] if denom[p] > 0 else np.nan
    return res


def fold_over_baseline(
    counts: CCGCounts,
    baseline: np.ndarray | None = None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Ratio ``H_b / B_b`` of observed to baseline counts, dimensionless.

    *baseline* defaults to the expected counts on the result, making this
    the pair correlation.  1 means "as often as the baseline predicts".
    """
    res = _alloc(counts, out)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in range(counts.n_pairs):
            res[p, :] = counts.counts[p] / _baseline_row(counts, baseline, p)
    return res


def pair_correlation(
    counts: CCGCounts, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Pair correlation ``g_ij(tau) = rho_ij / (lambda_i lambda_j)``.

    :func:`fold_over_baseline` against the independence baseline.
    Dimensionless, and 1 rather than 0 under independence, so plots usually
    show ``g - 1``.
    """
    return fold_over_baseline(counts, None, out=out)


def legacy_auto_normalized(
    counts: CCGCounts, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Historical ``corrcoef`` statistic ``(H_b - E_b) / sqrt(A_i A_j)``.

    Retained for continuity with existing results and with
    ``SpikeAnalysis.jl``.  It is a shape statistic, not a correlation
    coefficient: reading it as one bounded to ``[-1, 1]`` is a category
    error, and its auto term carries a known zero-lag bias.  Prefer
    :func:`normalized_covariance` for new work.
    """
    if counts.auto_direct is None or counts.auto_paired is None:
        raise ValueError(
            "the legacy statistic needs auto terms; this result was built "
            "without them."
        )
    a_i = counts.auto_direct[counts.pairs[:, 0]]
    a_j = counts.auto_paired[counts.pairs[:, 1]]
    denom = np.sqrt(np.abs(a_i * a_j))
    res = _alloc(counts, out)
    for p in range(counts.n_pairs):
        row = counts.counts[p] - counts.expected_row(p)
        res[p, :] = row / denom[p] if denom[p] > 0 else row
    return res


def _window_weights(
    lags: np.ndarray, bin_size: float, window: tuple[float, float] | None
) -> np.ndarray | None:
    """Fraction of each lag bin lying inside *window*, or ``None`` for all.

    Every bin spans ``bin_size``, centred on its lag -- the centre bin
    included, which is why a window edge at zero splits it in half rather
    than falling between bins.
    """
    if window is None:
        return None
    low, high = window
    if not high > low:
        raise ValueError(f"window must satisfy low < high, got {window!r}.")
    half = bin_size / 2
    overlap = np.minimum(lags + half, high) - np.maximum(lags - half, low)
    return np.clip(overlap, 0.0, None) / bin_size


def directional_excess(
    counts: CCGCounts, window: tuple[float, float] | None = None
) -> np.ndarray:
    """Integrated excess target spikes per source spike over *window*.

    ``sum_{b in W} (H_b - E_b) / n_i`` -- one number per pair, not a lag
    profile.  This is the effect size to report: it is a count ratio, so
    unlike a peak height it does not move with the bin width, and summing
    over ``W`` is unaffected by how ``W`` is subdivided.  Report ``W``
    alongside it.

    *window* is a ``(low, high)`` lag interval, half-open on the right, and
    defaults to every lag.  Positive lags are the ``i -> j`` direction.

    Bins straddling a window edge are weighted by the fraction of their lag
    span that falls inside it, so *window* means the same interval at every
    bin width instead of snapping to whichever bin edges happen to exist.
    A window starting at zero therefore takes exactly half the centre bin,
    which is the right answer twice over: that bin spans ``[-w/2, +w/2)``,
    and its exposure is even in lag, so half of it is genuinely the
    ``i -> j`` side.

    What binning cannot do is resolve direction *within* a bin.  Half the
    centre bin is the neutral split, exact for the symmetric zero-lag
    synchrony that dominates that band, and it understates a true coupling
    whose delay is shorter than ``bin_size / 2``.  Resolving that needs a
    finer bin, not a different weighting.
    """
    if counts.n_spikes is None:
        raise ValueError(
            "directional_excess needs `n_spikes`; this result was built "
            "without observation metadata."
        )
    weights = _window_weights(counts.lags, counts.bin_size, window)
    n_i = counts.n_spikes[counts.pairs[:, 0]].astype(np.float64)
    totals = np.empty(counts.n_pairs, dtype=np.float64)
    for p in range(counts.n_pairs):
        row = counts.counts[p] - counts.expected_row(p)
        totals[p] = row.sum() if weights is None else float(row @ weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(totals / n_i)


def to_dense(
    values: np.ndarray | None,
    counts: CCGCounts,
    *,
    fill: float = np.nan,
) -> np.ndarray:
    """Project a pair-major array into the dense ``(N, N, B)`` layout.

    This is the single owner of the mirror policy.  ``pairs`` carries only
    ``(i, j)`` with ``i <= j``; the transposed cell comes from the flip rule
    ``C[j, i] = reverse(C[i, j])``, which holds only when the trial pairing
    is an involution.  Under any other pairing ``C[j, i]`` is not
    recoverable from ``C[i, j]``, so it is filled with ``NaN`` rather than a
    plausible wrong number.

    Parameters
    ----------
    values
        ``(n_pairs, B)`` array to project; ``None`` projects
        ``counts.counts``.
    counts
        Supplies ``pairs``, ``n_units`` and ``pairing``.
    fill
        Value for cells that no pair covers.

    Returns
    -------
    np.ndarray
        ``(n_units, n_units, B)`` array.
    """
    vals = counts.counts if values is None else values
    pairs = counts.pairs
    n_units = counts.n_units
    out = np.full((n_units, n_units, vals.shape[1]), fill, dtype=np.float64)
    mirrorable = counts.mirror_is_defined()
    for p in range(pairs.shape[0]):
        i, j = int(pairs[p, 0]), int(pairs[p, 1])
        out[i, j, :] = vals[p]
        if i != j:
            out[j, i, :] = vals[p][::-1] if mirrorable else np.nan
    return out


def compute_ccg_counts(  # noqa: C901
    trial_segments: TrialSegments,
    pairing: np.ndarray,
    *,
    bin_size: float = 0.001,
    max_lag: float = 0.1,
    include_autocorr: bool = True,
    with_expected: bool = True,
    pairs: np.ndarray | None = None,
    buffers: _CCGBuffers | None = None,
) -> CCGCounts:
    """Compute raw pair-major CCG counts and their interpretation metadata.

    The entry point for the transform layer.  Nothing is normalized and
    nothing is projected to dense here::

        counts = compute_ccg_counts(segments, pairing)
        values = normalized_covariance(counts)
        dense = to_dense(values, counts)

    Expected counts are kept factored rather than materialized as
    ``(n_pairs, B)``, which would be ~1 GB at ``N=499, B=1001``; the
    transforms read them one lag profile at a time.

    Parameters
    ----------
    trial_segments
        Output of :func:`clip_spikes_to_trials`.
    pairing
        ``(n_trials,)`` trial pairing ``sigma``.  Identity gives the real
        CCG; a derangement gives a surrogate.
    bin_size, max_lag
        Lag binning, in the timebase of the spike times.
    include_autocorr
        Whether to include the ``(i, i)`` self-pairs in the pair list.
    with_expected
        Compute the expected-count and auto terms.  ``False`` returns raw
        counts alone, which is all a count-space surrogate comparison needs.
    pairs
        ``(n_pairs, 2)`` explicit pair list; defaults to the upper triangle.
    buffers
        Reusable :class:`_CCGBuffers`, for repeated calls at one geometry.

    Returns
    -------
    CCGCounts
        Raw counts plus lags, pairs, pairing, exposure, expected-count
        factors, per-unit spike counts, per-pair durations, and auto terms.
    """
    ts = trial_segments
    N = len(ts.segments)
    half = int(round(max_lag / bin_size))
    B = 2 * half + 1
    lags = (np.arange(-half, half + 1) * bin_size).astype(np.float64)

    pairing = np.asarray(pairing, dtype=np.int64)
    _validate_pairing_support(ts, pairing)

    # Overlap window per trial.  A no-op under the support contract; kept so
    # the kernel's clipped and unclipped branches stay interchangeable.
    overlap_left = np.maximum(-ts.pre, -ts.pre[pairing])
    overlap_right = np.minimum(ts.post, ts.post[pairing])

    # Use pre-allocated buffers or create new ones
    if buffers is not None:
        bufs = buffers
        if with_expected and bufs.auto_per_trial.size == 0:
            raise ValueError(
                "buffers were built with need_auto=False and cannot serve "
                "expected and auto terms."
            )
        bufs.zero()
    else:
        bufs = _CCGBuffers.for_segments(
            ts,
            bin_size,
            max_lag,
            include_autocorr,
            pairs=pairs,
            need_auto=with_expected,
        )

    pair_arr = bufs.pair_arr
    assert pair_arr is not None
    n_pairs = pair_arr.shape[0]

    # Skip clipping only when every trial shares the same support, so each
    # pair's overlap equals that support (e.g., after clip_to_window).
    skip_clipping = _uniform_support(ts)

    # Single numba call: all pairs × all trials
    if n_pairs > 0:
        _ccg_all_pairs_trials(
            ts.all_spikes,
            ts.offsets,
            ts.counts,
            overlap_left,
            overlap_right,
            pair_arr,
            pairing,
            bin_size,
            B,
            bufs.out_hist,
            bufs.out_dur,
            skip_clipping,
        )

    # Corrcoef normalization
    use_corrcoef = with_expected
    if use_corrcoef:
        bufs.auto_per_trial[:] = 0
        _auto_terms_clipped(
            ts.all_spikes,
            ts.offsets,
            ts.counts,
            overlap_left,
            overlap_right,
            bin_size,
            bufs.auto_per_trial,
        )
        overlap_valid = (overlap_right - overlap_left) > 0
        auto_masked = bufs.auto_per_trial * overlap_valid[np.newaxis, :]
        auto_sum_direct = auto_masked.sum(axis=1)
        auto_sum_paired = auto_masked[:, pairing].sum(axis=1)

    # Per-trial expected counts: sum_k ec_shape(d_k) * n_i(k) * n_j(σ(k))
    # Edge correction is a within-trial boundary effect, so must use the
    # per-trial duration and spike counts, not totals.  When durations are
    # uniform, ec_shape(d) factors out and we only need the cross-product
    # sum_k n_i(k) * n_j(σ(k)).
    if use_corrcoef:
        trial_counts = ts.counts  # (N, n_trials)
        overlap_durs = overlap_right - overlap_left  # (n_trials,)

        if skip_clipping:
            # Uniform durations: ec_shape factors out, precompute once
            d_uniform = float(overlap_durs[0])
            ec_shape_single = _expected_counts_shape(B, bin_size, d_uniform)
            # When ``pairs`` is set, only compute the cross-products
            # we'll actually read.  Avoids the int64 non-BLAS matmul
            # (~70 ms at N=499) and the unnecessary N²-cell materialise
            # when n_pairs ≪ N².  Falls back to dense matmul (with
            # float-promotion to enable BLAS) when n_pairs ≈ N² or
            # the sparse intermediate would exceed 256 MB.
            paired_counts = trial_counts[:, pairing]  # (N, n_trials)
            sparse_bytes = n_pairs * paired_counts.shape[1] * 8
            if pairs is not None and sparse_bytes < 256 * 1024 * 1024:
                cross_per_pair = (
                    trial_counts[pair_arr[:, 0]].astype(np.float64)
                    * paired_counts[pair_arr[:, 1]].astype(np.float64)
                ).sum(axis=1)
            else:
                cm_full = (
                    trial_counts.astype(np.float64)
                    @ paired_counts.astype(np.float64).T
                )
                cross_per_pair = cm_full[pair_arr[:, 0], pair_arr[:, 1]]
        else:
            # Non-uniform: ec_weighted[k] = ec_shape(d_k), a (n_trials, B)
            # array.  Exposure enters only through the overlap duration, and
            # matched supports make durations repeat, so build one shape per
            # distinct duration rather than one per trial.
            n_tr = trial_counts.shape[1]
            valid = overlap_durs > 0
            ec_weighted = np.zeros((n_tr, B), dtype=np.float64)
            uniq_d, inverse = np.unique(
                overlap_durs[valid], return_inverse=True
            )
            if uniq_d.size:
                shapes = np.stack(
                    [
                        _expected_counts_shape(B, bin_size, float(d))
                        for d in uniq_d
                    ]
                )
                ec_weighted[valid, :] = shapes[inverse]

    # When ``pairs`` is provided, allocate only the per-pair output.
    # Skipping the (N, N, B) materialisation is a 2 GB saving at
    # N=499, B=1001 — and lets the caller stay per-pair end-to-end
    # without having to project after the fact.
    return CCGCounts(
        counts=bufs.out_hist[:n_pairs, :].astype(np.float64),
        lags=lags,
        pairs=pair_arr,
        n_units=N,
        bin_size=bin_size,
        pairing=pairing,
        exposure=(
            # Counts are summed over trials, so the exposure they are a
            # density with respect to is too: one trial's Q_b times the
            # number of trials.  Using the per-trial value here makes every
            # density normalization wrong by that factor.
            _lag_exposure(B, bin_size, d_uniform) * ts.durations.size
            if use_corrcoef and skip_clipping
            else None
        ),
        expected_factors=(
            (ec_shape_single, cross_per_pair)
            if use_corrcoef and skip_clipping
            else None
        ),
        expected_trial=(
            (ec_weighted, trial_counts, pairing)
            if use_corrcoef and not skip_clipping
            else None
        ),
        n_spikes=ts.counts.sum(axis=1),
        durations=bufs.out_dur[:n_pairs].copy(),
        auto_direct=auto_sum_direct if use_corrcoef else None,
        auto_paired=auto_sum_paired if use_corrcoef else None,
    )


def ccg_trial_paired(  # noqa: C901
    trial_segments: TrialSegments,
    pairing: np.ndarray,
    *,
    bin_size: float = 0.001,
    max_lag: float = 0.1,
    normalize: str = "corrcoef",
    exclude_zero_lag_autocorr: bool = True,
    include_autocorr: bool = True,
    pairs: np.ndarray | None = None,
    buffers: _CCGBuffers | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CCGs with a given trial pairing.

    The pairs x trials loop runs entirely inside numba with ``prange``
    over pairs.  Use :func:`clip_spikes_to_trials` once to build the
    :class:`TrialSegments`, then call this function repeatedly with
    different *pairing* arrays.

    Parameters
    ----------
    trial_segments
        Output of :func:`clip_spikes_to_trials`.
    pairing
        ``(n_trials,)`` integer array.  ``pairing[k]`` is the trial index
        for the target unit when the reference unit uses trial *k*.
        Identity (``np.arange(n_trials)``) for raw CCG.
    bin_size
        Histogram bin width (seconds).
    max_lag
        Maximum lag (seconds).
    normalize
        ``"none"``, ``"counts"``, ``"rate"``, ``"conditional"``, ``"unbiased"``, or
        ``"corrcoef"``.
    exclude_zero_lag_autocorr
        Zero the central bin for autocorrelograms.
    include_autocorr
        Compute autocorrelograms on the diagonal.
    pairs
        Optional ``(n_pairs, 2)`` int64 array of ``(i, j)`` unit index
        pairs to compute.  If ``None``, computes all upper-triangle
        pairs.  Use this to restrict to e.g. cross-region pairs.
    buffers
        Pre-allocated buffers from :meth:`_CCGBuffers.for_segments`.
        If ``None``, buffers are allocated internally (slower for
        repeated calls).  Must have been created with the same *pairs*.

    Returns
    -------
    lags : (B,) float64
    C : float64
        Shape ``(N, N, B)`` when ``pairs`` is ``None`` (with auto/cross
        symmetric mirror).  Shape ``(n_pairs, B)`` when ``pairs`` is
        provided, with row ``p`` corresponding to the input pair
        ``pairs[p]``.  In the per-pair case the lower-triangle mirror
        is *not* applied; reverse ``C[p, :]`` to obtain the ``(j, i)``
        CCG.  This avoids the (N, N, B) materialisation entirely when
        callers only need a subset of pairs (e.g. cross-region pairs
        for surrogate testing).

    Notes
    -----
    **corrcoef normalization** (``normalize="corrcoef"``):

    Produces a unitless correlation coefficient that is zero when the two
    spike trains are independent and conditionally uniform within each
    trial.  Derived from first principles (see ``SpikeAnalysis.jl``
    ``xcorr_discrete_normed`` for the full derivation).

    For each pair ``(i, j)`` the normalized value at lag bin ``b`` is::

        C[b] = (h[b] - E[b]) / sqrt(A_i * A_j)

    **h[b]** is the raw coincidence count accumulated across all trials
    by the numba kernel (``_ccg_all_pairs_trials``).

    **E[b]** is the expected count under within-trial uniformity with
    edge correction::

        E[b] = ec_shape[b] * sum_k( n_i(k) * n_j(pairing[k]) )

    where ``ec_shape[b] = scale[b] * w[b] * (d - binstop[b] + w[b]/2) / d^2``
    and the sum is over trials ``k``.  The per-trial product
    ``n_i(k) * n_j(pairing[k])`` respects trial-by-trial rate variation:
    trials where both neurons fire more contribute proportionally more
    expected coincidences.  When trial durations are uniform (after
    ``clip_to_window``), ``ec_shape`` factors out and the per-trial
    products are computed as a single matrix multiply
    (``trial_counts @ paired_counts.T``).

    The centre bin receives coincidences from both positive and negative
    lags, so ``scale = 2`` and ``w = bin_size / 2`` for that bin;
    side bins have ``scale = 1`` and ``w = bin_size``.

    **A_i, A_j** are the edge-corrected auto-correlation terms, each
    summed across trials::

        A_i = sum_k [ count_auto_first(u_i_k, bin_size)
                      - 2 * E_auto(n_i_k, d_k) ]

    ``count_auto_first`` counts spike pairs within ``bin_size`` of each
    other (including self-pairs).  The factor of 2 on the expected auto
    count (rather than 1) matches the cross-correlation context: the
    denominator must normalize the two-sided cross-histogram at lag 0.
    """
    _validate_trial_normalize(normalize)
    result = compute_ccg_counts(
        trial_segments,
        pairing,
        bin_size=bin_size,
        max_lag=max_lag,
        include_autocorr=include_autocorr,
        with_expected=(normalize == "corrcoef"),
        pairs=pairs,
        buffers=buffers,
    )
    lags = result.lags
    use_corrcoef = normalize == "corrcoef"
    half = (lags.size - 1) // 2
    pair_arr = result.pairs

    C = legacy_auto_normalized(result) if use_corrcoef else result.counts
    # A pair with no observed overlap has no estimate; the expected-count
    # subtraction would otherwise leave -E there.
    C[result.durations <= 0, :] = 0.0
    # Blank after normalization, not before: subtracting the expected count
    # from an already-zeroed bin left -E/denom here while the session path
    # returned 0.
    if exclude_zero_lag_autocorr:
        C[pair_arr[:, 0] == pair_arr[:, 1], half] = 0.0

    if pairs is not None:
        return lags, C

    # Dense is a projection, not a second compute path.  ``fill=0.0`` keeps
    # the historical dense semantics: with ``pairs=None`` the pair list is
    # the upper triangle, so the only uncovered cells are the diagonal when
    # ``include_autocorr`` is False.
    return lags, to_dense(C, result, fill=0.0)


def ccg_trial_surrogates(  # noqa: C901
    trial_segments: TrialSegments,
    pairings: Iterable[np.ndarray],
    *,
    bin_size: float = 0.001,
    max_lag: float = 0.1,
    normalize: str = "corrcoef",
    exclude_zero_lag_autocorr: bool = True,
    include_autocorr: bool = True,
    pairs: np.ndarray | None = None,
    reduce: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Iterator[np.ndarray]:
    """Compute trial-paired CCGs for many pairings with double-buffered pipeline.

    Overlaps the numba kernel (worker thread) with Python
    scatter/normalization (main thread) for ~30% throughput improvement
    over sequential :func:`ccg_trial_paired` calls.

    Parameters
    ----------
    trial_segments
        Output of :func:`clip_spikes_to_trials`.
    pairings
        Iterable of ``(n_trials,)`` integer pairing arrays.
    bin_size
        Histogram bin width (seconds).
    max_lag
        Maximum lag (seconds).
    normalize
        Normalization mode.
    exclude_zero_lag_autocorr
        Zero the central bin for autocorrelograms.
    include_autocorr
        Compute autocorrelograms on the diagonal.
    reduce
        Optional reduction ``(lags, C) -> result``.  Applied during
        scatter so the full CCG is never returned.

    Yields
    ------
    np.ndarray
        The reduced result if *reduce* is provided, otherwise the
        per-pair CCG.

        When ``pairs`` is ``None``: the un-reduced CCG is shape
        ``(N, N, B)`` with the cross/auto symmetric structure, and
        ``reduce`` is called with that shape.

        When ``pairs`` is provided: the un-reduced CCG is shape
        ``(n_pairs, B)``, with row ``p`` corresponding to the input
        pair ``pairs[p]``.  ``reduce`` is called with this per-pair
        layout — much cheaper for ``np.max(C, axis=-1)``-style
        reductions when ``n_pairs ≪ N²``.  The lower triangle is not
        mirrored; reverse ``C[p, :]`` to obtain the ``(j, i)`` CCG
        from a ``(i, j)`` row.
    """
    _validate_binning(bin_size, max_lag)
    _validate_trial_normalize(normalize)
    ts = trial_segments
    N = len(ts.segments)
    half = int(round(max_lag / bin_size))
    B = 2 * half + 1
    lags = (np.arange(-half, half + 1) * bin_size).astype(np.float64)
    skip_clipping = _uniform_support(ts)
    use_corrcoef = normalize == "corrcoef"

    n_trials_seg = len(ts.durations)
    bufs_a = _CCGBuffers.for_segments(
        ts,
        bin_size,
        max_lag,
        include_autocorr,
        pairs=pairs,
        need_auto=use_corrcoef,
    )
    bufs_b = _CCGBuffers.for_segments(
        ts,
        bin_size,
        max_lag,
        include_autocorr,
        pairs=pairs,
        need_auto=use_corrcoef,
    )

    # --- Fast path for uniform durations ---
    # When all trials have equal duration (after clip_to_window), overlap
    # bounds and auto terms are identical across all pairings.  Precompute
    # once to avoid redundant work per surrogate.
    #
    # Correctness:
    #   ol/orr: max(-pre[k], -pre[σ(k)]) = -pre[0] when all pre equal.
    #   auto terms: depend only on (unit, trial, ol, orr) — no pairing.
    #   auto_sum: sum is permutation-invariant → same for direct & paired.
    #
    # With the out_dur fix (duration accumulates even for silent trials),
    # total_dur is now truly constant across pairs when durations are
    # uniform: total_dur = n_trials × (pre + post).  This means ec_buf
    # can also be precomputed.
    uniform_fast = skip_clipping and use_corrcoef
    if skip_clipping:
        uniform_ol = np.full(n_trials_seg, -ts.pre[0], dtype=np.float64)
        uniform_orr = np.full(n_trials_seg, ts.post[0], dtype=np.float64)
    if uniform_fast:
        # Auto terms: compute once
        auto_precomputed = np.zeros((N, n_trials_seg), dtype=np.float64)
        _auto_terms_clipped(
            ts.all_spikes,
            ts.offsets,
            ts.counts,
            uniform_ol,
            uniform_orr,
            bin_size,
            auto_precomputed,
        )
        # Sum is the same for direct and paired (permutation-invariant)
        uniform_auto_sum: np.ndarray = auto_precomputed.sum(axis=1)

        # Per-trial expected-count shape: ec_shape(d) is constant when
        # all trials have equal duration d.  The scatter loop multiplies
        # this by sum_k n_i(k)*n_j(σ(k)) per pair.
        uniform_d = float(ts.post[0] + ts.pre[0])
        uniform_ec: np.ndarray = _expected_counts_shape(B, bin_size, uniform_d)

    def _launch_kernel(
        bufs: _CCGBuffers,
        pairing: np.ndarray,
        ol: np.ndarray,
        orr: np.ndarray,
    ) -> None:
        """Zero buffers and run the numba kernel for one pairing."""
        bufs.zero()
        _ccg_all_pairs_trials(
            ts.all_spikes,
            ts.offsets,
            ts.counts,
            ol,
            orr,
            bufs.pair_arr,
            pairing,
            bin_size,
            B,
            bufs.out_hist,
            bufs.out_dur,
            skip_clipping,
        )
        if use_corrcoef and not uniform_fast:
            bufs.auto_per_trial[:] = 0
            _auto_terms_clipped(
                ts.all_spikes,
                ts.offsets,
                ts.counts,
                ol,
                orr,
                bin_size,
                bufs.auto_per_trial,
            )

    def _scatter(  # noqa: C901
        bufs: _CCGBuffers,
        pairing: np.ndarray,
        ol: np.ndarray,
        orr: np.ndarray,
    ) -> np.ndarray:
        """Normalize kernel output into per-pair peak values."""
        pair_arr = bufs.pair_arr
        assert pair_arr is not None
        n_pairs = pair_arr.shape[0]

        if use_corrcoef:
            if uniform_fast:
                auto_sum_i = uniform_auto_sum
                auto_sum_j = uniform_auto_sum
            else:
                overlap_valid = (orr - ol) > 0
                auto_masked = (
                    bufs.auto_per_trial * overlap_valid[np.newaxis, :]
                )
                auto_sum_i = auto_masked.sum(axis=1)
                auto_sum_j = auto_masked[:, pairing].sum(axis=1)

        trial_counts = ts.counts
        overlap_durs = orr - ol
        paired_counts = trial_counts[:, pairing]  # (N, n_trials)

        if use_corrcoef:
            if uniform_fast:
                # We need ``cross_matrix[i, j] = sum_k trial_counts[i, k] *
                # paired_counts[j, k]`` only for ``(i, j) ∈ pair_arr`` —
                # not the full N×N grid.  The original code computed the
                # full matmul, which is unnecessary work and, on int64
                # arrays, falls back to a non-BLAS generic loop that's
                # ~4x slower than the float64 BLAS path.  When pair_arr
                # is much smaller than N² (typical for surrogate testing
                # with a pre-filter or candidate set), gather + multiply
                # + sum on just the relevant rows is dramatically
                # cheaper than the full matmul.
                #
                # For large pair_arr (≈ N²) the dense path is faster
                # because BLAS amortizes per-flop better; we cast to
                # float64 there to enable BLAS dispatch.  Threshold at
                # 256 MB of intermediate to bound memory.
                n_p = pair_arr.shape[0]
                sparse_bytes = n_p * n_trials_seg * 8
                if sparse_bytes < 256 * 1024 * 1024:
                    # (n_p, n_trials) gather of int64 counts; promote to
                    # float64 for the multiply+sum so the result matches
                    # the existing dense-matmul return dtype.
                    cross_per_pair = (
                        trial_counts[pair_arr[:, 0]].astype(np.float64)
                        * paired_counts[pair_arr[:, 1]].astype(np.float64)
                    ).sum(axis=1)
                else:
                    # Dense matmul, float-promoted for BLAS.
                    cm_full = (
                        trial_counts.astype(np.float64)
                        @ paired_counts.astype(np.float64).T
                    )
                    cross_per_pair = cm_full[pair_arr[:, 0], pair_arr[:, 1]]
            else:
                # Non-uniform: one shape per distinct overlap duration.
                n_tr = trial_counts.shape[1]
                valid = overlap_durs > 0
                ec_weighted = np.zeros((n_tr, B), dtype=np.float64)
                uniq_d, inverse = np.unique(
                    overlap_durs[valid], return_inverse=True
                )
                if uniq_d.size:
                    shapes = np.stack(
                        [
                            _expected_counts_shape(B, bin_size, float(d))
                            for d in uniq_d
                        ]
                    )
                    ec_weighted[valid, :] = shapes[inverse]

        # When ``pairs`` is explicit (the caller passed a specific pair
        # list), construct only an ``(n_pairs, B)`` output instead of
        # the full ``(N, N, B)`` matrix.  The reduce callback then
        # operates on a per-pair-row array, which for surrogate testing
        # is a massive saving — e.g. for N=499, n_pairs=1730, the
        # ``np.max(C, axis=-1)`` reduce drops from ~41 ms (250k cells)
        # to <1 ms (1730 cells).  Caller is responsible for indexing
        # the result by pair index.
        #
        # When ``pairs`` is ``None`` (default), preserve the original
        # ``(N, N, B)`` behaviour with the auto/cross mirror so the
        # public API is unchanged for that path.
        C = np.zeros((n_pairs, B), dtype=np.float64)
        for p in range(n_pairs):
            i, j = pair_arr[p, 0], pair_arr[p, 1]
            total_dur = bufs.out_dur[p]
            if total_dur <= 0:
                continue
            h = bufs.out_hist[p, :].astype(np.float64)
            if use_corrcoef:
                if uniform_fast:
                    h -= uniform_ec * cross_per_pair[p]
                else:
                    cross_per_trial = trial_counts[i, :] * paired_counts[j, :]
                    h -= ec_weighted.T @ cross_per_trial
                denom = np.sqrt(abs(auto_sum_i[i] * auto_sum_j[j]))
                if denom > 0:
                    h /= denom
            if i == j and exclude_zero_lag_autocorr:
                h[half] = 0.0
            C[p, :] = h

        if pairs is None:
            C = to_dense(
                C,
                CCGCounts(
                    counts=C,
                    lags=lags,
                    pairs=pair_arr,
                    n_units=N,
                    bin_size=bin_size,
                    pairing=pairing,
                ),
                fill=0.0,
            )

        return reduce(lags, C) if reduce is not None else C

    # Pipeline — overlap bounds are constant when durations are uniform
    def _overlap_for(pairing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-pair overlap bounds for a trial pairing."""
        _validate_pairing_support(ts, pairing)
        if skip_clipping:
            return uniform_ol, uniform_orr
        return (
            np.maximum(-ts.pre, -ts.pre[pairing]),
            np.minimum(ts.post, ts.post[pairing]),
        )

    pairing_iter = iter(pairings)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        first_pairing = np.asarray(next(pairing_iter), dtype=np.int64)
        ol, orr = _overlap_for(first_pairing)
        future = executor.submit(
            _launch_kernel, bufs_a, first_pairing, ol, orr
        )
        future.result()

        prev_bufs = bufs_a
        prev_pairing = first_pairing
        prev_ol, prev_orr = ol, orr

        for next_pairing in pairing_iter:
            next_pairing = np.asarray(next_pairing, dtype=np.int64)
            next_ol, next_orr = _overlap_for(next_pairing)
            next_bufs = bufs_b if prev_bufs is bufs_a else bufs_a

            future = executor.submit(
                _launch_kernel,
                next_bufs,
                next_pairing,
                next_ol,
                next_orr,
            )
            yield _scatter(prev_bufs, prev_pairing, prev_ol, prev_orr)
            future.result()

            prev_bufs = next_bufs
            prev_pairing = next_pairing
            prev_ol, prev_orr = next_ol, next_orr

        yield _scatter(prev_bufs, prev_pairing, prev_ol, prev_orr)

    except StopIteration:
        pass
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------


def monte_carlo_pvalue(
    t_obs: np.ndarray,
    t_surr: np.ndarray,
    tail: str = "upper",
) -> np.ndarray:
    """Monte Carlo p-value with correct finite-sample correction.

    Implements ``p = (1 + sum(I(t_b >= t_obs))) / (B + 1)`` (Davison &
    Hinkley, 1997) which accounts for the observed statistic being one of
    the ``B + 1`` values under the null.

    Parameters
    ----------
    t_obs
        Observed test statistic, any shape.
    t_surr
        Surrogate statistics, shape ``(B, *t_obs.shape)`` where ``B`` is
        the number of surrogates.
    tail
        ``"upper"``  — p = P(T >= t_obs)  (right tail, e.g., peak height).
        ``"lower"``  — p = P(T <= t_obs)  (left tail).
        ``"both"``   — p = P(|T| >= |t_obs|)  (two-tailed).

    Returns
    -------
    np.ndarray
        p-values, same shape as *t_obs*.
    """
    B = t_surr.shape[0]
    if tail == "upper":
        count = np.sum(t_surr >= t_obs[np.newaxis], axis=0)
    elif tail == "lower":
        count = np.sum(t_surr <= t_obs[np.newaxis], axis=0)
    elif tail == "both":
        count = np.sum(np.abs(t_surr) >= np.abs(t_obs)[np.newaxis], axis=0)
    else:
        raise ValueError(
            f"tail must be 'upper', 'lower', or 'both', got {tail!r}"
        )
    return (1 + count) / (B + 1)


def _window_slice(
    ccg_1d: np.ndarray,
    lags_1d: np.ndarray,
    search_half_width: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a CCG slice to ``|lag| <= search_half_width``."""
    if search_half_width is None:
        return ccg_1d, lags_1d
    mask = np.abs(lags_1d) <= search_half_width
    return ccg_1d[mask], lags_1d[mask]


def measure_fwhm(
    ccg_1d: np.ndarray,
    lags_1d: np.ndarray,
    *,
    search_half_width: float | None = None,
    half_height_mode: str = "absolute",
) -> float:
    """Estimate FWHM of the tallest peak in a 1-D CCG slice.

    By default uses **absolute half-max** (the textbook FWHM
    definition: width at ``0.5 * peak_height``), which is stable for
    broad peaks and appropriate for shift-corrected CCGs whose
    baseline is already zero-mean.  Pass ``half_height_mode=
    "prominence"`` for the legacy scipy behaviour (width at
    ``peak_height − 0.5 * prominence``).

    No smoothing happens here — pass an already-smoothed slice if you
    want denoising.  Pre-smoothing once at the array level (e.g.
    :func:`scipy.ndimage.gaussian_filter1d` along the lag axis of an
    ``(N, N, K)`` shift-corrected CCG) is cheaper than per-pair
    smoothing and avoids edge effects from smoothing-after-windowing.

    Parameters
    ----------
    ccg_1d
        1-D correlogram values (e.g. one row of the corrcoef matrix).
    lags_1d
        Corresponding lag values (seconds).  Must be uniformly spaced.
    search_half_width
        If given, restrict measurement to lags within
        ``[-search_half_width, search_half_width]`` (same units as
        *lags_1d*).  Prevents distant baseline wiggles from
        bounding ``peak_widths``' outward walk.
    half_height_mode
        ``"absolute"`` (default) or ``"prominence"``.  See above.

    Returns
    -------
    float
        Full width at half maximum in the same units as *lags_1d*.
        Returns ``np.inf`` if no positive peak is found.
    """
    if half_height_mode not in {"absolute", "prominence"}:
        raise ValueError(
            "half_height_mode must be 'absolute' or 'prominence'; "
            f"got {half_height_mode!r}"
        )
    ccg_1d, lags_1d = _window_slice(ccg_1d, lags_1d, search_half_width)
    if ccg_1d.size == 0:
        return np.inf
    peak_idx = int(np.argmax(ccg_1d))
    peak_value = float(ccg_1d[peak_idx])
    if peak_value <= 0:
        return np.inf
    if half_height_mode == "absolute":
        # Override prominence so peak_widths evaluates at
        # ``peak_value − 0.5 * peak_value = 0.5 * peak_value`` —
        # textbook FWHM, independent of any shoulder structure.
        prominence_data = (
            np.array([peak_value]),
            np.array([0]),
            np.array([ccg_1d.size - 1]),
        )
        widths_samples, _, _, _ = peak_widths(
            ccg_1d,
            np.array([peak_idx]),
            rel_height=0.5,
            prominence_data=prominence_data,
        )
    else:
        widths_samples, _, _, _ = peak_widths(
            ccg_1d,
            np.array([peak_idx]),
            rel_height=0.5,
        )
    bin_spacing = float(lags_1d[1] - lags_1d[0]) if len(lags_1d) > 1 else 1.0
    return float(widths_samples[0] * bin_spacing)


def measure_prominence(
    ccg_1d: np.ndarray,
    lags_1d: np.ndarray,
    *,
    search_half_width: float | None = None,
) -> float:
    """Measure the prominence of the tallest peak in a 1-D CCG slice.

    Prominence quantifies how much a peak stands out from its
    surrounding baseline.  Uses :func:`scipy.signal.peak_prominences`.

    No smoothing happens here — pass an already-smoothed slice if you
    want denoising.  See :func:`measure_fwhm` for guidance.

    Parameters
    ----------
    ccg_1d
        1-D correlogram values (e.g. one row of the corrcoef matrix).
    lags_1d
        Corresponding lag values (seconds).
    search_half_width
        If given, restrict measurement to lags within
        ``[-search_half_width, search_half_width]``.

    Returns
    -------
    float
        Prominence of the tallest peak (same units as *ccg_1d*).
        Returns ``0.0`` if no positive peak is found.
    """
    ccg_1d, _ = _window_slice(ccg_1d, lags_1d, search_half_width)
    if ccg_1d.size == 0:
        return 0.0
    peak_idx = int(np.argmax(ccg_1d))
    if ccg_1d[peak_idx] <= 0:
        return 0.0
    prominences, _, _ = peak_prominences(ccg_1d, np.array([peak_idx]))
    return float(prominences[0])


def measure_per_pair(
    corr: np.ndarray,
    lags_1d: np.ndarray,
    pairs: Iterable[tuple[int, int]],
    measure_fn: Callable[..., float],
    *,
    n_units: int,
    fill: float,
    **measure_kwargs: Any,
) -> np.ndarray:
    """Apply a per-pair scalar shape measurement, mirrored across (i,j) and (j,i).

    Helper for sweeping :func:`measure_fwhm` or :func:`measure_prominence`
    (or any equivalent scalar-returning function with the
    ``(ccg_1d, lags_1d, **kwargs) -> float`` signature) over a list of
    pairs from an ``(N, N, K)`` CCG slice.  Both triangles get filled
    so downstream symmetric-mask code works in either direction.

    Parameters
    ----------
    corr
        ``(N, N, K)`` shift-corrected CCG slice (typically pre-smoothed
        — pass an already-smoothed array if you want denoising; this
        helper does no smoothing of its own).
    lags_1d
        ``(K,)`` lag axis matching the last dimension of ``corr``.
    pairs
        Iterable of ``(i, j)`` index tuples.  Pairs may be supplied in
        either triangle; both ``out[i,j]`` and ``out[j,i]`` are filled
        regardless.
    measure_fn
        A function with signature ``(ccg_1d, lags_1d, **kwargs) -> float``,
        e.g. :func:`measure_fwhm` or :func:`measure_prominence`.
    n_units
        ``N`` — used to allocate the output ``(N, N)`` array.
    fill
        Default value for un-measured entries (e.g. ``np.inf`` for FWHM,
        ``0.0`` for prominence so downstream threshold cuts behave
        correctly on missing data).
    **measure_kwargs
        Forwarded to ``measure_fn`` on every call.

    Returns
    -------
    np.ndarray
        ``(N, N)`` of measurements; entries not in ``pairs`` (and not
        their mirror) hold ``fill``.
    """
    out = np.full((n_units, n_units), fill, dtype=float)
    for i, j in pairs:
        out[i, j] = measure_fn(corr[i, j, :], lags_1d, **measure_kwargs)
        out[j, i] = measure_fn(corr[j, i, :], lags_1d, **measure_kwargs)
    return out


def jitter_spikes_times(
    spike_times_by_unit: list[np.ndarray],
    jitter_window: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Add uniform jitter to each spike train and re-sort.

    Parameters
    ----------
    spike_times_by_unit
        List of sorted spike time arrays.
    jitter_window
        Half-width of the jitter window in seconds.
    rng
        NumPy random generator.  If ``None``, a new default generator is used.
    """
    if rng is None:
        rng = np.random.default_rng()
    out: list[np.ndarray] = []
    for st in spike_times_by_unit:
        jittered = st + rng.uniform(
            -jitter_window, jitter_window, size=len(st)
        )
        out.append(np.sort(jittered))
    return out


def cc_leads_over_std(
    lags: np.ndarray,
    C: np.ndarray,
    std_mult: float = 4,
    window_excl: float = 0.05,
    remove_auto: bool = False,
) -> np.ndarray:
    """Detect unit pairs whose short-lag CCG peak exceeds a noise-floor threshold.

    The baseline std is estimated from bins *outside* ``[-window_excl, window_excl]``,
    and then only the bins *inside* that window are tested against the threshold.

    Parameters
    ----------
    lags : (B,)
    C : (N, N, B)
    std_mult
        Number of standard deviations above baseline.
    window_excl
        Half-width around zero lag (seconds).  Bins outside this window are
        used as the baseline; bins inside are tested for significant peaks.
    remove_auto
        Zero out the diagonal (auto-correlations).

    Returns
    -------
    X : (N, N) bool-like
        ``X[i, j] == 1`` when a bin of ``C[i, j]`` inside the short-lag
        window exceeds the threshold **and** the peak lag is negative.
    """
    baseline_mask = (lags < -window_excl) | (lags > window_excl)
    short_lag_mask = ~baseline_mask
    mx_lag = np.empty(C.shape[0:2])
    mx_pass = np.empty(C.shape[0:2])
    for n1 in range(C.shape[0]):
        for n2 in range(C.shape[1]):
            ccg = C[n1, n2, :]
            threshold = np.std(ccg[baseline_mask]) * std_mult
            short_lag_bins = ccg[short_lag_mask]
            mx_lag[n1, n2] = lags[short_lag_mask][np.argmax(short_lag_bins)]
            mx_pass[n1, n2] = np.any(short_lag_bins > threshold)
    X = mx_pass * (mx_lag < 0)
    if remove_auto:
        X[np.where(np.eye(len(X)))] = 0
    return X


@njit(cache=True)
def _peak_contributing_mask(t_pre, t_post, lag_lo, lag_hi):
    """Two-pointer sweep: mark pre spikes with a post partner in [lag_lo, lag_hi]."""
    n_pre = t_pre.size
    n_post = t_post.size
    mask = np.zeros(n_pre, dtype=np.bool_)
    j_lo = 0
    for i in range(n_pre):
        target_lo = t_pre[i] + lag_lo
        target_hi = t_pre[i] + lag_hi
        while j_lo < n_post and t_post[j_lo] < target_lo:
            j_lo += 1
        if j_lo < n_post and t_post[j_lo] <= target_hi:
            mask[i] = True
    return mask


def peak_contributing_spikes(
    t_pre: np.ndarray,
    t_post: np.ndarray,
    peak_lag: float,
    half_width: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Find pre-synaptic spikes that have a post-synaptic partner near *peak_lag*.

    Uses a numba-accelerated two-pointer sweep for O(n) runtime.

    Parameters
    ----------
    t_pre
        Sorted spike times of the pre-synaptic (driving) unit.
    t_post
        Sorted spike times of the post-synaptic (following) unit.
    peak_lag
        Expected lag in seconds (positive = post fires after pre).
    half_width
        Tolerance around *peak_lag* in seconds.  A spike pair counts if
        ``|dt - peak_lag| <= half_width`` where ``dt = t_post - t_pre``.
        Typically set to the CCG bin width or the FWHM of the peak.

    Returns
    -------
    contributing : np.ndarray
        Pre-synaptic spike times that have a matching post-synaptic spike.
    mask : np.ndarray
        Boolean mask into *t_pre* (``contributing == t_pre[mask]``).
    """
    t_pre = np.asarray(t_pre, dtype=np.float64)
    t_post = np.asarray(t_post, dtype=np.float64)
    mask = _peak_contributing_mask(
        t_pre, t_post, peak_lag - half_width, peak_lag + half_width
    )
    return t_pre[mask], mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_observation_window(
    observation_window: float | tuple[float, float] | None,
    spike_trains: list[np.ndarray],
) -> float:
    """Resolve *observation_window* to a scalar duration."""
    if observation_window is None:
        warnings.warn(
            "observation_window not specified. Inferring from max spike time. "
            "Please specify observation_window explicitly for accurate normalization.",
            UserWarning,
            stacklevel=3,
        )
        max_times = [s[-1] for s in spike_trains if s.size]
        if not max_times:
            return 0.0
        return float(max(max_times))
    if isinstance(observation_window, (tuple, list)):
        t_start, t_end = observation_window
        return float(t_end - t_start)
    return float(observation_window)


def _apply_normalization(
    C: np.ndarray,
    lags: np.ndarray,
    normalize: str,
    T: float,
    spike_trains: list[np.ndarray],
    bin_size: float,
) -> None:
    """Apply normalization to CCG matrix *C* in place.

    Modes:

    - ``"counts"`` divides each row by the number of reference spikes,
      giving mean count per reference spike per bin (dimensionless).
    - ``"rate"`` divides by ``bin_size·T``, giving counts per bin per
      second.  For an auto-CCG the long-lag baseline is **rate squared**
      (``(N/T)²``), not rate — useful for CCG normalisation against
      the product of two rates, but misleading when interpreted as a
      single-spike rate.  Prefer ``"conditional"`` for ACG plots.
    - ``"conditional"`` gives the conditional intensity
      ``λ(t | spike at 0)`` in Hz.  Counts are divided by ``N_ref``
      (per-row, like ``"counts"``) and then by ``bin_size``, so the
      long-lag baseline of an auto-CCG is the unit's mean firing rate.
      This is the textbook ACG/CCG normalisation for shape comparison.
    - ``"unbiased"`` corrects for the shrinking window of available
      lag pairs near the edges of the recording.
    - ``"none"`` leaves raw counts.
    """
    if normalize == "counts":
        nsp = np.array([len(s) for s in spike_trains], dtype=np.float32)
        nsp[nsp == 0] = 1.0
        C /= nsp[:, None, None]
    elif normalize == "rate":
        C /= bin_size * max(T, 1e-12)
    elif normalize == "conditional":
        nsp = np.array([len(s) for s in spike_trains], dtype=np.float32)
        nsp[nsp == 0] = 1.0
        C /= (nsp * bin_size)[:, None, None]
    elif normalize == "unbiased":
        avail = np.maximum(T - np.abs(lags), bin_size).astype(np.float32)
        C /= avail[None, None, :]
    elif normalize != "none":
        raise ValueError(f"Unknown normalize mode: {normalize!r}")


# ---------------------------------------------------------------------------
# Per-pair <-> (N, N) conversion
# ---------------------------------------------------------------------------


def pair_vec_to_NN(
    pair_values: np.ndarray,
    pairs: np.ndarray,
    n_units: int,
    *,
    fill: float = 0.0,
    mirror: bool,
    dtype: object = None,
) -> np.ndarray:
    """Scatter a per-pair vector into an ``(n_units, n_units)`` array.

    The sparse per-pair representation used by the trial-paired /
    surrogate CCG path (and by ``ccg_between_sets_sparse``) is projected
    back into the dense ``(N, N)`` matrix that ``arr[i, j]`` lookups
    expect.  Cells outside ``pairs`` (and their mirror, when
    ``mirror=True``) hold ``fill``.

    Parameters
    ----------
    pair_values
        Per-pair values in the same row order as ``pairs``.
    pairs
        ``(n_pairs, 2)`` array of ``(i, j)`` unit-index pairs.
    n_units
        ``N`` — size of the square output.
    fill
        Value for cells not covered by ``pairs``.
    mirror
        Whether to also write each value at the transposed cell ``(j, i)``.
        Required rather than defaulted: mirroring is valid only for a
        statistic that is symmetric under swapping the pair.  A max over
        *all* lags is (``max(reverse(x)) == max(x)``); a directional one --
        peak lag, or a max over positive lags only, the natural ``i -> j``
        test -- is not, and mirroring it is silently wrong in the same
        ``sigma^-1`` way as the dense CCG mirror.  See
        :meth:`CCGCounts.mirror_is_defined` for the histogram-valued case.
    dtype
        Output dtype; defaults to ``pair_values.dtype``.

    Returns
    -------
    np.ndarray
        ``(n_units, n_units)`` array.
    """
    if dtype is None:
        dtype = pair_values.dtype
    out = np.full((n_units, n_units), fill, dtype=dtype)
    out[pairs[:, 0], pairs[:, 1]] = pair_values
    if mirror:
        out[pairs[:, 1], pairs[:, 0]] = pair_values
    return out


def NN_to_pair_vec(arr_NN: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Gather a per-pair vector from an ``(n_units, n_units)`` array.

    Inverse of :func:`pair_vec_to_NN` (modulo ``fill`` cells outside the
    pair set), for projecting an existing ``(N, N)`` array onto the
    per-pair representation.

    Parameters
    ----------
    arr_NN
        ``(n_units, n_units)`` array.
    pairs
        ``(n_pairs, 2)`` array of ``(i, j)`` unit-index pairs.

    Returns
    -------
    np.ndarray
        Per-pair vector ``arr_NN[i, j]`` in ``pairs`` row order.
    """
    return arr_NN[pairs[:, 0], pairs[:, 1]]


# ---------------------------------------------------------------------------
# Surrogate trial pairings
# ---------------------------------------------------------------------------


def derangements(
    n: int, count: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield *count* derangements (permutations with no fixed points).

    A derangement of ``range(n)`` maps no index to itself, so using one
    to re-pair trials destroys cross-unit correlations while preserving
    each unit's within-trial structure — the basis for trial-identity
    surrogate testing (feed the yielded permutations to
    :func:`ccg_trial_surrogates`).

    Parameters
    ----------
    n
        Number of trials (must be >= 2; no derangement exists for n < 2).
    count
        Number of derangements to yield.
    rng
        NumPy random generator.

    Yields
    ------
    np.ndarray
        A length-``n`` permutation with no fixed points.
    """
    if n < 2:
        raise ValueError("derangements require n >= 2.")
    identity = np.arange(n)
    for _ in range(count):
        while True:
            p = rng.permutation(n)
            if np.all(p != identity):
                yield p
                break
