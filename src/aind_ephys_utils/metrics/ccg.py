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
in ``SpikeAnalysis.jl`` and ``np.histogram``.  The ``"corrcoef"``
normalization mode subtracts edge-corrected expected counts and divides by
``sqrt(auto_u * auto_v)`` to produce a proper correlation coefficient, also
matching the Julia implementation.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_prominences, peak_widths

from aind_ephys_utils._numba import njit, prange

# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def _ccg_two_pointer(t1, t2, max_lag, bin_size, nbins):
    """Numba-accelerated two-pointer cross-correlogram kernel.

    Uses left-closed bins: bin *k* covers ``[k*bin_size - bin_size/2,
    k*bin_size + bin_size/2)`` relative to zero lag.
    """
    h = np.zeros(nbins, dtype=np.int64)
    _ccg_two_pointer_accum(t1, t2, max_lag, bin_size, nbins, h)
    return h


@njit(cache=True, fastmath=True)
def _ccg_two_pointer_accum(t1, t2, max_lag, bin_size, nbins, out):
    """Accumulate cross-correlogram counts into *out* (no allocation)."""
    half = nbins // 2
    inv = 1.0 / bin_size
    window_margin = max_lag + 0.5 * bin_size

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
# Edge-corrected normalization helpers (ported from SpikeAnalysis.jl)
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


def _expected_counts_shape(
    nbins: int, bin_size: float, dur: float
) -> np.ndarray:
    """Precompute the per-bin edge-correction shape (independent of spike counts).

    Returns a vector of length *nbins* such that the expected count for bin *k*
    given spike counts ``nu, nv`` is ``shape[k] * nu * nv``.
    This is factored out of ``_subtract_expected_counts_symm`` so it can be
    computed once and reused across all pairs.
    """
    n_sidebins = (nbins - 1) // 2
    halfbin = bin_size / 2
    dur2 = dur**2

    shape = np.empty(nbins, dtype=np.float64)
    # Center bin uses halfbin as both binstop and bin_size
    shape[n_sidebins] = 2 * (halfbin * (dur - halfbin + halfbin / 2)) / dur2
    for i in range(1, n_sidebins + 1):
        binstop = halfbin + i * bin_size
        e = (bin_size * (dur - binstop + bin_size / 2)) / dur2
        shape[n_sidebins + i] = e
        shape[n_sidebins - i] = e
    return shape


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

    for s in spike_times_set1 + spike_times_set2:
        if s.size and np.any(np.diff(s) < 0):
            raise ValueError("Spike times must be sorted.")

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
    def _compute_pairs(pair_idx, S1, S2, half, bin_size, max_lag, out_buf):
        """Numba kernel: accumulate CCGs for a chunk of set1×set2 pairs."""
        nbins = 2 * half + 1
        for k in prange(pair_idx.shape[0]):
            i = pair_idx[k, 0]
            j = pair_idx[k, 1]
            if S1[i].size == 0 or S2[j].size == 0:
                continue
            out_buf[k, :] = _ccg_two_pointer(
                S1[i], S2[j], max_lag, bin_size, nbins
            )

    _compute_pairs(pair_list, S1, S2, half, bin_size, max_lag, out_buf)

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
            if i == j and exclude_zero_lag_autocorr:
                h[half] = 0.0

            if use_corrcoef and auto_terms is not None:
                denom = np.sqrt(abs(auto_terms[i] * auto_terms[j]))
                if denom > 0:
                    h = h / denom

            C[i, j, :] = h
            if i != j:
                C[j, i, :] = h[::-1]
            else:
                C[i, i, :] = 0.5 * (h + h[::-1])
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
    for s in spike_times_by_unit:
        if s.size and np.any(np.diff(s) < 0):
            raise ValueError("Spike times must be sorted.")

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
    def _compute_pairs(pair_idx, S, half, bin_size, max_lag, out_buf):
        """Numba kernel: accumulate CCGs for a chunk of all-pairs."""
        nbins = 2 * half + 1
        for k in prange(pair_idx.shape[0]):
            i = pair_idx[k, 0]
            j = pair_idx[k, 1]
            if S[i].size == 0 or S[j].size == 0:
                continue
            out_buf[k, :] = _ccg_two_pointer(
                S[i], S[j], max_lag, bin_size, nbins
            )

    _compute_pairs(pair_list_arr, S, half, bin_size, max_lag, out_buf)

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


def rescale_ccgs_zero_mean(
    C: np.ndarray, axis: int = -1, eps: float = 1e-12
) -> np.ndarray:
    """Rescale correlograms: subtract mean, then min-max to [0, 1]."""
    meanv = C.mean(axis=axis, keepdims=True)
    C_centered = C - meanv
    minv = C_centered.min(axis=axis, keepdims=True)
    maxv = C_centered.max(axis=axis, keepdims=True)
    rng = np.maximum(maxv - minv, eps)
    return (C_centered - minv) / rng


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
    max_lag,
    bin_size,
    nbins,
    out_hist,
    out_nspikes_i,
    out_nspikes_j,
    out_dur,
    skip_clipping,
):
    """Numba kernel: all pairs × all trials, prange over pairs.

    When *skip_clipping* is True, segments are used as-is (no searchsorted).
    Use this after ``clip_to_window`` when all trials have equal duration.
    """
    n_trials = overlap_left.shape[0]
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
            # Duration accumulates for every valid overlap, even if one
            # unit is silent — silence is a valid rate=0 observation.
            out_dur[p] += od
            if ni == 0 or nj == 0:
                continue
            # Spike counts only from trials that contribute to the
            # histogram, consistent with the expected-count formula.
            _ccg_two_pointer_accum(
                si, sj, max_lag, bin_size, nbins, out_hist[p, :]
            )
            out_nspikes_i[p] += ni
            out_nspikes_j[p] += nj


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
    N = len(ts.segments)
    n_trials = len(ts.durations)
    new_dur = right - left

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
        "out_nspikes_i",
        "out_nspikes_j",
        "out_dur",
        "auto_per_trial",
    )

    def __init__(
        self, n_pairs: int, nbins: int, n_units: int, n_trials: int
    ) -> None:
        """Allocate reusable output buffers for trial-paired CCGs."""
        self.pair_arr: np.ndarray | None = None
        self.out_hist = np.zeros((n_pairs, nbins), dtype=np.int64)
        self.out_nspikes_i = np.zeros(n_pairs, dtype=np.int64)
        self.out_nspikes_j = np.zeros(n_pairs, dtype=np.int64)
        self.out_dur = np.zeros(n_pairs, dtype=np.float64)
        self.auto_per_trial = np.zeros((n_units, n_trials), dtype=np.float64)

    def zero(self) -> None:
        """Reset all buffers to zero (avoids reallocation)."""
        self.out_hist[:] = 0
        self.out_nspikes_i[:] = 0
        self.out_nspikes_j[:] = 0
        self.out_dur[:] = 0
        self.auto_per_trial[:] = 0

    @staticmethod
    def for_segments(
        ts: TrialSegments,
        bin_size: float,
        max_lag: float,
        include_autocorr: bool = True,
        pairs: np.ndarray | None = None,
    ) -> _CCGBuffers:
        """Create buffers sized for the given segments and parameters.

        Parameters
        ----------
        pairs
            Optional ``(n_pairs, 2)`` int64 array of ``(i, j)`` unit index
            pairs.  If ``None``, all upper-triangle pairs are used.
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
        bufs = _CCGBuffers(pair_arr.shape[0], B, N, n_trials)
        bufs.pair_arr = pair_arr
        return bufs


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
    ts = trial_segments
    N = len(ts.segments)
    half = int(round(max_lag / bin_size))
    B = 2 * half + 1
    lags = (np.arange(-half, half + 1) * bin_size).astype(np.float64)

    pairing = np.asarray(pairing, dtype=np.int64)

    # Overlap window per trial
    overlap_left = np.maximum(-ts.pre, -ts.pre[pairing])
    overlap_right = np.minimum(ts.post, ts.post[pairing])

    # Use pre-allocated buffers or create new ones
    if buffers is not None:
        bufs = buffers
        bufs.zero()
    else:
        bufs = _CCGBuffers.for_segments(
            ts,
            bin_size,
            max_lag,
            include_autocorr,
            pairs=pairs,
        )

    pair_arr = bufs.pair_arr
    assert pair_arr is not None
    n_pairs = pair_arr.shape[0]

    # Skip clipping when all trials have uniform duration (e.g., after clip_to_window)
    skip_clipping = bool(np.all(ts.durations == ts.durations[0]))

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
            max_lag,
            bin_size,
            B,
            bufs.out_hist,
            bufs.out_nspikes_i,
            bufs.out_nspikes_j,
            bufs.out_dur,
            skip_clipping,
        )

    # Corrcoef normalization
    use_corrcoef = normalize == "corrcoef"
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

    # Precompute binstop array for expected-count calculation (constant across pairs)
    if use_corrcoef:
        n_sidebins = (B - 1) // 2
        halfbin = bin_size / 2
        binstops = np.empty(B, dtype=np.float64)
        binstops[n_sidebins] = halfbin  # center bin
        for ib in range(1, n_sidebins + 1):
            binstops[n_sidebins + ib] = halfbin + ib * bin_size
            binstops[n_sidebins - ib] = halfbin + ib * bin_size
        # Center bin uses halfbin as bin_size; side bins use bin_size
        bin_widths = np.full(B, bin_size, dtype=np.float64)
        bin_widths[n_sidebins] = halfbin
        # Scale factor: center bin has factor 2 (symmetric)
        center_scale = np.ones(B, dtype=np.float64)
        center_scale[n_sidebins] = 2.0

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
            d2 = d_uniform * d_uniform
            ec_shape_single = (
                center_scale
                * bin_widths
                * (d_uniform - binstops + bin_widths / 2)
                / d2
            )
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
            # Non-uniform: precompute per-trial ec shapes, weighted by duration
            # ec_weighted[k] = ec_shape(d_k) as a (n_trials, B) array
            n_tr = trial_counts.shape[1]
            valid = overlap_durs > 0
            ec_weighted = np.zeros((n_tr, B), dtype=np.float64)
            for k in range(n_tr):
                if not valid[k]:
                    continue
                d_k = overlap_durs[k]
                d2 = d_k * d_k
                ec_weighted[k, :] = (
                    center_scale
                    * bin_widths
                    * (d_k - binstops + bin_widths / 2)
                    / d2
                )

    # When ``pairs`` is provided, allocate only the per-pair output.
    # Skipping the (N, N, B) materialisation is a 2 GB saving at
    # N=499, B=1001 — and lets the caller stay per-pair end-to-end
    # without having to project after the fact.
    if pairs is not None:
        C = np.zeros((n_pairs, B), dtype=np.float64)
        for p in range(n_pairs):
            i, j = pair_arr[p, 0], pair_arr[p, 1]
            total_dur = bufs.out_dur[p]
            if total_dur <= 0:
                continue
            h = bufs.out_hist[p, :].astype(np.float64)
            if i == j and exclude_zero_lag_autocorr:
                h[half] = 0.0
            if use_corrcoef:
                if skip_clipping:
                    h -= ec_shape_single * cross_per_pair[p]
                else:
                    cross_per_trial = (
                        trial_counts[i, :] * trial_counts[j, pairing]
                    )
                    h -= ec_weighted.T @ cross_per_trial
                denom = np.sqrt(abs(auto_sum_direct[i] * auto_sum_paired[j]))
                if denom > 0:
                    h /= denom
            # Autocorr (i == j) gets the symmetrised treatment matching
            # the dense path's diagonal cell.  Cross-pairs store h
            # directly; callers wanting the (j, i) CCG can reverse h.
            if i == j:
                h = 0.5 * (h + h[::-1])
            C[p, :] = h
    else:
        C = np.zeros((N, N, B), dtype=np.float64)
        for p in range(n_pairs):
            i, j = pair_arr[p, 0], pair_arr[p, 1]
            total_dur = bufs.out_dur[p]
            if total_dur <= 0:
                continue
            h = bufs.out_hist[p, :].astype(np.float64)
            if i == j and exclude_zero_lag_autocorr:
                h[half] = 0.0
            if use_corrcoef:
                if skip_clipping:
                    h -= ec_shape_single * cross_per_pair[p]
                else:
                    cross_per_trial = (
                        trial_counts[i, :] * trial_counts[j, pairing]
                    )
                    h -= ec_weighted.T @ cross_per_trial
                denom = np.sqrt(abs(auto_sum_direct[i] * auto_sum_paired[j]))
                if denom > 0:
                    h /= denom
            C[i, j, :] = h
            if i != j:
                C[j, i, :] = h[::-1]
            else:
                C[i, i, :] = 0.5 * (h + h[::-1])

    return lags, C


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
    ts = trial_segments
    N = len(ts.segments)
    half = int(round(max_lag / bin_size))
    B = 2 * half + 1
    lags = (np.arange(-half, half + 1) * bin_size).astype(np.float64)
    skip_clipping = bool(np.all(ts.durations == ts.durations[0]))
    use_corrcoef = normalize == "corrcoef"

    n_trials_seg = len(ts.durations)
    bufs_a = _CCGBuffers.for_segments(
        ts, bin_size, max_lag, include_autocorr, pairs=pairs
    )
    bufs_b = _CCGBuffers.for_segments(
        ts, bin_size, max_lag, include_autocorr, pairs=pairs
    )

    # Precompute normalization constants for corrcoef
    if use_corrcoef:
        n_sidebins = (B - 1) // 2
        halfbin = bin_size / 2
        binstops = np.empty(B, dtype=np.float64)
        binstops[n_sidebins] = halfbin
        for ib in range(1, n_sidebins + 1):
            binstops[n_sidebins + ib] = halfbin + ib * bin_size
            binstops[n_sidebins - ib] = halfbin + ib * bin_size
        bin_widths = np.full(B, bin_size, dtype=np.float64)
        bin_widths[n_sidebins] = halfbin
        center_scale = np.ones(B, dtype=np.float64)
        center_scale[n_sidebins] = 2.0

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
        uniform_d = ts.post[0] + ts.pre[0]
        d2 = uniform_d * uniform_d
        uniform_ec: np.ndarray = (
            center_scale
            * bin_widths
            * (uniform_d - binstops + bin_widths / 2)
            / d2
        )

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
            max_lag,
            bin_size,
            B,
            bufs.out_hist,
            bufs.out_nspikes_i,
            bufs.out_nspikes_j,
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
                # Non-uniform: precompute per-trial ec shapes
                n_tr = trial_counts.shape[1]
                valid = overlap_durs > 0
                ec_weighted = np.zeros((n_tr, B), dtype=np.float64)
                for k in range(n_tr):
                    if not valid[k]:
                        continue
                    d_k = overlap_durs[k]
                    d2 = d_k * d_k
                    ec_weighted[k, :] = (
                        center_scale
                        * bin_widths
                        * (d_k - binstops + bin_widths / 2)
                        / d2
                    )

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
        if pairs is not None:
            C = np.zeros((n_pairs, B), dtype=np.float64)
            for p in range(n_pairs):
                i, j = pair_arr[p, 0], pair_arr[p, 1]
                total_dur = bufs.out_dur[p]
                if total_dur <= 0:
                    continue
                h = bufs.out_hist[p, :].astype(np.float64)
                if i == j and exclude_zero_lag_autocorr:
                    h[half] = 0.0
                if use_corrcoef:
                    if uniform_fast:
                        h -= uniform_ec * cross_per_pair[p]
                    else:
                        cross_per_trial = (
                            trial_counts[i, :] * paired_counts[j, :]
                        )
                        h -= ec_weighted.T @ cross_per_trial
                    denom = np.sqrt(abs(auto_sum_i[i] * auto_sum_j[j]))
                    if denom > 0:
                        h /= denom
                # Autocorrs (i==j) get the symmetrised (h + h[::-1]) / 2
                # treatment to match the dense path's behaviour at that
                # diagonal cell.  Cross-pairs just store h directly;
                # callers wanting the lower-triangle mirror can reverse
                # h themselves at index p (since h[::-1] is the (j, i)
                # CCG by definition of the cross-correlation).
                if i == j:
                    h = 0.5 * (h + h[::-1])
                C[p, :] = h
        else:
            C = np.zeros((N, N, B), dtype=np.float64)
            for p in range(n_pairs):
                i, j = pair_arr[p, 0], pair_arr[p, 1]
                total_dur = bufs.out_dur[p]
                if total_dur <= 0:
                    continue
                h = bufs.out_hist[p, :].astype(np.float64)
                if i == j and exclude_zero_lag_autocorr:
                    h[half] = 0.0
                if use_corrcoef:
                    if uniform_fast:
                        h -= uniform_ec * cross_per_pair[p]
                    else:
                        cross_per_trial = (
                            trial_counts[i, :] * paired_counts[j, :]
                        )
                        h -= ec_weighted.T @ cross_per_trial
                    denom = np.sqrt(abs(auto_sum_i[i] * auto_sum_j[j]))
                    if denom > 0:
                        h /= denom
                C[i, j, :] = h
                if i != j:
                    C[j, i, :] = h[::-1]
                else:
                    C[i, i, :] = 0.5 * (h + h[::-1])

        return reduce(lags, C) if reduce is not None else C

    # Pipeline — overlap bounds are constant when durations are uniform
    def _overlap_for(pairing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-pair overlap bounds for a trial pairing."""
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
