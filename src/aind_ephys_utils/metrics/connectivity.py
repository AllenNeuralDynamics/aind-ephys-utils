"""CCG-based connectivity detection: pre-filtering and surrogate testing.

A higher-level workflow built on the cross-correlogram primitives in
:mod:`aind_ephys_utils.metrics.ccg`:

- :func:`build_shape_prefilter` — Bourgon+ 2010 independent (latency +
  FWHM) pre-filter that shrinks the multiple-testing burden before
  surrogates are run.
- :func:`run_surrogates` — derangement surrogate driver returning the
  per-pair surrogate peak distribution.
- :func:`run_two_stage_mc` — screen-then-refine Monte Carlo significance
  testing over the pre-filtered pair set.

The surrogate functions drive the numba CCG kernel via
``ccg_trial_surrogates``, so they require the ``numba`` optional extra at
run time (``pip install aind-ephys-utils[numba]``).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .ccg import (
    TrialSegments,
    ccg_trial_surrogates,
    derangements,
    monte_carlo_pvalue,
    pair_vec_to_NN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Independent shape pre-filter (Bourgon+ 2010)
# ---------------------------------------------------------------------------


def build_shape_prefilter(
    *,
    n_units: int,
    cross_pairs: np.ndarray,
    peak_lag: np.ndarray,
    peak_fwhm: np.ndarray,
    fwhm_range_s: tuple[float, float] | None,
    latency_range_s: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the latency + FWHM independent pre-filter for pairs.

    Bourgon+ 2010: pre-filtering on covariates that are approximately
    null-independent of the test statistic (peak bin location for
    latency; noise-driven FWHM for FWHM) is statistically sound and
    shrinks the multiple-testing burden.  Prominence is *not* part of
    this — it correlates with peak height under the null and would
    inflate FDR if used as a pre-filter; apply it post-hoc instead.

    Parameters
    ----------
    n_units
        ``N`` — used to allocate the ``(N, N)`` mask arrays.
    cross_pairs
        Iterable of ``(i, j)`` pair tuples with ``i < j``.
    peak_lag
        ``(N, N)`` peak-lag-in-seconds array.
    peak_fwhm
        ``(N, N)`` peak-FWHM-in-seconds array, ``inf`` for un-measured.
    fwhm_range_s
        ``(low, high)`` FWHM bounds in seconds, or ``None`` to disable.
    latency_range_s
        ``(low, high)`` ``|lag|`` bounds in seconds, or ``None`` to
        disable.

    Returns
    -------
    upper_mask, lat_ok, fwhm_ok, prefilter_mask : (N, N) bool arrays
        ``upper_mask`` marks the upper triangle of supplied pairs;
        ``lat_ok`` / ``fwhm_ok`` are per-pair shape masks (all-``True``
        when the corresponding range is ``None``); ``prefilter_mask`` is
        their conjunction.
    prefilter_pairs : (K, 2) int64 array
        The subset of ``cross_pairs`` that survived the pre-filter.
    """
    upper_mask = np.zeros((n_units, n_units), dtype=bool)
    for i, j in cross_pairs:
        upper_mask[i, j] = True

    if fwhm_range_s is not None:
        fw_lo, fw_hi = fwhm_range_s
        fwhm_ok = (peak_fwhm >= fw_lo) & (peak_fwhm <= fw_hi)
    else:
        fwhm_ok = np.ones((n_units, n_units), dtype=bool)

    if latency_range_s is not None:
        lat_lo, lat_hi = latency_range_s
        lat_ok = (np.abs(peak_lag) >= lat_lo) & (np.abs(peak_lag) <= lat_hi)
    else:
        lat_ok = np.ones((n_units, n_units), dtype=bool)

    prefilter_mask = upper_mask & lat_ok & fwhm_ok
    prefilter_pairs = np.array(
        [(i, j) for i, j in cross_pairs if prefilter_mask[i, j]],
        dtype=np.int64,
    )
    return upper_mask, lat_ok, fwhm_ok, prefilter_mask, prefilter_pairs


# ---------------------------------------------------------------------------
# Surrogate driver
# ---------------------------------------------------------------------------


def run_surrogates(
    trial_segs: TrialSegments,
    n_trials: int,
    n_units: int,
    n_surr: int,
    pairs: np.ndarray,
    rng: np.random.Generator,
    *,
    bin_size: float,
    max_lag: float,
    normalize: str,
    label: str = "",
) -> np.ndarray:
    """Run derangement surrogates, return a ``(n_surr, n_pairs)`` array.

    Each surrogate re-pairs trials with a derangement (no fixed points),
    recomputes the CCG for every pair in ``pairs``, and reduces it to the
    per-pair peak (max over lag bins).  Output rows are in ``pairs`` order
    and are ``float32`` (surrogate corrcoef peaks live in roughly
    ``[-1, 1]``, so float32 precision is ample and halves memory).

    Parameters
    ----------
    trial_segs
        Trial-clipped spike segments (see ``clip_spikes_to_trials``).
    n_trials
        Number of trials (the derangement length).
    n_units
        Number of units; kept for signature symmetry — the output shape
        is per-pair, so callers project to ``(N, N)`` themselves.
    n_surr
        Number of surrogate draws.
    pairs
        ``(n_pairs, 2)`` unit-index pairs to evaluate.
    rng
        NumPy random generator.
    bin_size, max_lag, normalize
        CCG kernel parameters.
    label
        Prefix for progress log messages.

    Returns
    -------
    np.ndarray
        ``(n_surr, n_pairs)`` float32 surrogate peak values.
    """
    n_pairs = pairs.shape[0]
    peaks = np.zeros((n_surr, n_pairs), dtype=np.float32)
    t0 = time.perf_counter()
    for k, peak_per_pair in enumerate(
        ccg_trial_surrogates(
            trial_segs,
            derangements(n_trials, n_surr, rng),
            bin_size=bin_size,
            max_lag=max_lag,
            normalize=normalize,
            pairs=pairs,
            reduce=lambda _l, C: np.max(C, axis=-1),
        )
    ):
        peaks[k] = peak_per_pair
        if (k + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (k + 1) / elapsed
            eta = (n_surr - k - 1) / rate
            logger.info(
                "  %s%d/%d (%.1f/s, ETA %.0fs)",
                label,
                k + 1,
                n_surr,
                rate,
                eta,
            )
    elapsed = time.perf_counter() - t0
    logger.info(
        "%s%d surrogates in %.1fs (%.2fs each)",
        label,
        n_surr,
        elapsed,
        elapsed / max(n_surr, 1),
    )
    return peaks


# ---------------------------------------------------------------------------
# Two-stage Monte Carlo significance testing
# ---------------------------------------------------------------------------


def _subset_trial_segments(
    ts: TrialSegments,
    unit_indices: np.ndarray,
) -> TrialSegments:
    """Return a TrialSegments containing only the given units, in order.

    Shrinks the per-surrogate scatter/normalize footprint when refining
    significance on a small candidate set: most selected units appear in
    no candidate pair, so their spike data is dead weight for the kernel.
    """
    n_sub = len(unit_indices)
    n_trials = ts.counts.shape[1]
    sub_segments = [ts.segments[int(i)] for i in unit_indices]
    sub_counts = ts.counts[unit_indices]
    total = int(sub_counts.sum())
    sub_all = np.empty(total, dtype=np.float64)
    sub_offsets = np.zeros((n_sub, n_trials), dtype=np.int64)
    pos = 0
    for new_i, old_i in enumerate(unit_indices):
        for k in range(n_trials):
            n = int(ts.counts[old_i, k])
            sub_offsets[new_i, k] = pos
            if n > 0:
                src = int(ts.offsets[old_i, k])
                sub_all[pos : pos + n] = ts.all_spikes[src : src + n]
                pos += n
    return TrialSegments(
        segments=sub_segments,
        durations=ts.durations,
        pre=ts.pre,
        post=ts.post,
        all_spikes=sub_all,
        offsets=sub_offsets,
        counts=sub_counts,
    )


def run_two_stage_mc(  # noqa: C901
    *,
    trial_segs: TrialSegments,
    n_trials: int,
    n_units: int,
    raw_peak_val: np.ndarray,
    prefilter_pairs: np.ndarray,
    prefilter_mask: np.ndarray,
    n_screen: int,
    n_total: int,
    bin_size: float,
    max_lag: float,
    normalize: str,
    rng: np.random.Generator,
    fdr_alpha: float,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Two-stage Monte Carlo significance testing on pre-filtered pairs.

    Stage 1 runs ``n_screen`` derangement surrogates over every
    pre-filtered pair, building coarse p-values.  Pairs at the
    ``1/(n_screen + 1)`` floor become candidates for stage 2, which runs
    ``max(n_total - n_screen, 0)`` additional surrogates so candidate
    p-values are computed against ``n_total`` exchangeable surrogates in
    total (stage-1 and stage-2 draws come from the same null and are
    concatenated; nothing is wasted).  The achievable minimum p-value for
    a candidate is ``1/(n_total + 1)``.

    Pairs outside the pre-filter never have MC computed and get ``p = 1``,
    so a Benjamini-Hochberg pass downstream cannot declare them
    significant.

    Parameters
    ----------
    trial_segs
        Trial-clipped spike segments.
    n_trials, n_units
        Trial and unit counts.
    raw_peak_val
        ``(N, N)`` observed peak heights (the test statistic).
    prefilter_pairs
        Output of :func:`build_shape_prefilter` — the pair list to test.
    prefilter_mask
        ``(N, N)`` bool mask matching ``prefilter_pairs`` (accepted for
        API symmetry; the function operates on ``prefilter_pairs``).
    n_screen
        Stage-1 surrogate count.
    n_total
        Target total surrogate count for refined candidates (stage 1 +
        stage 2).  Stage 2 runs ``max(n_total - n_screen, 0)`` extra
        surrogates; if ``n_total <= n_screen`` the refinement is a no-op.
    bin_size, max_lag, normalize, rng
        CCG kernel parameters.
    fdr_alpha
        Used for the surrogate threshold percentile returned for plotting.

    Returns
    -------
    p_values : (N, N) array
        Final p-values; held at 1.0 outside the pre-filter.
    surr_threshold : (N, N) array
        Per-pair ``(1 - fdr_alpha)`` percentile of surrogate peaks.
    n_candidates : int
        Number of pairs that hit the stage-1 floor (= stage-2 size).
    n_surr_total : int
        ``n_screen`` if there were no candidates / no extra surrogates,
        else ``n_total``.
    candidate_pairs : (n_candidates, 2) int64 array
        The pairs sent through stage 2 (empty when none).
    """
    logger.info(
        "Stage 1: %d surrogates, %d shape-passing pairs",
        n_screen,
        len(prefilter_pairs),
    )
    surr_screen = run_surrogates(
        trial_segs,
        n_trials,
        n_units,
        n_screen,
        prefilter_pairs,
        rng,
        bin_size=bin_size,
        max_lag=max_lag,
        normalize=normalize,
        label="[screen] ",
    )

    raw_peak_pf = raw_peak_val[prefilter_pairs[:, 0], prefilter_pairs[:, 1]]
    p_screen_pf = monte_carlo_pvalue(raw_peak_pf, surr_screen, tail="upper")
    p_floor = 1.0 / (n_screen + 1)

    cand_mask_pf = p_screen_pf <= p_floor
    cand_idx_in_pf = np.where(cand_mask_pf)[0]
    candidate_pairs = prefilter_pairs[cand_idx_in_pf].astype(np.int64)
    if candidate_pairs.size == 0:
        candidate_pairs = candidate_pairs.reshape(0, 2)
    n_candidates = len(candidate_pairs)
    logger.info(
        "Stage 1: %d / %d pre-filtered pairs at p-value floor (%.4f)",
        n_candidates,
        len(prefilter_pairs),
        p_floor,
    )

    n_additional = max(n_total - n_screen, 0)
    if n_candidates > 0 and n_additional > 0:
        logger.info(
            "Stage 2: %d additional surrogates (target total %d), "
            "%d candidate pairs",
            n_additional,
            n_total,
            n_candidates,
        )

        # Subset trial_segs to only the units in a candidate pair: most
        # selected units appear in no candidate, so their spike data is
        # dead weight for the refine kernel.
        unique_units = np.unique(candidate_pairs)
        n_sub = len(unique_units)
        old_to_new = np.full(n_units, -1, dtype=np.int64)
        old_to_new[unique_units] = np.arange(n_sub)
        sub_segs = _subset_trial_segments(trial_segs, unique_units)
        sub_pairs_remapped = np.column_stack(
            [
                old_to_new[candidate_pairs[:, 0]],
                old_to_new[candidate_pairs[:, 1]],
            ]
        ).astype(np.int64)
        logger.info(
            "  subsetting trial_segs: %d -> %d units (%.0f%%)",
            n_units,
            n_sub,
            100 * n_sub / n_units,
        )

        surr_refine = run_surrogates(
            sub_segs,
            n_trials,
            n_sub,
            n_additional,
            sub_pairs_remapped,
            rng,
            bin_size=bin_size,
            max_lag=max_lag,
            normalize=normalize,
            label="[refine] ",
        )

        sub_surr_screen = surr_screen[:, cand_idx_in_pf]
        sub_surr_combined = np.concatenate(
            [sub_surr_screen, surr_refine], axis=0
        )
        n_surr_total = n_screen + n_additional  # == n_total

        raw_peak_cand = raw_peak_val[
            candidate_pairs[:, 0], candidate_pairs[:, 1]
        ]
        p_cand = monte_carlo_pvalue(
            raw_peak_cand, sub_surr_combined, tail="upper"
        )
        cand_threshold = np.percentile(
            sub_surr_combined, 100 * (1 - fdr_alpha), axis=0
        )
        pf_threshold = np.percentile(
            surr_screen, 100 * (1 - fdr_alpha), axis=0
        )

        p_values_pf = p_screen_pf.copy()
        p_values_pf[cand_idx_in_pf] = p_cand
        threshold_pf = pf_threshold.copy()
        threshold_pf[cand_idx_in_pf] = cand_threshold
    else:
        p_values_pf = p_screen_pf
        n_surr_total = n_screen
        threshold_pf = np.percentile(
            surr_screen, 100 * (1 - fdr_alpha), axis=0
        )

    # Scatter per-prefilter-pair vectors into (N, N) for the public API.
    p_values = pair_vec_to_NN(
        p_values_pf, prefilter_pairs, n_units, fill=1.0, dtype=np.float64
    )
    surr_threshold = pair_vec_to_NN(
        threshold_pf, prefilter_pairs, n_units, fill=np.nan, dtype=np.float64
    )

    return (
        p_values,
        surr_threshold,
        n_candidates,
        n_surr_total,
        candidate_pairs,
    )
