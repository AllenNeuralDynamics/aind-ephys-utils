"""Regression tests for CCG correctness defects, all of them now fixed.

Each test pins behaviour that was once wrong or unspecified, so a
reintroduction fails here rather than in someone's analysis.  Each says in
its docstring what the wrong answer looked like, which is the part worth
keeping: a test whose failure mode is unnamed tends to get "fixed" by
adjusting the assertion.

Several guard properties that no symmetric or single-pair fixture can see
-- the dense lower-triangle mirror under a non-involutive pairing, the lag
sign, and the direction-dependence of the integrated excess.  Those are the
ones to be most careful about weakening.

``ccg_implementation_plan.md`` records the decisions behind the cases that
needed one: what ``"corrcoef"`` promises, the dense mirror under a general
pairing, explicit ``(i, i)`` pairs, and the observation-window contract.
Each had to be settled before its assertion could be written at all.
"""

import unittest
import warnings

import numpy as np
import xarray as xr

from aind_ephys_utils.metrics.ccg import (
    CCGCounts,
    SurrogateTest,
    TrialSegments,
    TrialShuffle,
    _build_csr,
    _expected_counts_shape,
    _lag_exposure,
    _uniform_support,
    _window_weights,
    blank_zero_lag,
    ccg,
    ccg_allpairs_sparse,
    ccg_between_sets_sparse,
    ccg_trial_paired,
    clip_spikes_to_trials,
    clip_to_window,
    compute_ccg_counts,
    covariance_density,
    cross_intensity,
)
from aind_ephys_utils.metrics.ccg import derangements as _module_derangements
from aind_ephys_utils.metrics.ccg import (
    directional_excess,
    excess_density,
    legacy_auto_normalized,
    normalized_covariance,
    pair_correlation,
    pair_vec_to_NN,
    shift_predictor_baseline,
    stationary_baseline,
    surrogate_mean_baseline,
    surrogate_null,
    to_dense,
)

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = unittest.skipUnless(
    HAS_NUMBA, "requires the numba optional extra"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brute_ccg(t1, t2, bin_size, half):
    """Reference CCG: every pair, left-closed bins, no windowing tricks.

    Deliberately O(n*m).  Bin *k* covers ``[(k - 0.5) * bin_size,
    (k + 0.5) * bin_size)``, matching the kernel's ``floor(dt / w + 0.5)``.
    """
    h = np.zeros(2 * half + 1, dtype=np.int64)
    for a in t1:
        for b in t2:
            k = int(np.floor((b - a) / bin_size + 0.5))
            if -half <= k <= half:
                h[k + half] += 1
    return h


def _ragged_da(data, units=None, window=(0.0, 1.0)):
    """Build a ragged spikes DataArray."""
    if units is None:
        units = [f"u{i}" for i in range(data.shape[0])]
    da = xr.DataArray(
        data,
        dims=("unit", "trial"),
        coords={"unit": units, "trial": np.arange(data.shape[1])},
    )
    da.attrs["ephys.valid_intervals"] = [window]
    return da


def _segments(per_unit_trials, durations, pre, post):
    """Build a TrialSegments directly from alignment-relative spike lists."""
    n_units = len(per_unit_trials)
    n_trials = len(durations)
    segs = [
        [np.asarray(t, dtype=np.float64) for t in unit]
        for unit in per_unit_trials
    ]
    all_spikes, offsets, counts = _build_csr(segs, n_units, n_trials)
    return TrialSegments(
        segs,
        np.asarray(durations, dtype=np.float64),
        np.asarray(pre, dtype=np.float64),
        np.asarray(post, dtype=np.float64),
        all_spikes,
        offsets,
        counts,
    )


def _trial_segments_from_relative(rel_by_unit, trial_len, n_trials):
    """Lay out per-trial relative spikes on a session timeline and clip."""
    starts = np.arange(n_trials) * (trial_len * 2.0)
    abs_spikes = [
        np.sort(
            np.concatenate(
                [starts[k] + np.asarray(rel[k]) for k in range(n_trials)]
            )
        )
        for rel in rel_by_unit
    ]
    epochs = np.column_stack([starts, starts + trial_len])
    return clip_spikes_to_trials(abs_spikes, epochs, align_times=starts)


# ---------------------------------------------------------------------------
# Item 4: outer lag bins must be fully searched
# ---------------------------------------------------------------------------


@needs_numba
class KernelBruteForceTest(unittest.TestCase):
    """The sparse kernel must reproduce a brute-force histogram exactly."""

    def _assert_matches_brute(self, ratio, seed):
        """Compare the kernel against brute force for one ratio/seed."""
        bin_size = 0.001
        max_lag = ratio * bin_size
        half = int(round(max_lag / bin_size))
        rng = np.random.default_rng(seed)
        t1 = np.sort(rng.uniform(0, 0.05, 25))
        t2 = np.sort(rng.uniform(0, 0.05, 25))
        _, C = ccg_between_sets_sparse(
            [t1],
            [t2],
            bin_size=bin_size,
            max_lag=max_lag,
            observation_window=(0.0, 0.05),
        )
        np.testing.assert_array_equal(
            C[0, 0].astype(np.int64),
            _brute_ccg(t1, t2, bin_size, half),
            err_msg=f"ratio={ratio} seed={seed}",
        )

    def test_integer_lag_ratios(self):
        """Integer max_lag/bin_size ratios match brute force."""
        for ratio in (1.0, 2.0, 5.0):
            for seed in range(5):
                with self.subTest(ratio=ratio, seed=seed):
                    self._assert_matches_brute(ratio, seed)

    def test_non_integer_lag_ratios(self):
        """Non-integer ratios must still search every returned bin.

        ``half = round(max_lag / bin_size)`` can round up, making the outer
        bin extend past the ``max_lag + bin_size / 2`` search bound.
        """
        for ratio in (1.2, 1.49, 1.51, 1.6, 2.49, 2.51):
            for seed in range(5):
                with self.subTest(ratio=ratio, seed=seed):
                    self._assert_matches_brute(ratio, seed)

    def test_coincident_and_boundary_spikes(self):
        """Exact coincidences and half-bin boundaries land in the right bin."""
        bin_size = 0.001
        t1 = np.array([0.010, 0.020, 0.030])
        # +0 (coincident), +half bin (rounds up), -half bin (rounds up to -0)
        t2 = np.array([0.010, 0.0205, 0.0295])
        _, C = ccg_between_sets_sparse(
            [t1],
            [t2],
            bin_size=bin_size,
            max_lag=0.005,
            observation_window=(0.0, 0.05),
        )
        np.testing.assert_array_equal(
            C[0, 0].astype(np.int64), _brute_ccg(t1, t2, bin_size, 5)
        )

    def test_empty_and_single_spike_trains(self):
        """Empty and one-spike trains produce the brute-force answer."""
        cases = [
            (np.array([]), np.array([0.5])),
            (np.array([0.5]), np.array([])),
            (np.array([0.5]), np.array([0.5005])),
        ]
        for t1, t2 in cases:
            with self.subTest(n1=t1.size, n2=t2.size):
                _, C = ccg_between_sets_sparse(
                    [t1],
                    [t2],
                    bin_size=0.001,
                    max_lag=0.005,
                    observation_window=(0.0, 1.0),
                )
                np.testing.assert_array_equal(
                    C[0, 0].astype(np.int64), _brute_ccg(t1, t2, 0.001, 5)
                )


# ---------------------------------------------------------------------------
# Item 2: trial fast path must key on alignment, not duration
# ---------------------------------------------------------------------------


@needs_numba
class TrialOverlapClippingTest(unittest.TestCase):
    """Trials only contribute spikes inside their shared overlap window."""

    @staticmethod
    def _equal_duration_different_alignment():
        """Two 1-s trials whose alignment-relative supports barely overlap.

        pre/post are (0.2, 0.8) and (0.8, 0.2), so the shared support under
        a swapped pairing is only [-0.2, +0.2].  Both spikes sit outside it.
        """
        return _segments(
            per_unit_trials=[
                [[0.5], [-0.5]],
                [[0.5], [-0.5]],
            ],
            durations=[1.0, 1.0],
            pre=[0.2, 0.8],
            post=[0.8, 0.2],
        )

    def test_identity_pairing_keeps_in_window_spikes(self):
        """Each trial's own window contains its spike, so both count."""
        ts = self._equal_duration_different_alignment()
        _, C = ccg_trial_paired(
            ts,
            np.array([0, 1]),
            bin_size=0.01,
            max_lag=1.2,
            normalize="none",
        )
        self.assertEqual(C[0, 1].sum(), 2.0)

    def test_equal_duration_different_alignment_rejected(self):
        """Equal durations do not imply equal support, and must not pass.

        Both trials last 1 s, so a duration-based check waves this through;
        the shared support is only [-0.2, +0.2].
        """
        ts = self._equal_duration_different_alignment()
        with self.assertRaises(ValueError) as cm:
            ccg_trial_paired(
                ts,
                np.array([1, 0]),
                bin_size=0.01,
                max_lag=1.2,
                normalize="none",
            )
        self.assertIn("support", str(cm.exception))

    def test_unequal_support_pairing_rejected(self):
        """Pairing trials with different support is out of contract."""
        ts = _segments(
            per_unit_trials=[[[0.5], [-0.5]], [[0.5], [-0.5]]],
            durations=[1.0, 1.5],
            pre=[0.2, 0.8],
            post=[0.8, 0.7],
        )
        common = dict(bin_size=0.01, max_lag=1.2, normalize="none")
        # Identity is always legal: each trial pairs with itself.
        _, identity = ccg_trial_paired(ts, np.array([0, 1]), **common)
        self.assertEqual(identity[0, 1].sum(), 2.0)
        with self.assertRaises(ValueError):
            ccg_trial_paired(ts, np.array([1, 0]), **common)

    def test_matched_support_classes_allowed(self):
        """Trials may differ in support as long as partners match.

        Two support classes, swapped within each.  The pairing is not the
        identity and the trials are not uniform, so this exercises the
        searchsorted path -- where the clip must be a no-op.
        """
        ts = _segments(
            per_unit_trials=[
                [[0.3], [0.4], [-0.6], [-0.7]],
                [[0.3], [0.4], [-0.6], [-0.7]],
            ],
            durations=[1.0, 1.0, 1.5, 1.5],
            pre=[0.2, 0.2, 0.8, 0.8],
            post=[0.8, 0.8, 0.7, 0.7],
        )
        common = dict(bin_size=0.01, max_lag=1.2, normalize="none")
        _, swapped = ccg_trial_paired(ts, np.array([1, 0, 3, 2]), **common)
        # No spike is clipped away: every trial keeps its single spike, so
        # each of the four trial pairs contributes one coincidence.
        self.assertEqual(swapped[0, 1].sum(), 4.0)


# ---------------------------------------------------------------------------
# Item 1: only advertise normalization modes that exist
# ---------------------------------------------------------------------------


@needs_numba
class TrialPairedNormalizationTest(unittest.TestCase):
    """The trial-paired path supports 'none' and 'corrcoef' only."""

    def setUp(self):
        """Build four identical-duration trials for two units."""
        rng = np.random.default_rng(0)
        rel = [
            [np.sort(rng.uniform(0.0, 1.0, 20)) for _ in range(4)]
            for _ in range(2)
        ]
        self.ts = _trial_segments_from_relative(rel, 1.0, 4)
        self.pairing = np.arange(4)

    def test_none_returns_raw_counts(self):
        """'none' leaves integer coincidence counts untouched."""
        _, C = ccg_trial_paired(
            self.ts,
            self.pairing,
            bin_size=0.001,
            max_lag=0.01,
            normalize="none",
        )
        np.testing.assert_array_equal(C, np.round(C))

    def test_unsupported_modes_raise(self):
        """Modes the path does not implement must fail loudly, not silently.

        They currently return raw counts identical to ``normalize="none"``.
        """
        for mode in ("counts", "rate", "conditional", "unbiased"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    ccg_trial_paired(
                        self.ts,
                        self.pairing,
                        bin_size=0.001,
                        max_lag=0.01,
                        normalize=mode,
                    )

    def test_invalid_mode_raises(self):
        """An unrecognised mode string is rejected."""
        with self.assertRaises(ValueError):
            ccg_trial_paired(
                self.ts,
                self.pairing,
                bin_size=0.001,
                max_lag=0.01,
                normalize="not_a_mode",
            )


# ---------------------------------------------------------------------------
# Item 1c: the legacy auto term (deferred to Phase 5 of the plan)
# ---------------------------------------------------------------------------


@needs_numba
class LegacyAutoTermTest(unittest.TestCase):
    """``corrcoef`` gives C[0] == 1 for two units with equal trains.

    Self-consistency for the legacy statistic: the denominator's auto term
    is now the ACG's own two-sided centre bin, the same quantity the
    numerator's centre bin is.  Previously it took a one-sided width-``w``
    count against a doubled expectation, so the two disagreed by a
    rate-dependent factor and C[0] came out anywhere from 1.016 to 1.107.
    """

    @staticmethod
    def _identical_trains(kind, seed, n_trials=6):
        """Two distinct units carrying byte-identical spike trains."""
        rng = np.random.default_rng(seed)
        per_trial = []
        for _ in range(n_trials):
            if kind == "poisson":
                t = np.sort(rng.uniform(0.0, 1.0, 40))
            elif kind == "regular":
                t = np.linspace(0.01, 0.99, 40)
            elif kind == "bursty":
                seeds = np.sort(rng.uniform(0.0, 0.9, 8))
                t = np.sort(
                    np.concatenate(
                        [s + rng.exponential(0.004, 5) for s in seeds]
                    )
                )
                t = t[t < 1.0]
            else:  # refractory
                t = np.sort(rng.uniform(0.0, 1.0, 60))
                t = t[np.insert(np.diff(t) > 0.005, 0, True)]
            per_trial.append(t)
        return _trial_segments_from_relative(
            [per_trial, per_trial], 1.0, n_trials
        )

    def test_identical_trains_give_unit_zero_lag(self):
        """Self-consistency: a unit correlated with its own copy gives 1."""
        for kind in ("poisson", "regular", "bursty", "refractory"):
            for bin_size in (0.0005, 0.002):
                with self.subTest(kind=kind, bin_size=bin_size):
                    ts = self._identical_trains(kind, seed=5)
                    _, C = ccg_trial_paired(
                        ts,
                        np.arange(6),
                        bin_size=bin_size,
                        max_lag=0.02,
                        normalize="corrcoef",
                        pairs=np.array([[0, 1]], dtype=np.int64),
                    )
                    half = C.shape[-1] // 2
                    self.assertAlmostEqual(C[0, half], 1.0, places=6)


# ---------------------------------------------------------------------------
# Item 6: the dense mirror is only valid for an involutive pairing
# ---------------------------------------------------------------------------


@needs_numba
class DenseMirrorTest(unittest.TestCase):
    """``C[j, i] = reverse(C[i, j])`` computes sigma^-1, not sigma."""

    def setUp(self):
        """Three uniform trials with distinguishable per-trial spikes."""
        rng = np.random.default_rng(7)
        rel = [
            [np.sort(rng.uniform(0.05, 0.95, 12)) for _ in range(3)]
            for _ in range(2)
        ]
        self.ts = _trial_segments_from_relative(rel, 1.0, 3)
        self.common = dict(bin_size=0.01, max_lag=0.1, normalize="none")

    def test_mirror_filled_under_involution(self):
        """Identity and swap pairings keep the exact mirror."""
        for pairing in (np.array([0, 1, 2]), np.array([1, 0, 2])):
            with self.subTest(pairing=pairing.tolist()):
                _, C = ccg_trial_paired(self.ts, pairing, **self.common)
                self.assertTrue(np.all(np.isfinite(C)))
                np.testing.assert_array_equal(C[1, 0], C[0, 1][::-1])

    def test_mirror_nan_under_three_cycle(self):
        """A 3-cycle is not its own inverse, so the mirror is unavailable."""
        _, C = ccg_trial_paired(self.ts, np.array([1, 2, 0]), **self.common)
        self.assertTrue(np.all(np.isfinite(C[0, 1])))
        self.assertTrue(np.all(np.isnan(C[1, 0])))

    def test_naive_mirror_would_have_been_wrong(self):
        """Pin the reason: sigma and sigma^-1 give different histograms.

        Guards against someone restoring the mirror on the grounds that it
        "looks symmetric enough".
        """
        pairs = np.array([[1, 0]], dtype=np.int64)
        _, forward = ccg_trial_paired(
            self.ts, np.array([1, 2, 0]), pairs=pairs, **self.common
        )
        _, inverse = ccg_trial_paired(
            self.ts, np.array([2, 0, 1]), pairs=pairs, **self.common
        )
        self.assertFalse(np.array_equal(forward[0], inverse[0]))

    def test_diagonal_not_symmetrised(self):
        """The diagonal holds h^sigma, not 0.5 * (h^sigma + h^sigma^-1)."""
        pairing = np.array([1, 2, 0])
        _, dense = ccg_trial_paired(self.ts, pairing, **self.common)
        _, per_pair = ccg_trial_paired(
            self.ts,
            pairing,
            pairs=np.array([[0, 0]], dtype=np.int64),
            **self.common,
        )
        np.testing.assert_array_equal(dense[0, 0], per_pair[0])
        self.assertFalse(
            np.array_equal(dense[0, 0], dense[0, 0][::-1]),
            "a shifted autocorrelogram is not symmetric",
        )


# ---------------------------------------------------------------------------
# Item 9: zero-lag blanking means one thing on every path
# ---------------------------------------------------------------------------


@needs_numba
class ZeroLagBlankingTest(unittest.TestCase):
    """A blanked self-pair centre bin is 0.0, never -E/denom."""

    def setUp(self):
        """Four uniform trials, two units."""
        rng = np.random.default_rng(3)
        rel = [
            [np.sort(rng.uniform(0.05, 0.95, 25)) for _ in range(4)]
            for _ in range(2)
        ]
        self.ts = _trial_segments_from_relative(rel, 1.0, 4)
        self.pairing = np.arange(4)

    def test_dense_and_per_pair_agree(self):
        """Both trial-path layouts blank to exactly zero."""
        for normalize in ("none", "corrcoef"):
            with self.subTest(normalize=normalize):
                common = dict(
                    bin_size=0.01,
                    max_lag=0.1,
                    normalize=normalize,
                    exclude_zero_lag_autocorr=True,
                )
                _, dense = ccg_trial_paired(self.ts, self.pairing, **common)
                _, per_pair = ccg_trial_paired(
                    self.ts,
                    self.pairing,
                    pairs=np.array([[0, 0]], dtype=np.int64),
                    **common,
                )
                half = dense.shape[-1] // 2
                self.assertEqual(dense[0, 0, half], 0.0)
                self.assertEqual(per_pair[0, half], 0.0)

    def test_blank_is_off_when_not_requested(self):
        """With the flag clear the centre bin keeps its value."""
        _, C = ccg_trial_paired(
            self.ts,
            self.pairing,
            pairs=np.array([[0, 0]], dtype=np.int64),
            bin_size=0.01,
            max_lag=0.1,
            normalize="none",
            exclude_zero_lag_autocorr=False,
        )
        half = C.shape[-1] // 2
        self.assertGreater(C[0, half], 0.0)


# ---------------------------------------------------------------------------
# Item 5: window= must mean the same thing for one trial and many
# ---------------------------------------------------------------------------


@needs_numba
class WrapperWindowTest(unittest.TestCase):
    """``ccg(window=...)`` clips spikes regardless of trial count."""

    @staticmethod
    def _spikes(n_trials):
        """Unit a/b pairs at 0.1 s (in window) and 2.0 s (out of window)."""
        data = np.empty((2, n_trials), dtype=object)
        data[0, 0] = np.array([0.1, 2.0])
        data[1, 0] = np.array([0.105, 2.005])
        for k in range(1, n_trials):
            data[0, k] = np.array([])
            data[1, k] = np.array([])
        return _ragged_da(data, units=["a", "b"], window=(0.0, 3.0))

    def test_single_trial_window_clips(self):
        """Only the in-window coincidence contributes."""
        _, out = ccg(
            self._spikes(1), bin_size=0.001, max_lag=0.02, window=(0.0, 1.0)
        )
        self.assertEqual(out.values[0, 1].sum(), 1.0)

    def test_multi_trial_window_clips(self):
        """The multi-trial path already clips; pin it against regression."""
        _, out = ccg(
            self._spikes(2),
            bin_size=0.001,
            max_lag=0.02,
            window=(0.0, 1.0),
            normalize="none",
        )
        self.assertEqual(out.values[0, 1].sum(), 1.0)

    def test_one_trial_matches_one_element_multi_trial(self):
        """The same spikes and window give the same answer either way.

        A single-trial array and a two-trial array whose second trial is
        empty carry identical data, but they dispatch to different code
        paths.  ``n_trials == 1`` always takes the session path, so this
        is the only way to compare the two.
        """
        common = dict(bin_size=0.001, max_lag=0.02, window=(0.0, 1.0))
        _, one = ccg(self._spikes(1), **common)
        _, many = ccg(self._spikes(2), normalize="none", **common)
        np.testing.assert_array_equal(one.values[0, 1], many.values[0, 1])

    def test_reported_window_matches_applied_window(self):
        """Metadata must not claim a window that was not enforced."""
        _, out = ccg(
            self._spikes(1), bin_size=0.001, max_lag=0.02, window=(0.0, 1.0)
        )
        self.assertEqual(out.attrs["window"], (0.0, 1.0))
        self.assertEqual(out.values[0, 1].sum(), 1.0)


# ---------------------------------------------------------------------------
# Items 10-12: input validation and invariants
# ---------------------------------------------------------------------------


@needs_numba
class InputValidationTest(unittest.TestCase):
    """Public entry points reject inputs that violate their invariants."""

    def test_clip_spikes_to_trials_rejects_unsorted(self):
        """The two-pointer algorithm requires sorted spike times."""
        with self.assertRaises(ValueError):
            clip_spikes_to_trials(
                [np.array([0.3, 0.1, 0.2])],
                np.array([[0.0, 1.0]]),
            )

    def test_non_finite_spike_times_rejected(self):
        """NaN/inf spike times corrupt the histogram silently."""
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ccg_between_sets_sparse(
                        [np.array([0.1, bad])],
                        [np.array([0.2])],
                        bin_size=0.001,
                        max_lag=0.01,
                        observation_window=(0.0, 1.0),
                    )

    def test_nonpositive_bin_size_rejected(self):
        """bin_size <= 0 makes the bin index undefined."""
        for bad in (0.0, -0.001):
            with self.subTest(bin_size=bad):
                with self.assertRaises(ValueError):
                    ccg_between_sets_sparse(
                        [np.array([0.1])],
                        [np.array([0.2])],
                        bin_size=bad,
                        max_lag=0.01,
                        observation_window=(0.0, 1.0),
                    )

    def test_negative_max_lag_rejected(self):
        """A negative max_lag has no meaningful bin layout.

        This passes today only incidentally — the negative bin count
        reaches ``np.zeros`` and raises there.  It should be rejected up
        front with a message that names the parameter.
        """
        with self.assertRaises(ValueError):
            ccg_between_sets_sparse(
                [np.array([0.1])],
                [np.array([0.2])],
                bin_size=0.001,
                max_lag=-0.01,
                observation_window=(0.0, 1.0),
            )

    def test_clip_to_window_rejects_expansion(self):
        """A window wider than a trial's support cannot be reconstructed.

        Re-slicing cannot recover spikes already clipped away, but the
        returned durations would claim the larger window and corrupt
        normalization.
        """
        ts = _segments(
            per_unit_trials=[[[0.0], [0.0]]],
            durations=[1.0, 1.0],
            pre=[0.5, 0.5],
            post=[0.5, 0.5],
        )
        with self.assertRaises(ValueError):
            clip_to_window(ts, (-2.0, 2.0))

    def test_clip_to_window_allows_shrinking(self):
        """Narrowing to a subset of every trial's support is valid."""
        ts = _segments(
            per_unit_trials=[[[-0.4, 0.0, 0.4], [-0.4, 0.0, 0.4]]],
            durations=[1.0, 1.0],
            pre=[0.5, 0.5],
            post=[0.5, 0.5],
        )
        out = clip_to_window(ts, (-0.2, 0.2))
        np.testing.assert_array_equal(out.counts, np.array([[1, 1]]))
        np.testing.assert_allclose(out.durations, [0.4, 0.4])


class LagExposureTest(unittest.TestCase):
    """``Q_b`` is the lag-band area of the trial square.

    The implementation reaches it in closed form.  These derive it by
    quadrature instead, so a change to the closed form has to survive an
    independent derivation rather than a restatement of itself.
    """

    @staticmethod
    def _band_area(lo: float, hi: float, dur: float) -> float:
        """Integrate the lag density over ``[lo, hi)`` numerically."""
        # Over ``[0, dur]**2`` the lag delta has unnormalized density
        # ``dur - |delta|``.  Put a node exactly on the kink at 0 so the
        # trapezoid rule is exact on each linear piece.
        nodes = np.linspace(lo, hi, 100001)
        if lo < 0.0 < hi:
            nodes = np.unique(np.concatenate([nodes, [0.0]]))
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(np.maximum(dur - np.abs(nodes), 0.0), nodes))

    def test_exposure_matches_numerical_integration(self):
        for bin_size, max_lag, dur in [
            (0.001, 0.01, 1.0),
            (0.05, 0.0031, 2.0),
            (0.002, 0.02, 0.4),
        ]:
            nbins = 2 * int(np.ceil(max_lag / bin_size)) + 1
            half = nbins // 2
            halfbin = bin_size / 2
            Q = _lag_exposure(nbins, bin_size, dur)
            with self.subTest(bin_size=bin_size, dur=dur):
                # Centre bin spans both lag signs.
                self.assertAlmostEqual(
                    Q[half],
                    self._band_area(-halfbin, halfbin, dur),
                    delta=1e-9 * dur**2,
                )
                for k in range(1, half + 1):
                    stop = halfbin + k * bin_size
                    area = self._band_area(stop - bin_size, stop, dur)
                    self.assertAlmostEqual(
                        Q[half + k], area, delta=1e-9 * dur**2
                    )
                    # Exposure is symmetric in lag sign.
                    self.assertEqual(Q[half - k], Q[half + k])

    def test_shape_is_exposure_over_duration_squared(self):
        Q = _lag_exposure(21, 0.002, 0.75)
        np.testing.assert_array_equal(
            _expected_counts_shape(21, 0.002, 0.75), Q / 0.75**2
        )

    def test_exposure_shrinks_with_lag(self):
        # Edge correction: bins further from zero lag have less of the
        # trial square available to them, so exposure decreases.
        Q = _lag_exposure(21, 0.002, 0.5)
        side = Q[11:]
        self.assertTrue(np.all(np.diff(side) < 0))


class DenseProjectionTest(unittest.TestCase):
    """``to_dense`` is the single owner of the mirror policy."""

    @staticmethod
    def _counts(pairing, n_units=3, nbins=5):
        n_pairs = n_units * (n_units + 1) // 2
        pairs = np.array(
            [(i, j) for i in range(n_units) for j in range(i, n_units)]
        )
        vals = np.arange(n_pairs * nbins, dtype=np.float64).reshape(
            n_pairs, nbins
        )
        return vals, CCGCounts(
            counts=vals,
            lags=np.arange(nbins, dtype=np.float64),
            pairs=pairs,
            n_units=n_units,
            bin_size=1.0,
            pairing=pairing,
        )

    def test_involution_mirrors_by_the_flip_rule(self):
        vals, counts = self._counts(np.array([1, 0, 3, 2]))
        self.assertTrue(counts.mirror_is_defined())
        dense = to_dense(vals, counts)
        for p, (i, j) in enumerate(counts.pairs):
            np.testing.assert_array_equal(dense[i, j], vals[p])
            if i != j:
                np.testing.assert_array_equal(dense[j, i], vals[p][::-1])

    def test_non_involution_refuses_to_invent_the_transpose(self):
        # A 3-cycle: sigma(sigma(k)) != k, so reverse(h) is h^{sigma^-1}
        # where the transposed cell needs h^sigma.  Unrecoverable, so NaN
        # rather than a plausible wrong number.
        vals, counts = self._counts(np.array([1, 2, 0]))
        self.assertFalse(counts.mirror_is_defined())
        dense = to_dense(vals, counts)
        for p, (i, j) in enumerate(counts.pairs):
            np.testing.assert_array_equal(dense[i, j], vals[p])
            if i != j:
                self.assertTrue(np.isnan(dense[j, i]).all())

    def test_session_result_has_no_pairing_and_mirrors(self):
        vals, counts = self._counts(None)
        self.assertTrue(counts.mirror_is_defined())
        self.assertFalse(np.isnan(to_dense(vals, counts)).any())

    def test_uncovered_cells_take_fill(self):
        vals = np.ones((1, 3))
        counts = CCGCounts(
            counts=vals,
            lags=np.arange(3, dtype=np.float64),
            pairs=np.array([[0, 1]]),
            n_units=3,
            bin_size=1.0,
        )
        dense = to_dense(vals, counts, fill=-7.0)
        np.testing.assert_array_equal(dense[2, 2], [-7.0] * 3)
        np.testing.assert_array_equal(dense[0, 1], [1.0] * 3)

    def test_values_default_to_the_carried_counts(self):
        vals, counts = self._counts(None)
        np.testing.assert_array_equal(
            to_dense(None, counts), to_dense(vals, counts)
        )


class MirrorPreconditionTest(unittest.TestCase):
    """``pair_vec_to_NN`` makes the caller state the mirror decision."""

    def test_mirror_is_required(self):
        # Defaulting it silently transposed directional statistics.
        with self.assertRaises(TypeError):
            pair_vec_to_NN(np.array([1.0]), np.array([[0, 1]]), 2)

    def test_explicit_mirror_still_works_both_ways(self):
        v, pairs = np.array([5.0]), np.array([[0, 1]])
        on = pair_vec_to_NN(v, pairs, 2, fill=0.0, mirror=True)
        off = pair_vec_to_NN(v, pairs, 2, fill=0.0, mirror=False)
        self.assertEqual(on[1, 0], 5.0)
        self.assertEqual(off[1, 0], 0.0)


@needs_numba
class TransformTest(unittest.TestCase):
    """Lag-profile statistics and the integrated effect size."""

    @staticmethod
    def _segments(bin_seed=0, n_trials=12, dur=2.0, rate=60.0, coupling=0.3):
        """Two units where j fires extra spikes 5 ms after each i spike."""
        rng = np.random.default_rng(bin_seed)
        starts = np.arange(n_trials) * (dur + 1.0)
        epochs = np.column_stack([starts, starts + dur])
        spikes_i, spikes_j = [], []
        for t0 in starts:
            si = np.sort(rng.uniform(0, dur, rng.poisson(rate * dur)))
            sj = np.sort(rng.uniform(0, dur, rng.poisson(rate * dur)))
            # Injected coupling: a fraction of i's spikes drive one at +5 ms.
            driven = si[rng.random(si.size) < coupling] + 0.005
            driven = driven[driven < dur]
            spikes_i.append(si + t0)
            spikes_j.append(np.sort(np.concatenate([sj, driven])) + t0)
        return clip_spikes_to_trials(
            [np.concatenate(spikes_i), np.concatenate(spikes_j)],
            epochs,
            align_times=starts,
        )

    def test_directional_excess_is_invariant_across_bin_sizes(self):
        # The plan's acceptance test 8.  A peak height is a density and
        # moves with the bin width; an integrated count ratio does not.
        # Averaged over seeds: a single draw carries enough sampling noise
        # to swamp the invariance being tested.  The window starts at zero
        # because edge bins are weighted by overlap, so it means the same
        # lag interval at every bin width.
        window = (0.0, 0.02)
        values = []
        for bin_size in [0.0005, 0.001, 0.002, 0.004]:
            per_seed = []
            for seed in range(8):
                counts = compute_ccg_counts(
                    self._segments(seed),
                    np.arange(12),
                    bin_size=bin_size,
                    max_lag=0.05,
                    include_autocorr=False,
                )
                per_seed.append(float(directional_excess(counts, window)[0]))
            values.append(float(np.mean(per_seed)))
        spread = max(values) - min(values)
        self.assertLess(
            spread,
            0.025 * abs(np.mean(values)),
            f"directional_excess moved with bin size: {values}",
        )
        # And it should recover roughly the injected coupling.
        self.assertGreater(np.mean(values), 0.15)

    def test_peak_density_does_move_with_bin_size(self):
        # The contrast that makes the previous test meaningful: a peak
        # height is a density, so it is resolution-dependent where the
        # integrated count ratio is not.
        ts = self._segments()
        peaks = []
        for bin_size in [0.0005, 0.004]:
            counts = compute_ccg_counts(
                ts,
                np.arange(len(ts.durations)),
                bin_size=bin_size,
                max_lag=0.05,
                include_autocorr=False,
            )
            peaks.append(float(np.nanmax(covariance_density(counts)[0])))
        self.assertGreater(peaks[0], 4 * peaks[1])

    def test_baselines_are_where_the_definitions_say(self):
        # Independent units: covariance density ~0, pair correlation ~1.
        rng = np.random.default_rng(7)
        n_trials, dur = 40, 2.0
        starts = np.arange(n_trials) * (dur + 1.0)
        epochs = np.column_stack([starts, starts + dur])
        trains = [
            np.concatenate(
                [
                    np.sort(rng.uniform(0, dur, rng.poisson(80.0 * dur))) + t0
                    for t0 in starts
                ]
            )
            for _ in range(2)
        ]
        ts = clip_spikes_to_trials(trains, epochs, align_times=starts)
        counts = compute_ccg_counts(
            ts,
            np.arange(n_trials),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
        )
        self.assertAlmostEqual(
            float(np.mean(covariance_density(counts)[0])), 0.0, delta=200.0
        )
        self.assertAlmostEqual(
            float(np.mean(pair_correlation(counts)[0])), 1.0, delta=0.05
        )
        # cross_intensity sits at lambda_i lambda_j, not at zero.
        lam_i, lam_j = counts.rates()
        self.assertAlmostEqual(
            float(np.mean(cross_intensity(counts)[0])),
            float(lam_i[0] * lam_j[0]),
            delta=0.1 * float(lam_i[0] * lam_j[0]),
        )

    def test_transforms_respect_out(self):
        ts = self._segments()
        counts = compute_ccg_counts(
            ts,
            np.arange(len(ts.durations)),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
        )
        for fn in (
            cross_intensity,
            covariance_density,
            normalized_covariance,
            pair_correlation,
            legacy_auto_normalized,
        ):
            with self.subTest(fn=fn.__name__):
                buf = np.zeros_like(counts.counts)
                got = fn(counts, out=buf)
                self.assertIs(got, buf)
                np.testing.assert_array_equal(buf, fn(counts))

    def test_density_normalizations_need_a_common_geometry(self):
        # Trials of differing duration have no single Q_b.
        starts = np.arange(4) * 5.0
        epochs = np.column_stack([starts, starts + np.array([1.0, 2.0] * 2)])
        trains = [starts + 0.1, starts + 0.2]
        ts = clip_spikes_to_trials(trains, epochs, align_times=starts)
        counts = compute_ccg_counts(
            ts, np.arange(4), bin_size=0.01, max_lag=0.05
        )
        self.assertIsNone(counts.exposure)
        with self.assertRaisesRegex(ValueError, "no single lag exposure"):
            covariance_density(counts)

    def test_legacy_matches_the_corrcoef_path(self):
        ts = self._segments()
        pairing = np.arange(len(ts.durations))
        pairs = np.array([[0, 1]])
        _, direct = ccg_trial_paired(
            ts,
            pairing,
            bin_size=0.002,
            max_lag=0.02,
            normalize="corrcoef",
            pairs=pairs,
        )
        counts = compute_ccg_counts(
            ts, pairing, bin_size=0.002, max_lag=0.02, pairs=pairs
        )
        np.testing.assert_array_equal(direct, legacy_auto_normalized(counts))


class WindowWeightTest(unittest.TestCase):
    """Window edges cut bins by overlap instead of snapping to bin edges."""

    @staticmethod
    def _lags(bin_size=0.002, half=5):
        return (np.arange(2 * half + 1) - half) * bin_size

    def test_zero_edge_takes_exactly_half_the_centre_bin(self):
        # The centre bin spans [-w/2, +w/2) and its exposure is even in
        # lag, so half of it is exactly the i -> j side.  Not an
        # approximation, and not a reason to drop the bin.
        bin_size = 0.002
        lags = self._lags(bin_size)
        w = _window_weights(lags, bin_size, (0.0, 0.02))
        self.assertEqual(w[len(lags) // 2], 0.5)

    def test_weights_are_one_inside_and_zero_outside(self):
        bin_size = 0.002
        lags = self._lags(bin_size)
        w = _window_weights(lags, bin_size, (0.003, 0.007))
        # Bin centred at 0.004 lies wholly inside [0.003, 0.007).
        self.assertEqual(w[lags.tolist().index(0.004)], 1.0)
        self.assertEqual(w[lags.tolist().index(-0.004)], 0.0)

    def test_the_edge_moves_the_weight_continuously(self):
        # The smell being fixed: with bin selection, nudging the window
        # edge across a bin boundary jumps the result by a whole bin.
        bin_size = 0.002
        lags = self._lags(bin_size)
        totals = [
            _window_weights(lags, bin_size, (edge, 0.01)).sum()
            for edge in np.linspace(-0.004, 0.004, 41)
        ]
        steps = np.abs(np.diff(totals))
        self.assertLess(steps.max(), 0.15)
        self.assertAlmostEqual(totals[0] - totals[-1], 4.0, places=6)

    def test_full_window_matches_no_window(self):
        bin_size = 0.002
        lags = self._lags(bin_size)
        w = _window_weights(lags, bin_size, (lags[0] - 1.0, lags[-1] + 1.0))
        np.testing.assert_allclose(w, 1.0, rtol=1e-12)

    def test_backwards_window_raises(self):
        with self.assertRaisesRegex(ValueError, "low < high"):
            _window_weights(self._lags(), 0.002, (0.01, 0.001))


class RectangularProjectionTest(unittest.TestCase):
    """Two distinct unit sets project to a rectangle, and never mirror."""

    @staticmethod
    def _counts(n_rows=2, n_cols=3, nbins=4):
        pairs = np.array(
            [(i, j) for i in range(n_rows) for j in range(n_cols)]
        )
        vals = np.arange(pairs.shape[0] * nbins, dtype=np.float64).reshape(
            pairs.shape[0], nbins
        )
        return vals, CCGCounts(
            counts=vals,
            lags=np.arange(nbins, dtype=np.float64),
            pairs=pairs,
            n_units=(n_rows, n_cols),
            bin_size=1.0,
        )

    def test_shape_follows_the_two_sets(self):
        vals, counts = self._counts()
        self.assertFalse(counts.is_square)
        self.assertEqual(counts.grid, (2, 3))
        self.assertEqual(to_dense(vals, counts).shape, (2, 3, 4))

    def test_nothing_is_mirrored(self):
        # (j, i) is not a cell of this array, so there is nothing to
        # mirror -- unlike the square non-involution case, which is a
        # genuinely unknown value and gets NaN.
        vals, counts = self._counts()
        self.assertFalse(counts.mirror_is_defined())
        dense = to_dense(vals, counts)
        for p, (i, j) in enumerate(counts.pairs):
            np.testing.assert_array_equal(dense[i, j], vals[p])

    def test_square_still_mirrors(self):
        vals, counts = self._counts()
        counts.n_units = 3
        self.assertTrue(counts.is_square)
        self.assertEqual(counts.grid, (3, 3))

    def test_dtype_is_honoured(self):
        vals, counts = self._counts()
        self.assertEqual(to_dense(vals, counts).dtype, np.float64)
        out = to_dense(vals, counts, dtype=np.float32)
        self.assertEqual(out.dtype, np.float32)
        # float32 halves a projection that is ~1 GB at N=499, B=1001.
        np.testing.assert_allclose(out, vals.reshape(2, 3, 4), rtol=1e-6)


@needs_numba
class BaselineLayerTest(unittest.TestCase):
    """`baseline=` is an axis separate from the statistic."""

    @staticmethod
    def _drifting(seed=0, n_trials=40, dur=1.0, base=40.0, swing=25.0):
        """Two units conditionally independent within every trial.

        Their rates rise and fall together across trials, so they are
        genuinely correlated -- this is not a null pair.  What they lack is
        any fine-timescale structure: given each trial's spike counts, the
        spike *times* are independent.
        """
        rng = np.random.default_rng(seed)
        starts = np.arange(n_trials) * (dur + 0.5)
        epochs = np.column_stack([starts, starts + dur])
        # Shared across-trial modulation, period 20 trials.
        rate = base + swing * np.sin(2 * np.pi * np.arange(n_trials) / 20)
        a, b = [], []
        for t0, r in zip(starts, rate):
            a.append(np.sort(rng.uniform(0, dur, rng.poisson(r * dur))) + t0)
            b.append(np.sort(rng.uniform(0, dur, rng.poisson(r * dur))) + t0)
        return clip_spikes_to_trials(
            [np.concatenate(a), np.concatenate(b)],
            epochs,
            align_times=starts,
        )

    def _counts(self, ts, **kw):
        kw.setdefault("bin_size", 0.002)
        kw.setdefault("max_lag", 0.02)
        kw.setdefault("include_autocorr", False)
        return compute_ccg_counts(ts, np.arange(len(ts.durations)), **kw)

    def test_stationary_reports_the_shared_modulation(self):
        # `stationary` is a reference, not a correction. The shared drift
        # is a real correlation, so reporting it is a correct answer to a
        # broader question; conditioning on per-trial counts answers the
        # narrower fine-timescale one and leaves ~0.
        cond, stat = [], []
        for seed in range(12):
            c = self._counts(self._drifting(seed))
            lam_i, lam_j = c.rates()
            scale = float(lam_i[0] * lam_j[0])
            cond.append(np.mean(excess_density(c)[0]) / scale)
            stat.append(
                np.mean(excess_density(c, stationary_baseline(c))[0]) / scale
            )
        cond_m, stat_m = float(np.mean(cond)), float(np.mean(stat))
        # Conditional-uniform: no residual. Stationary: a clear offset.
        self.assertLess(abs(cond_m), 0.01)
        self.assertGreater(stat_m, 0.05)
        self.assertGreater(stat_m, 5 * abs(cond_m))

    def test_stationary_is_rank_one_and_matches_its_definition(self):
        c = self._counts(self._drifting())
        base = stationary_baseline(c)
        lam_i, lam_j = c.rates()
        np.testing.assert_allclose(
            base.row(0), c.exposure * lam_i[0] * lam_j[0], rtol=1e-12
        )

    def test_shift_predictor_is_a_ccg_under_a_shifted_pairing(self):
        ts = self._drifting()
        shift = shift_predictor_baseline(
            ts, bin_size=0.002, max_lag=0.02, include_autocorr=False
        )
        direct = compute_ccg_counts(
            ts,
            np.roll(np.arange(len(ts.durations)), 1),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
            with_expected=False,
        )
        np.testing.assert_array_equal(shift.counts, direct.counts)

    def test_shift_predictor_rejects_a_no_op_offset(self):
        ts = self._drifting(n_trials=40)
        with self.assertRaisesRegex(ValueError, "identity"):
            shift_predictor_baseline(
                ts, offset=40, bin_size=0.002, max_lag=0.02
            )

    def test_a_baseline_may_be_another_ccg_or_a_plain_array(self):
        ts = self._drifting()
        c = self._counts(ts)
        shift = shift_predictor_baseline(
            ts, bin_size=0.002, max_lag=0.02, include_autocorr=False
        )
        # A CCGCounts works directly; so does its array.
        np.testing.assert_allclose(
            excess_density(c, shift), excess_density(c, shift.counts)
        )

    def test_surrogate_mean_averages_in_count_space(self):
        ts = self._drifting()
        rng = np.random.default_rng(1)
        n = len(ts.durations)
        pairings = [rng.permutation(n) for _ in range(4)]
        got = surrogate_mean_baseline(
            ts, pairings, bin_size=0.002, max_lag=0.02, include_autocorr=False
        )
        want = np.mean(
            [
                compute_ccg_counts(
                    ts,
                    pr,
                    bin_size=0.002,
                    max_lag=0.02,
                    include_autocorr=False,
                    with_expected=False,
                ).counts
                for pr in pairings
            ],
            axis=0,
        )
        np.testing.assert_allclose(got, want)

    def test_surrogate_mean_rejects_no_pairings(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            surrogate_mean_baseline(self._drifting(), [])


class BlankZeroLagTest(unittest.TestCase):
    """Blanking is a presentation step, after normalization."""

    @staticmethod
    def _counts():
        pairs = np.array([[0, 0], [0, 1], [1, 1]])
        vals = np.ones((3, 5))
        return vals, CCGCounts(
            counts=vals,
            lags=np.arange(5, dtype=np.float64),
            pairs=pairs,
            n_units=2,
            bin_size=1.0,
        )

    def test_only_self_pairs_are_blanked(self):
        vals, counts = self._counts()
        out = blank_zero_lag(vals, counts)
        self.assertEqual(out[0, 2], 0.0)
        self.assertEqual(out[2, 2], 0.0)
        self.assertEqual(out[1, 2], 1.0)

    def test_it_does_not_touch_the_input_by_default(self):
        vals, counts = self._counts()
        blank_zero_lag(vals, counts)
        self.assertEqual(vals[0, 2], 1.0)

    def test_out_allows_in_place(self):
        vals, counts = self._counts()
        got = blank_zero_lag(vals, counts, out=vals)
        self.assertIs(got, vals)
        self.assertEqual(vals[0, 2], 0.0)


@needs_numba
class BaselineSeparationTest(unittest.TestCase):
    """What each baseline does and does not remove."""

    @staticmethod
    def _shared_time_locked(seed=0, n_trials=60, dur=1.0):
        """Both units bump at the same latency in every trial.

        No across-trial rate drift at all, so per-trial counts carry no
        information about the confound.
        """
        rng = np.random.default_rng(seed)
        starts = np.arange(n_trials) * (dur + 0.5)
        epochs = np.column_stack([starts, starts + dur])
        a, b = [], []
        for t0 in starts:
            trains = []
            for _ in range(2):
                x = np.concatenate(
                    [
                        rng.uniform(0, dur, rng.poisson(25 * dur)),
                        rng.normal(0.5, 0.05, rng.poisson(15)),
                    ]
                )
                trains.append(np.sort(x[(x >= 0) & (x < dur)]))
            a.append(trains[0] + t0)
            b.append(trains[1] + t0)
        return clip_spikes_to_trials(
            [np.concatenate(a), np.concatenate(b)],
            epochs,
            align_times=starts,
        )

    def test_baselines_differ_by_a_constant_in_lag(self):
        # Both are proportional to Q_b, so the choice cannot change any
        # shape statistic -- only the level.
        ts = BaselineLayerTest._drifting(0)
        counts = compute_ccg_counts(
            ts,
            np.arange(len(ts.durations)),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
        )
        diff = (
            excess_density(counts, stationary_baseline(counts))[0]
            - excess_density(counts)[0]
        )
        self.assertLess(diff.max() - diff.min(), 1e-9 * abs(diff.mean()))

    def test_conditioning_on_counts_misses_time_locked_comodulation(self):
        # Per-trial counts say nothing about *when* in the trial spikes
        # fall, so a shared stimulus response survives the default
        # baseline and needs the shift predictor.
        cond, shift = [], []
        for seed in range(6):
            ts = self._shared_time_locked(seed)
            n = len(ts.durations)
            counts = compute_ccg_counts(
                ts,
                np.arange(n),
                bin_size=0.002,
                max_lag=0.02,
                include_autocorr=False,
            )
            predictor = shift_predictor_baseline(
                ts, bin_size=0.002, max_lag=0.02, include_autocorr=False
            )
            lam_i, lam_j = counts.rates()
            scale = float(lam_i[0] * lam_j[0])
            cond.append(np.mean(excess_density(counts)[0]) / scale)
            shift.append(np.mean(excess_density(counts, predictor)[0]) / scale)
        self.assertGreater(np.mean(cond), 0.2)
        self.assertLess(abs(np.mean(shift)), 0.05)


@needs_numba
class SurrogateNullTest(unittest.TestCase):
    """Subtracting E lets an unrestricted derangement reject slow drift.

    Plan acceptance tests 15 and 16.  Kept small -- the effect is large
    enough that a few seeds separate the arms cleanly, and this runs in
    the unit suite.
    """

    BIN, MAXLAG, N_TRIALS = 0.002, 0.02, 60

    @classmethod
    def _drifting(cls, seed, coupling=0.0, dur=1.0):
        """Rates co-vary across trials; optional real 5 ms coupling."""
        rng = np.random.default_rng(seed)
        n = cls.N_TRIALS
        starts = np.arange(n) * (dur + 0.5)
        epochs = np.column_stack([starts, starts + dur])
        a, b = [], []
        for k, t0 in enumerate(starts):
            rate = 40 + 25 * np.sin(2 * np.pi * k / 20)
            si = np.sort(rng.uniform(0, dur, rng.poisson(rate * dur)))
            sj = np.sort(rng.uniform(0, dur, rng.poisson(rate * dur)))
            driven = si[rng.random(si.size) < coupling] + 0.005
            a.append(si + t0)
            b.append(np.sort(np.concatenate([sj, driven[driven < dur]])) + t0)
        return clip_spikes_to_trials(
            [np.concatenate(a), np.concatenate(b)],
            epochs,
            align_times=starts,
        )

    @staticmethod
    def _derangement(rng, n):
        while True:
            p = rng.permutation(n)
            if np.all(p != np.arange(n)):
                return p

    @classmethod
    def _statistic(cls, ts, pairing, subtract_expected):
        counts = compute_ccg_counts(
            ts,
            pairing,
            bin_size=cls.BIN,
            max_lag=cls.MAXLAG,
            include_autocorr=False,
            with_expected=subtract_expected,
        )
        if not subtract_expected:
            return counts.counts[0]
        return counts.counts[0] - counts.expected_row(0)

    @classmethod
    def _z(cls, ts, seed, subtract_expected, n_surrogates=40):
        """Mean per-bin z against an unrestricted derangement null."""
        rng = np.random.default_rng(seed)
        n = cls.N_TRIALS
        obs = cls._statistic(ts, np.arange(n), subtract_expected)
        surr = np.stack(
            [
                cls._statistic(ts, cls._derangement(rng, n), subtract_expected)
                for _ in range(n_surrogates)
            ]
        )
        sd = surr.std(axis=0, ddof=1)
        sd[sd == 0] = np.inf
        return float(np.mean((obs - surr.mean(axis=0)) / sd))

    def test_raw_counts_alone_report_the_drift_as_an_effect(self):
        # The false positive that motivates constrained shuffles: no
        # fine-timescale coupling, but the rates co-vary.
        z = np.mean(
            [self._z(self._drifting(s), 100 + s, False) for s in range(3)]
        )
        self.assertGreater(z, 1.5)

    def test_subtracting_expected_removes_it_without_constraining_sigma(self):
        # E is pairing-dependent, so it tracks the count covariance on both
        # sides of the comparison rather than cancelling.
        z = np.mean(
            [self._z(self._drifting(s), 200 + s, True) for s in range(3)]
        )
        self.assertLess(abs(z), 0.6)

    def test_a_real_coupling_still_survives_the_subtraction(self):
        # The property a constrained shuffle gives up: E removes only the
        # count-level covariance, leaving fine-timescale structure intact.
        z = np.mean(
            [
                self._z(self._drifting(s, coupling=0.25), 300 + s, True)
                for s in range(3)
            ]
        )
        self.assertGreater(z, 1.2)


class TrialShuffleTest(unittest.TestCase):
    """Shuffle schemes are valid derangements and carry their provenance."""

    def test_every_mode_yields_derangements(self):
        rng = np.random.default_rng(0)
        ident = np.arange(24)
        for mode, kw in [
            ("global", {}),
            ("within_block", {"block_size": 6}),
            ("circular_offset", {}),
        ]:
            with self.subTest(mode=mode):
                for pairing in TrialShuffle(24, mode=mode, **kw).draw(20, rng):
                    self.assertEqual(sorted(pairing.tolist()), ident.tolist())
                    self.assertTrue(np.all(pairing != ident))

    def test_within_block_never_crosses_a_boundary(self):
        rng = np.random.default_rng(1)
        shuffle = TrialShuffle(24, mode="within_block", block_size=6)
        for pairing in shuffle.draw(20, rng):
            self.assertTrue(np.all(pairing // 6 == np.arange(24) // 6))

    def test_a_trailing_singleton_block_is_absorbed(self):
        # A block of one trial has no derangement, so it must join the
        # block before it rather than silently keeping a fixed point.
        rng = np.random.default_rng(2)
        shuffle = TrialShuffle(13, mode="within_block", block_size=4)
        self.assertEqual([b.size for b in shuffle._blocks()], [4, 4, 5])
        for pairing in shuffle.draw(10, rng):
            self.assertTrue(np.all(pairing != np.arange(13)))

    def test_provenance_records_the_scheme(self):
        prov = TrialShuffle(20, mode="within_block", block_size=5).provenance()
        self.assertEqual(prov["surrogate_method"], "trial_permutation")
        self.assertEqual(prov["shuffle_scope"], "within_block")
        self.assertEqual(prov["block_size"], 5)

    def test_invalid_schemes_are_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "deferred"):
            TrialShuffle(10, mode="local")
        with self.assertRaisesRegex(ValueError, "unknown shuffle mode"):
            TrialShuffle(10, mode="wiggle")
        with self.assertRaisesRegex(ValueError, "requires block_size"):
            TrialShuffle(10, mode="within_block")
        with self.assertRaisesRegex(ValueError, ">= 2"):
            TrialShuffle(10, mode="within_block", block_size=1)
        with self.assertRaisesRegex(ValueError, "identity"):
            TrialShuffle(10, mode="circular_offset", offset=10)

    def test_a_fixed_offset_is_not_random(self):
        self.assertFalse(
            TrialShuffle(10, mode="circular_offset", offset=1).is_random
        )
        self.assertTrue(TrialShuffle(10, mode="circular_offset").is_random)
        self.assertTrue(TrialShuffle(10).is_random)


@needs_numba
class SurrogateTestApiTest(unittest.TestCase):
    """Count-space inference against a trial-permutation null."""

    KW = dict(bin_size=0.002, max_lag=0.02, include_autocorr=False)
    N_TRIALS = 60

    @classmethod
    def _fixture(cls, seed, coupling=0.0, dur=1.0):
        rng = np.random.default_rng(seed)
        n = cls.N_TRIALS
        starts = np.arange(n) * (dur + 0.5)
        epochs = np.column_stack([starts, starts + dur])
        a, b = [], []
        for k, t0 in enumerate(starts):
            rate = 40 + 25 * np.sin(2 * np.pi * k / 20)
            si = np.sort(rng.uniform(0, dur, rng.poisson(rate * dur)))
            sj = np.sort(rng.uniform(0, dur, rng.poisson(rate * dur)))
            driven = si[rng.random(si.size) < coupling] + 0.005
            a.append(si + t0)
            b.append(np.sort(np.concatenate([sj, driven[driven < dur]])) + t0)
        return clip_spikes_to_trials(
            [np.concatenate(a), np.concatenate(b)],
            epochs,
            align_times=starts,
        )

    @staticmethod
    def _peak(values, counts):
        window = (counts.lags >= 0.002) & (counts.lags < 0.012)
        return values[:, window].max(axis=1)

    def _run(self, ts, mode="global", n=40, seed=7, **kw):
        shuffle = TrialShuffle(self.N_TRIALS, mode=mode, **kw)
        return surrogate_null(
            ts,
            shuffle,
            n,
            np.random.default_rng(seed),
            reduce=self._peak,
            **self.KW,
        )

    def test_a_drifting_null_pair_is_not_significant(self):
        z = np.mean(
            [
                float(self._run(self._fixture(s), seed=10 + s).z()[0])
                for s in range(3)
            ]
        )
        self.assertLess(abs(z), 1.5)

    def test_a_real_coupling_is_detected(self):
        result = self._run(self._fixture(0, coupling=0.25))
        self.assertGreater(float(result.z()[0]), 5.0)
        self.assertLessEqual(float(result.p_value()[0]), 1.0 / 41.0 + 1e-12)

    def test_welford_matches_a_direct_computation(self):
        result = self._run(self._fixture(0), n=25)
        np.testing.assert_allclose(
            result.mean, result.draws.mean(axis=0), rtol=1e-10
        )
        np.testing.assert_allclose(
            result.sd, result.draws.std(axis=0, ddof=1), rtol=1e-10
        )

    def test_p_values_need_the_draws(self):
        shuffle = TrialShuffle(self.N_TRIALS)
        result = surrogate_null(
            self._fixture(0),
            shuffle,
            5,
            np.random.default_rng(0),
            **self.KW,
        )
        self.assertIsNone(result.draws)
        with self.assertRaisesRegex(ValueError, "need the surrogate draws"):
            result.p_value()
        # Welford still works without them, over the full (pair, lag) grid.
        self.assertEqual(result.mean.shape, result.observed.shape)

    def test_on_draw_sees_every_surrogate(self):
        # The hook a long run reports progress from; it must count the
        # draws, not the observed statistic, which is not one of them.
        seen = []
        surrogate_null(
            self._fixture(0),
            TrialShuffle(self.N_TRIALS),
            7,
            np.random.default_rng(0),
            reduce=self._peak,
            on_draw=seen.append,
            **self.KW,
        )
        self.assertEqual(seen, list(range(1, 8)))

    def test_p_value_is_never_zero(self):
        result = self._run(self._fixture(0, coupling=0.5), n=10)
        self.assertGreater(float(result.p_value()[0]), 0.0)

    def test_a_deterministic_shuffle_is_refused(self):
        # Every draw identical means zero spread and universal
        # significance; that must fail loudly rather than report p.
        shuffle = TrialShuffle(self.N_TRIALS, mode="circular_offset", offset=1)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            surrogate_null(
                self._fixture(0),
                shuffle,
                20,
                np.random.default_rng(0),
                reduce=self._peak,
                **self.KW,
            )

    def test_the_scheme_travels_with_the_result(self):
        result = self._run(
            self._fixture(0), mode="within_block", block_size=10
        )
        self.assertEqual(result.provenance["shuffle_scope"], "within_block")
        self.assertEqual(result.provenance["block_size"], 10)
        self.assertEqual(result.provenance["n_surrogates"], 40)
        self.assertTrue(result.provenance["subtract_expected"])

    def test_a_mismatched_trial_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "built for"):
            surrogate_null(
                self._fixture(0),
                TrialShuffle(7),
                5,
                np.random.default_rng(0),
                **self.KW,
            )

    def test_auto_terms_are_off_by_default(self):
        # They belong to the legacy statistic; count-space comparison never
        # uses them, and they are a third of the cost at many units.
        self.assertIsInstance(self._run(self._fixture(0), n=3), SurrogateTest)


@needs_numba
class SurrogateDenominatorTest(unittest.TestCase):
    """The legacy denominator does not vary with the trial pairing.

    ``auto_sum_paired`` sums each unit's auto terms over *all* trials
    reindexed by sigma, and a sum is permutation-invariant.  Under the
    support contract the overlap mask is sigma-independent too, so the
    whole denominator is a per-pair constant.  That is why comparing
    corrcoef-normalized surrogates is equivalent to comparing
    ``H - E`` in count space: the constant cancels in any z-score.
    """

    @staticmethod
    def _denominators(ts, pairings):
        out = []
        for pairing in pairings:
            counts = compute_ccg_counts(
                ts,
                pairing,
                bin_size=0.01,
                max_lag=0.05,
                include_autocorr=False,
            )
            out.append(
                np.sqrt(abs(counts.auto_direct[0] * counts.auto_paired[1]))
            )
        return np.array(out)

    def test_invariant_under_uniform_support(self):
        n = 20
        rng = np.random.default_rng(0)
        starts = np.arange(n) * 2.0
        epochs = np.column_stack([starts, starts + 1.0])
        trains = [
            np.sort(rng.uniform(0, n * 2.0, rng.poisson(30.0 * n * 2.0)))
            for _ in range(3)
        ]
        ts = clip_spikes_to_trials(trains, epochs, align_times=starts)
        pairings = [np.arange(n)] + list(
            _module_derangements(n, 4, np.random.default_rng(1))
        )
        d = self._denominators(ts, pairings)
        self.assertLess((d.max() - d.min()) / d.mean(), 1e-12)

    def test_invariant_under_matched_mixed_support(self):
        # Two support classes, so the overlap mask is live rather than
        # trivially all-true.
        n = 8
        rng = np.random.default_rng(5)
        starts = np.arange(n) * 3.0
        epochs = np.column_stack([starts, starts + 1.0])
        align = starts + np.where(np.arange(n) % 2 == 0, 0.2, 0.8)
        trains = [
            np.sort(rng.uniform(0, n * 3.0, rng.poisson(30.0 * n * 3.0)))
            for _ in range(3)
        ]
        ts = clip_spikes_to_trials(trains, epochs, align_times=align)
        self.assertFalse(_uniform_support(ts))
        d = self._denominators(
            ts,
            [
                np.arange(n),
                np.array([2, 3, 0, 1, 6, 7, 4, 5]),
                np.array([6, 7, 4, 5, 2, 3, 0, 1]),
            ],
        )
        self.assertLess((d.max() - d.min()) / d.mean(), 1e-12)


@needs_numba
class CorrcoefDeprecationTest(unittest.TestCase):
    """``"corrcoef"`` is an alias for ``"legacy_auto_normalized"``."""

    @staticmethod
    def _segments(n=20, dur=1.0, seed=0):
        rng = np.random.default_rng(seed)
        starts = np.arange(n) * 2.0
        epochs = np.column_stack([starts, starts + dur])
        trains = [
            np.concatenate(
                [
                    np.sort(rng.uniform(0, dur, rng.poisson(40))) + t0
                    for t0 in starts
                ]
            )
            for _ in range(2)
        ]
        return clip_spikes_to_trials(trains, epochs, align_times=starts)

    def test_the_old_spelling_warns(self):
        ts = self._segments()
        with self.assertWarns(DeprecationWarning):
            ccg_trial_paired(
                ts,
                np.arange(20),
                bin_size=0.002,
                max_lag=0.02,
                normalize="corrcoef",
            )

    def test_the_new_spelling_does_not_warn(self):
        ts = self._segments()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ccg_trial_paired(
                ts,
                np.arange(20),
                bin_size=0.002,
                max_lag=0.02,
                normalize="legacy_auto_normalized",
            )
        self.assertEqual(
            [w for w in caught if issubclass(w.category, DeprecationWarning)],
            [],
        )

    def test_both_spellings_give_the_same_numbers(self):
        ts = self._segments()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _, old = ccg_trial_paired(
                ts,
                np.arange(20),
                bin_size=0.002,
                max_lag=0.02,
                normalize="corrcoef",
            )
        _, new = ccg_trial_paired(
            ts,
            np.arange(20),
            bin_size=0.002,
            max_lag=0.02,
            normalize="legacy_auto_normalized",
        )
        np.testing.assert_array_equal(old, new)

    def test_an_unknown_mode_is_still_rejected(self):
        ts = self._segments()
        with self.assertRaisesRegex(ValueError, "Unknown or unsupported"):
            ccg_trial_paired(ts, np.arange(20), normalize="bogus")


class UndefinedAutoTermTest(unittest.TestCase):
    """A non-positive auto product makes the statistic undefined, not |x|."""

    def test_non_positive_product_gives_nan(self):
        # abs() was a port-side deviation; the Julia reference's bare sqrt
        # throws.  NaN, so one degenerate pair does not fail a batch.
        pairs = np.array([[0, 1]])
        counts = CCGCounts(
            counts=np.ones((1, 5)),
            lags=np.arange(5, dtype=np.float64),
            pairs=pairs,
            n_units=2,
            bin_size=1.0,
            expected_factors=(np.zeros(5), np.zeros(1)),
            auto_direct=np.array([-4.0, 1.0]),
            auto_paired=np.array([1.0, 1.0]),
        )
        self.assertTrue(np.isnan(legacy_auto_normalized(counts)).all())


@needs_numba
class LagSignTest(unittest.TestCase):
    """``C[i, j]`` histograms ``t_j - t_i``, so a positive lag means *i* leads.

    Guarded on its own rather than as a side effect of some other test,
    because a reversal is invisible to every symmetric fixture --
    identical trains, autocorrelograms, a max taken over all lags.  The
    Julia reference this statistic was ported from ran for seven years
    taking ``u - v`` while its docstring described ``v - u``, and the only
    cross-language check used identical trains, which cannot tell the two
    apart.
    """

    DUR, DELAY, BIN, MAXLAG = 60.0, 0.004, 0.001, 0.02

    @classmethod
    def _leader_follower(cls, seed=11):
        """Unit 1 fires ``DELAY`` after unit 0, plus independent background."""
        rng = np.random.default_rng(seed)
        lead = np.sort(rng.uniform(0, cls.DUR, 400))
        driven = lead + cls.DELAY
        follow = np.sort(
            np.concatenate(
                [driven[driven < cls.DUR], rng.uniform(0, cls.DUR, 400)]
            )
        )
        return lead, follow

    def _peak_lag(self, lags, profile):
        """Lag of the maximum of one correlogram."""
        return float(lags[int(np.argmax(profile))])

    def test_session_path_puts_the_follower_at_positive_lag(self):
        lead, follow = self._leader_follower()
        lags, C = ccg_allpairs_sparse(
            [lead, follow],
            bin_size=self.BIN,
            max_lag=self.MAXLAG,
            normalize="none",
            observation_window=(0.0, self.DUR),
        )
        peak = self._peak_lag(lags, C[0, 1])
        self.assertAlmostEqual(peak, self.DELAY, delta=self.BIN)
        # Say the failure out loud: a reversal lands it at -DELAY.
        self.assertGreater(peak, 0.0)

    def test_the_transposed_cell_mirrors_it(self):
        # If C[1, 0] did not peak at -DELAY the sign would be an accident
        # rather than a convention.
        lead, follow = self._leader_follower()
        lags, C = ccg_allpairs_sparse(
            [lead, follow],
            bin_size=self.BIN,
            max_lag=self.MAXLAG,
            normalize="none",
            observation_window=(0.0, self.DUR),
        )
        self.assertAlmostEqual(
            self._peak_lag(lags, C[1, 0]), -self.DELAY, delta=self.BIN
        )

    def test_swapping_the_arguments_flips_the_sign(self):
        lead, follow = self._leader_follower()
        kw = dict(
            bin_size=self.BIN,
            max_lag=self.MAXLAG,
            normalize="none",
            observation_window=(0.0, self.DUR),
        )
        lags, forward = ccg_between_sets_sparse([lead], [follow], **kw)
        _, reverse = ccg_between_sets_sparse([follow], [lead], **kw)
        self.assertAlmostEqual(
            self._peak_lag(lags, forward[0, 0]), self.DELAY, delta=self.BIN
        )
        self.assertAlmostEqual(
            self._peak_lag(lags, reverse[0, 0]), -self.DELAY, delta=self.BIN
        )

    def test_the_trial_path_agrees_with_the_session_path(self):
        # Two compute paths, one convention.
        lead, follow = self._leader_follower()
        n_trials = 20
        edges = np.linspace(0.0, self.DUR, n_trials + 1)
        epochs = np.column_stack([edges[:-1], edges[1:]])
        ts = clip_spikes_to_trials(
            [lead, follow], epochs, align_times=edges[:-1]
        )
        lags, C = ccg_trial_paired(
            ts,
            np.arange(n_trials),
            bin_size=self.BIN,
            max_lag=self.MAXLAG,
            normalize="none",
            pairs=np.array([[0, 1]]),
        )
        peak = self._peak_lag(lags, C[0])
        self.assertAlmostEqual(peak, self.DELAY, delta=self.BIN)
        self.assertGreater(peak, 0.0)


@needs_numba
class ReverseDirectionalExcessTest(unittest.TestCase):
    """``reverse=True`` reads j -> i off the i -> j rows, no second pass."""

    WINDOW = (0.001, 0.015)

    @staticmethod
    def _lopsided(seed=3, n_trials=40, dur=1.0):
        """A sparse unit driving a dense one, so the divisors differ."""
        rng = np.random.default_rng(seed)
        starts = np.arange(n_trials) * 2.0
        epochs = np.column_stack([starts, starts + dur])
        a, b = [], []
        for t0 in starts:
            src = np.sort(rng.uniform(0, dur, rng.poisson(12)))
            tgt = np.sort(rng.uniform(0, dur, rng.poisson(60)))
            driven = src[rng.random(src.size) < 0.4] + 0.004
            a.append(src + t0)
            b.append(np.sort(np.concatenate([tgt, driven[driven < dur]])) + t0)
        return clip_spikes_to_trials(
            [np.concatenate(a), np.concatenate(b)], epochs, align_times=starts
        )

    def _counts(self, pairs):
        ts = self._lopsided()
        return compute_ccg_counts(
            ts,
            np.arange(40),
            pairs=np.array(pairs),
            bin_size=0.001,
            max_lag=0.02,
        )

    def test_reverse_matches_a_second_computation(self):
        # The whole point: one kernel pass yields both orientations.
        forward = self._counts([[0, 1]])
        swapped = self._counts([[1, 0]])
        np.testing.assert_allclose(
            directional_excess(forward, self.WINDOW, reverse=True),
            directional_excess(swapped, self.WINDOW),
            rtol=1e-12,
        )

    def test_the_two_orientations_differ_only_by_the_source_count(self):
        # Read over corresponding windows the numerators are identical and
        # only the divisor changes, so the ratio is exactly n_j / n_i.  A
        # mirror-symmetric statistic would give 1 here instead.
        counts = self._counts([[0, 1]])
        mirrored = (-self.WINDOW[1], -self.WINDOW[0])
        fwd = float(directional_excess(counts, self.WINDOW)[0])
        rev = float(directional_excess(counts, mirrored, reverse=True)[0])
        n_i, n_j = (float(counts.n_spikes[k]) for k in (0, 1))
        self.assertAlmostEqual(fwd / rev, n_j / n_i, places=6)
        self.assertGreater(n_j / n_i, 2.0)  # the fixture is lopsided

    def test_reverse_reads_the_opposite_lag_side(self):
        # With coupling at +4 ms only, the j -> i direction over the same
        # nominal window sees background: it integrates negative lags.
        counts = self._counts([[0, 1]])
        fwd = float(directional_excess(counts, self.WINDOW)[0])
        rev = float(directional_excess(counts, self.WINDOW, reverse=True)[0])
        self.assertGreater(fwd, 20 * abs(rev))

    def test_reverse_needs_an_involutive_pairing(self):
        ts = self._lopsided()
        rolled = np.roll(np.arange(40), 3)  # a 40-cycle, not an involution
        counts = compute_ccg_counts(
            ts,
            rolled,
            pairs=np.array([[0, 1]]),
            bin_size=0.001,
            max_lag=0.02,
        )
        self.assertFalse(counts.mirror_is_defined())
        with self.assertRaisesRegex(ValueError, "involutive"):
            directional_excess(counts, self.WINDOW, reverse=True)
        # The forward direction is unaffected.
        self.assertTrue(
            np.isfinite(directional_excess(counts, self.WINDOW)).all()
        )


@needs_numba
class ChunkedExcessTest(unittest.TestCase):
    """Subtracting expected counts in chunks, into a reused buffer."""

    @staticmethod
    def _segments(n_units=6, n_trials=20, dur=1.0, seed=0):
        rng = np.random.default_rng(seed)
        starts = np.arange(n_trials) * 2.0
        epochs = np.column_stack([starts, starts + dur])
        trains = [
            np.concatenate(
                [
                    np.sort(rng.uniform(0, dur, rng.poisson(40))) + t0
                    for t0 in starts
                ]
            )
            for _ in range(n_units)
        ]
        return clip_spikes_to_trials(trains, epochs, align_times=starts)

    def _counts(self, **kw):
        ts = self._segments()
        return compute_ccg_counts(
            ts,
            np.arange(20),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
            **kw,
        )

    def test_chunks_reassemble_the_whole_array(self):
        counts = self._counts()
        whole = counts.expected_array()
        chunked = np.concatenate(
            [
                counts.expected_chunk(lo, min(lo + 4, counts.n_pairs))
                for lo in range(0, counts.n_pairs, 4)
            ]
        )
        np.testing.assert_array_equal(chunked, whole)

    def test_a_chunk_size_that_does_not_divide_evenly(self):
        # The last chunk is short; an off-by-one here would silently drop
        # or duplicate pairs.
        counts = self._counts()
        self.assertNotEqual(counts.n_pairs % 4, 0)
        self.assertEqual(
            counts.expected_chunk(0, counts.n_pairs).shape,
            counts.counts.shape,
        )

    def test_chunked_excess_matches_the_naive_difference(self):
        from aind_ephys_utils.metrics.ccg import _excess_into

        counts = self._counts()
        np.testing.assert_allclose(
            _excess_into(counts, None),
            counts.counts - counts.expected_array(),
            rtol=1e-12,
        )

    def test_the_buffer_is_reused_but_the_observed_is_not_clobbered(self):
        # statistic() hands back the shared scratch when nothing reduces
        # it, so surrogate_null must copy the observed before drawing.
        ts = self._segments()
        result = surrogate_null(
            ts,
            TrialShuffle(20),
            4,
            np.random.default_rng(0),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
        )
        direct = compute_ccg_counts(
            ts,
            np.arange(20),
            bin_size=0.002,
            max_lag=0.02,
            include_autocorr=False,
        )
        expected_observed = direct.counts - direct.expected_array()
        np.testing.assert_allclose(
            result.observed, expected_observed, rtol=1e-12
        )
        self.assertFalse(np.shares_memory(result.observed, result.mean))


if __name__ == "__main__":
    unittest.main()
