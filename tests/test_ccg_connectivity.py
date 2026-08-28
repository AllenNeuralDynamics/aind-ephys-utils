"""Tests for CCG per-pair helpers and the connectivity workflow."""

import unittest

import numpy as np

from aind_ephys_utils.metrics.ccg import (
    NN_to_pair_vec,
    TrialShuffle,
    ccg_trial_paired,
    clip_spikes_to_trials,
    compute_ccg_counts,
    derangements,
    legacy_auto_denominator,
    legacy_auto_normalized,
    pair_vec_to_NN,
    surrogate_null,
)
from aind_ephys_utils.metrics.connectivity import (
    build_shape_prefilter,
    run_surrogates,
    run_two_stage_mc,
)

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class PairVecTest(unittest.TestCase):
    """Per-pair <-> (N, N) conversion."""

    def test_round_trip_and_mirror(self) -> None:
        """Scatter then gather recovers the values; mirror fills (j, i)."""
        pairs = np.array([[0, 1], [0, 2]])
        values = np.array([5.0, 7.0])
        nn = pair_vec_to_NN(values, pairs, 3, fill=0.0, mirror=True)
        self.assertEqual(nn.shape, (3, 3))
        self.assertEqual(nn[0, 1], 5.0)
        self.assertEqual(nn[1, 0], 5.0)  # mirrored
        self.assertEqual(nn[0, 2], 7.0)
        self.assertEqual(nn[2, 0], 7.0)
        self.assertEqual(nn[1, 2], 0.0)  # fill
        np.testing.assert_array_equal(NN_to_pair_vec(nn, pairs), values)

    def test_no_mirror_and_fill(self) -> None:
        """mirror=False leaves (j, i) at fill."""
        pairs = np.array([[0, 1]])
        nn = pair_vec_to_NN(np.array([3.0]), pairs, 2, fill=-1.0, mirror=False)
        self.assertEqual(nn[0, 1], 3.0)
        self.assertEqual(nn[1, 0], -1.0)


class DerangementsTest(unittest.TestCase):
    """Derangement generator."""

    def test_no_fixed_points(self) -> None:
        """Every yielded permutation maps no index to itself."""
        rng = np.random.default_rng(0)
        out = list(derangements(6, 20, rng))
        self.assertEqual(len(out), 20)
        identity = np.arange(6)
        for p in out:
            self.assertEqual(p.shape, (6,))
            self.assertTrue(np.all(p != identity))
            self.assertEqual(sorted(p.tolist()), list(range(6)))

    def test_too_small_raises(self) -> None:
        """n < 2 has no derangement and is rejected."""
        with self.assertRaises(ValueError):
            list(derangements(1, 3, np.random.default_rng(0)))


class ShapePrefilterTest(unittest.TestCase):
    """Bourgon independent pre-filter."""

    def _peaks(self):
        """Synthetic (peak_lag, peak_fwhm) arrays for three pairs."""
        peak_lag = np.zeros((3, 3))
        peak_fwhm = np.zeros((3, 3))
        peak_lag[0, 1], peak_fwhm[0, 1] = 0.002, 0.003  # passes both
        peak_lag[0, 2], peak_fwhm[0, 2] = 0.050, 0.003  # latency too long
        peak_lag[1, 2], peak_fwhm[1, 2] = 0.001, 0.200  # FWHM too wide
        return peak_lag, peak_fwhm

    def test_ranges_filter(self) -> None:
        """Only the pair passing both latency and FWHM survives."""
        peak_lag, peak_fwhm = self._peaks()
        cross_pairs = np.array([[0, 1], [0, 2], [1, 2]])
        *_, prefilter_pairs = build_shape_prefilter(
            n_units=3,
            cross_pairs=cross_pairs,
            peak_lag=peak_lag,
            peak_fwhm=peak_fwhm,
            fwhm_range_s=(0.001, 0.01),
            latency_range_s=(0.0005, 0.01),
        )
        np.testing.assert_array_equal(prefilter_pairs, [[0, 1]])

    def test_ranges_none_keeps_all(self) -> None:
        """Disabling both ranges keeps every cross pair."""
        peak_lag, peak_fwhm = self._peaks()
        cross_pairs = np.array([[0, 1], [0, 2], [1, 2]])
        *_, prefilter_pairs = build_shape_prefilter(
            n_units=3,
            cross_pairs=cross_pairs,
            peak_lag=peak_lag,
            peak_fwhm=peak_fwhm,
            fwhm_range_s=None,
            latency_range_s=None,
        )
        np.testing.assert_array_equal(prefilter_pairs, cross_pairs)


def _make_trial_segments(rng, n_units=4, n_trials=8, rate=20.0):
    """Build TrialSegments from independent within-trial Poisson spikes."""
    trial_len = 1.0
    starts = np.arange(n_trials) * 2.0
    trial_epochs = np.column_stack([starts, starts + trial_len])
    spikes = []
    for _ in range(n_units):
        per_trial = [
            t + np.sort(rng.uniform(0, trial_len, rng.poisson(rate)))
            for t in starts
        ]
        spikes.append(np.sort(np.concatenate(per_trial)))
    segs = clip_spikes_to_trials(spikes, trial_epochs, align_times=starts)
    return segs, starts


@unittest.skipUnless(HAS_NUMBA, "requires the numba optional extra")
class SurrogateWorkflowTest(unittest.TestCase):
    """run_surrogates and run_two_stage_mc on a small real dataset."""

    def test_run_surrogates_shape(self) -> None:
        """Surrogate driver returns a finite (n_surr, n_pairs) float32."""
        rng = np.random.default_rng(0)
        segs, _ = _make_trial_segments(rng)
        pairs = np.array([[0, 1], [0, 2], [2, 3]])
        peaks = run_surrogates(
            segs,
            n_trials=8,
            n_units=4,
            n_surr=20,
            pairs=pairs,
            rng=rng,
            bin_size=0.001,
            max_lag=0.02,
            normalize="corrcoef",
        )
        self.assertEqual(peaks.shape, (20, 3))
        self.assertEqual(peaks.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(peaks)))

    def test_two_stage_mc_detects_strong_pair(self) -> None:
        """A pair with an implausibly high peak becomes significant."""
        rng = np.random.default_rng(1)
        segs, _ = _make_trial_segments(rng)
        n_units = 4
        cross_pairs = np.array([[0, 1], [0, 2], [1, 2], [2, 3]])
        *_, prefilter_mask, prefilter_pairs = build_shape_prefilter(
            n_units=n_units,
            cross_pairs=cross_pairs,
            peak_lag=np.zeros((n_units, n_units)),
            peak_fwhm=np.zeros((n_units, n_units)),
            fwhm_range_s=None,
            latency_range_s=None,
        )
        # Observed statistic: pair (0,1) far above any surrogate peak,
        # the rest well below.
        raw_peak_val = np.full((n_units, n_units), -1.0)
        raw_peak_val[0, 1] = 10.0
        p_values, surr_threshold, n_cand, n_surr_total, cand_pairs = (
            run_two_stage_mc(
                trial_segs=segs,
                n_trials=8,
                n_units=n_units,
                raw_peak_val=raw_peak_val,
                prefilter_pairs=prefilter_pairs,
                prefilter_mask=prefilter_mask,
                n_screen=50,
                n_total=150,
                bin_size=0.001,
                max_lag=0.02,
                normalize="corrcoef",
                rng=rng,
                fdr_alpha=0.05,
            )
        )
        self.assertEqual(p_values.shape, (n_units, n_units))
        self.assertEqual(surr_threshold.shape, (n_units, n_units))
        # The strong pair is a refined candidate with a tiny p-value.
        self.assertGreaterEqual(n_cand, 1)
        self.assertIn([0, 1], cand_pairs.tolist())
        self.assertLess(p_values[0, 1], 0.05)
        self.assertEqual(n_surr_total, 150)
        # A weak prefilter pair stays non-significant.
        self.assertGreater(p_values[2, 3], 0.5)
        # Cells outside the prefilter pair set are held at the fill (1.0).
        self.assertEqual(p_values[0, 3], 1.0)
        self.assertEqual(p_values[3, 0], 1.0)


@unittest.skipUnless(HAS_NUMBA, "requires the numba optional extra")
class LegacyDenominatorTest(unittest.TestCase):
    """The divisor run_surrogates lifts out of the surrogate loop."""

    def setUp(self) -> None:
        """Segments on a common support, plus a pair list."""
        self.segs, _ = _make_trial_segments(np.random.default_rng(3))
        self.pairs = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)
        self.kw = dict(bin_size=0.001, max_lag=0.02, pairs=self.pairs)

    def test_matches_the_transform_it_replaces(self) -> None:
        """Dividing after the reduce equals normalizing before it."""
        counts = compute_ccg_counts(
            self.segs, np.arange(8), with_auto=True, **self.kw
        )
        normalized = legacy_auto_normalized(counts)
        denom = legacy_auto_denominator(self.segs, self.pairs, 0.001)
        excess = counts.counts - counts.expected_array()
        np.testing.assert_allclose(
            np.max(normalized, axis=-1), np.max(excess, axis=-1) / denom
        )

    def test_invariant_under_the_pairing(self) -> None:
        """Every derangement gives the same divisor, so it can be hoisted.

        This is what lets surrogate_null work in count space and still
        produce draws comparable to a legacy-scaled observed peak.
        """
        denom = legacy_auto_denominator(self.segs, self.pairs, 0.001)
        rng = np.random.default_rng(11)
        for pairing in derangements(8, 5, rng):
            counts = compute_ccg_counts(
                self.segs, pairing, with_auto=True, **self.kw
            )
            a_i = counts.auto_direct[self.pairs[:, 0]]
            a_j = counts.auto_paired[self.pairs[:, 1]]
            np.testing.assert_allclose(np.sqrt(a_i * a_j), denom, rtol=1e-12)

    def test_requires_a_common_support(self) -> None:
        """Off a common support the divisor would move with the pairing."""
        rng = np.random.default_rng(4)
        starts = np.arange(8) * 2.0
        stops = starts + 1.0
        stops[0] -= 0.3  # one short trial is enough
        spikes = [np.sort(rng.uniform(0, 16.0, 400)) for _ in range(4)]
        ragged = clip_spikes_to_trials(
            spikes, np.column_stack([starts, stops]), align_times=starts
        )
        with self.assertRaisesRegex(ValueError, "common support"):
            legacy_auto_denominator(ragged, self.pairs, 0.001)


@unittest.skipUnless(HAS_NUMBA, "requires the numba optional extra")
class SurrogateDriverTest(unittest.TestCase):
    """run_surrogates against the surrogate_null it is built on."""

    def test_normalize_none_is_raw_counts(self) -> None:
        """With no normalization the peak is the raw count peak."""
        segs, _ = _make_trial_segments(np.random.default_rng(5))
        pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
        peaks = run_surrogates(
            segs,
            n_trials=8,
            n_units=4,
            n_surr=4,
            pairs=pairs,
            rng=np.random.default_rng(6),
            bin_size=0.001,
            max_lag=0.02,
            normalize="none",
        )
        self.assertTrue(np.all(peaks == np.floor(peaks)))
        self.assertTrue(np.all(peaks >= 0))

    def test_agrees_with_surrogate_null(self) -> None:
        """The driver is the count-space null, rescaled once."""
        segs, _ = _make_trial_segments(np.random.default_rng(7))
        pairs = np.array([[0, 1], [0, 3], [1, 2]], dtype=np.int64)
        common = dict(bin_size=0.001, max_lag=0.02, pairs=pairs)
        peaks = run_surrogates(
            segs,
            n_trials=8,
            n_units=4,
            n_surr=6,
            rng=np.random.default_rng(8),
            normalize="legacy_auto_normalized",
            **common,
        )
        test = surrogate_null(
            segs,
            TrialShuffle(8),
            6,
            np.random.default_rng(8),
            reduce=lambda v, _c: np.max(v, axis=-1),
            **common,
        )
        denom = legacy_auto_denominator(segs, pairs, 0.001)
        np.testing.assert_allclose(
            peaks, (test.draws / denom).astype(np.float32), rtol=1e-6
        )

    def test_matches_a_per_pairing_ccg_trial_paired_loop(self) -> None:
        """The whole driver, against the independent trial-paired path.

        ccg_trial_paired normalizes each pairing itself, which is what
        the driver no longer does; agreeing with it end to end is what
        says hoisting the divisor out of the loop changed nothing.
        """
        segs, _ = _make_trial_segments(np.random.default_rng(12))
        pairs = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.int64)
        common = dict(bin_size=0.001, max_lag=0.02, pairs=pairs)
        peaks = run_surrogates(
            segs,
            n_trials=8,
            n_units=4,
            n_surr=5,
            rng=np.random.default_rng(13),
            normalize="legacy_auto_normalized",
            **common,
        )
        # Same seed, same scheme, so the same five pairings.
        reference = np.stack(
            [
                np.max(ccg_trial_paired(segs, p, **common)[1], axis=-1)
                for p in TrialShuffle(8).draw(5, np.random.default_rng(13))
            ]
        )
        np.testing.assert_allclose(peaks, reference, rtol=1e-6)

    def test_rejects_a_session_wide_normalization(self) -> None:
        """Modes defined against one global duration are not available."""
        segs, _ = _make_trial_segments(np.random.default_rng(9))
        with self.assertRaisesRegex(ValueError, "run_surrogates supports"):
            run_surrogates(
                segs,
                n_trials=8,
                n_units=4,
                n_surr=2,
                pairs=np.array([[0, 1]]),
                rng=np.random.default_rng(0),
                bin_size=0.001,
                max_lag=0.02,
                normalize="rate",
            )


if __name__ == "__main__":
    unittest.main()
