"""Tests for CCG per-pair helpers and the connectivity workflow."""

import unittest

import numpy as np

from aind_ephys_utils.metrics.ccg import (
    NN_to_pair_vec,
    clip_spikes_to_trials,
    derangements,
    pair_vec_to_NN,
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


if __name__ == "__main__":
    unittest.main()
