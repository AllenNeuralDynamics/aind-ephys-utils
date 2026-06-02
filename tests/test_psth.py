"""Tests for the PSTH and bin-width-selection module."""

import unittest

import numpy as np

from aind_ephys_utils.spiketrain.psth import (
    select_psth_bin_width,
    sliding_window_psth,
)


class SlidingWindowPSTHTest(unittest.TestCase):
    """Tests for sliding_window_psth."""

    def test_rate_recovery_per_trial_relative(self) -> None:
        """PSTH (spikes/s) recovers a flat Poisson rate."""
        rng = np.random.default_rng(0)
        rate = 20.0
        trials = []
        for _ in range(400):
            n = rng.poisson(rate * 1.0)
            trials.append(np.sort(rng.uniform(0.0, 1.0, size=n)))
        time_bins, psth, n_eff = sliding_window_psth(
            trials,
            window_size=0.1,
            bin_size=0.01,
            time_range=(0.0, 1.0),
        )
        self.assertEqual(time_bins.shape, psth.shape)
        self.assertEqual(time_bins.shape, n_eff.shape)
        central = (time_bins > 0.2) & (time_bins < 0.8)
        self.assertAlmostEqual(float(psth[central].mean()), rate, delta=2.0)

    def test_absolute_mode_matches_relative(self) -> None:
        """Absolute spikes + trial_starts agree with relative arrays."""
        rng = np.random.default_rng(1)
        starts = np.arange(0, 50) * 10.0
        rel = [np.sort(rng.uniform(0.0, 1.0, size=30)) for _ in starts]
        abs_spikes = np.sort(
            np.concatenate([r + s for r, s in zip(rel, starts)])
        )
        tb_a, psth_a, _ = sliding_window_psth(
            abs_spikes,
            starts,
            window_size=0.1,
            bin_size=0.01,
            time_range=(0.0, 1.0),
        )
        tb_r, psth_r, _ = sliding_window_psth(
            rel,
            window_size=0.1,
            bin_size=0.01,
            time_range=(0.0, 1.0),
        )
        np.testing.assert_array_almost_equal(tb_a, tb_r)
        np.testing.assert_array_almost_equal(psth_a, psth_r)

    def test_invalid_time_range_raises(self) -> None:
        """A non-increasing time_range is rejected."""
        with self.assertRaises(ValueError):
            sliding_window_psth([np.array([0.1])], time_range=(1.0, 0.0))


class SelectBinWidthTest(unittest.TestCase):
    """Tests for Shimazaki-Shinomoto bin-width selection."""

    def test_structural_outputs(self) -> None:
        """Selected width is the argmin over the candidate grid."""
        rng = np.random.default_rng(2)
        trials = [np.sort(rng.uniform(0, 1.0, size=50)) for _ in range(60)]
        res = select_psth_bin_width(
            trials,
            t_start=0.0,
            t_stop=1.0,
            base_bin_width_s=0.005,
            refine=False,
        )
        self.assertIn(res.best_width_s, res.candidate_widths_s)
        self.assertAlmostEqual(res.best_cost, float(res.candidate_costs.min()))
        self.assertAlmostEqual(res.best_width_s, res.best_multiplier * 0.005)

    def test_flat_process_prefers_coarse_bins(self) -> None:
        """A featureless rate should not select the finest bin width."""
        rng = np.random.default_rng(3)
        trials = [np.sort(rng.uniform(0, 1.0, size=40)) for _ in range(80)]
        res = select_psth_bin_width(
            trials,
            t_start=0.0,
            t_stop=1.0,
            base_bin_width_s=0.005,
            refine=False,
        )
        self.assertGreater(res.best_width_s, 0.005)


if __name__ == "__main__":
    unittest.main()
