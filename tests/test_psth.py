"""Tests for automatic PSTH bin-width selection."""

import unittest

import numpy as np

from aind_ephys_utils.ops.bin import select_psth_bin_width


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
