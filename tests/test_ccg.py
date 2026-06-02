"""Tests for the cross-correlogram (ccg) module."""

import unittest

import numpy as np

from aind_ephys_utils.metrics.ccg import (
    ccg_allpairs_sparse,
    ccg_between_sets_sparse,
    rescale_ccgs,
    smooth_ccgs,
)


class CCGTest(unittest.TestCase):
    """Tests for sparse CCG computation and post-processing."""

    def test_between_sets_lag_convention(self) -> None:
        """C[i, j] histograms t_j - t_i, so a +delta lag is positive."""
        delta = 0.005
        a = np.arange(0.1, 1.0, 0.1)
        b = a + delta
        lags, C = ccg_between_sets_sparse(
            [a],
            [b],
            bin_size=0.001,
            max_lag=0.05,
            observation_window=(0.0, 1.0),
        )
        self.assertEqual(C.shape, (1, 1, lags.size))
        peak_lag = lags[np.argmax(C[0, 0])]
        # set2 fires delta AFTER set1 -> peak at positive lag = +delta.
        self.assertLess(abs(peak_lag - delta), 0.001 + 1e-9)
        # Every one of the 9 source spikes contributes exactly one pair.
        self.assertEqual(int(C[0, 0].max()), a.size)

    def test_allpairs_symmetry(self) -> None:
        """C[i, j](tau) must equal C[j, i](-tau)."""
        rng = np.random.default_rng(0)
        u0 = np.sort(rng.uniform(0, 10, size=200))
        u1 = np.sort(rng.uniform(0, 10, size=150))
        lags, C = ccg_allpairs_sparse(
            [u0, u1],
            bin_size=0.001,
            max_lag=0.02,
            observation_window=(0.0, 10.0),
        )
        self.assertEqual(C.shape, (2, 2, lags.size))
        np.testing.assert_array_almost_equal(C[0, 1], C[1, 0][::-1])

    def test_allpairs_autocorr_zero_lag_excluded(self) -> None:
        """Zero-lag autocorr bin is blanked when requested."""
        rng = np.random.default_rng(1)
        u0 = np.sort(rng.uniform(0, 10, size=300))
        lags, C = ccg_allpairs_sparse(
            [u0],
            bin_size=0.001,
            max_lag=0.02,
            exclude_zero_lag_autocorr=True,
            observation_window=(0.0, 10.0),
        )
        center = lags.size // 2
        self.assertEqual(C[0, 0, center], 0.0)

    def test_corrcoef_normalization_is_finite(self) -> None:
        """corrcoef normalization returns finite correlation values."""
        rng = np.random.default_rng(2)
        u0 = np.sort(rng.uniform(0, 10, size=200))
        u1 = np.sort(rng.uniform(0, 10, size=200))
        _, C = ccg_allpairs_sparse(
            [u0, u1],
            bin_size=0.001,
            max_lag=0.02,
            normalize="corrcoef",
            observation_window=(0.0, 10.0),
        )
        self.assertTrue(np.all(np.isfinite(C)))

    def test_unsorted_spikes_raise(self) -> None:
        """Unsorted spike trains are rejected."""
        with self.assertRaises(ValueError):
            ccg_allpairs_sparse(
                [np.array([0.3, 0.1, 0.2])],
                observation_window=(0.0, 1.0),
            )

    def test_smooth_and_rescale(self) -> None:
        """Smoothing preserves shape; rescaling maps into [0, 1]."""
        rng = np.random.default_rng(3)
        C = rng.uniform(0, 5, size=(2, 2, 41))
        smoothed = smooth_ccgs(C, bin_size=0.001, kernel_width=0.003)
        self.assertEqual(smoothed.shape, C.shape)
        scaled = rescale_ccgs(C, axis=-1)
        self.assertTrue(np.all(scaled >= -1e-9))
        self.assertTrue(np.all(scaled <= 1 + 1e-9))


if __name__ == "__main__":
    unittest.main()
