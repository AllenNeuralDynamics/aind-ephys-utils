"""Tests for the Ripley's K/L point-process module."""

import unittest

import numpy as np

from aind_ephys_utils.metrics.ripley import (
    compare_ripley_l,
    ripley_k,
    ripley_k_envelope,
)


class RipleyTest(unittest.TestCase):
    """Tests for Ripley's K/L estimation on 1D point processes."""

    def test_poisson_k_matches_theory(self) -> None:
        """For homogeneous Poisson in 1D, K(r) ~= 2r and L(r) ~= 0."""
        rng = np.random.default_rng(0)
        T = 1.0
        rate = 300.0
        trials = []
        for _ in range(40):
            n = rng.poisson(rate * T)
            trials.append(np.sort(rng.uniform(0, T, size=n)))
        radii = np.linspace(0, 0.2, 21)
        res = ripley_k(trials, radii=radii, window_lengths=T)
        # K(r) ~= 2r for Poisson.
        idx = 10  # r = 0.1
        self.assertAlmostEqual(res.K[idx], 2 * radii[idx], delta=0.04)
        # L centering: L == K/2 - r exactly.
        np.testing.assert_array_almost_equal(res.L, res.K / 2.0 - res.r)
        # Centered L stays close to zero for CSR.
        self.assertLess(np.max(np.abs(res.L)), 0.02)

    def test_regular_process_is_underdispersed(self) -> None:
        """A regular lattice has no pairs below its spacing -> L < 0."""
        spacing = 0.05
        pts = np.arange(spacing, 1.0, spacing)
        radii = np.linspace(0, 0.04, 9)
        res = ripley_k([pts], radii=radii, window_lengths=1.0)
        # Below the lattice spacing there are no neighbour pairs.
        self.assertEqual(res.K[-1], 0.0)
        # So centered L is negative (regular / under-dispersed).
        self.assertTrue(np.all(res.L[1:] < 0))

    def test_envelope_shapes_and_bounds(self) -> None:
        """Envelope arrays line up and lo <= hi everywhere."""
        rng = np.random.default_rng(1)
        trials = [np.sort(rng.uniform(0, 1.0, size=120)) for _ in range(8)]
        radii = np.linspace(0, 0.2, 11)
        res = ripley_k_envelope(
            trials,
            radii=radii,
            window_lengths=1.0,
            n_simulations=49,
            rng=np.random.default_rng(2),
        )
        self.assertEqual(res.K.shape, radii.shape)
        self.assertEqual(res.K_lo.shape, radii.shape)
        self.assertTrue(np.all(res.K_lo <= res.K_hi + 1e-9))
        self.assertTrue(np.all(res.L_lo <= res.L_hi + 1e-9))
        self.assertEqual(res.n_simulations, 49)

    def test_compare_ripley_l_returns_named_results(self) -> None:
        """compare_ripley_l keys its output by process name."""
        rng = np.random.default_rng(3)
        processes = {
            "a": [np.sort(rng.uniform(0, 1.0, size=80)) for _ in range(5)],
            "b": [np.sort(rng.uniform(0, 1.0, size=80)) for _ in range(5)],
        }
        out = compare_ripley_l(
            processes,
            radii=np.linspace(0, 0.2, 11),
            window_lengths=1.0,
            n_simulations=19,
            rng=np.random.default_rng(4),
        )
        self.assertEqual(set(out), {"a", "b"})
        self.assertEqual(out["a"].r.size, 11)


if __name__ == "__main__":
    unittest.main()
