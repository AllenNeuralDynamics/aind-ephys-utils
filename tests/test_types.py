"""Tests for the RaggedSpikes and BinnedSpikes dataclass types."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from aind_ephys_utils.types import BinnedSpikes, RaggedSpikes


# ---- RaggedSpikes ----


class RaggedFromArraysTest(unittest.TestCase):
    def test_with_alignment(self):
        spike_times = [
            np.array([0.5, 1.5, 2.5, 3.5]),  # unit 0
            np.array([0.3, 1.3, 2.3, 3.3]),  # unit 1
        ]
        anchors = np.array([1.0, 2.0, 3.0])
        rs = RaggedSpikes.from_arrays(spike_times, anchors, window=(-0.6, 0.6))
        self.assertEqual(rs.n_trials, 3)
        self.assertEqual(rs.n_units, 2)
        self.assertEqual(rs.time_window, (-0.6, 0.6))
        # Trial 0, unit 0: spikes at 0.5 within [0.4, 1.6] → -0.5, 0.5
        assert_allclose(rs.spike_times[0][0], [-0.5, 0.5])

    def test_no_alignment(self):
        spike_times = [np.array([1.0, 2.0]), np.array([3.0])]
        rs = RaggedSpikes.from_arrays(spike_times)
        self.assertEqual(rs.n_trials, 1)
        self.assertEqual(rs.n_units, 2)
        self.assertIsNone(rs.time_window)

    def test_from_dict(self):
        spike_dict = {
            "V1_01": np.array([0.1, 0.5]),
            "M1_02": np.array([0.3, 0.7]),
        }
        rs = RaggedSpikes.from_arrays(spike_dict)
        self.assertEqual(rs.n_units, 2)
        assert_array_equal(rs.unit_meta["unit_id"], ["V1_01", "M1_02"])

    def test_missing_window_raises(self):
        with self.assertRaises(ValueError):
            RaggedSpikes.from_arrays([np.array([1.0])], anchor_times=np.array([0.0]))


# ---- BinnedSpikes ----


class BinnedFromArraysTest(unittest.TestCase):
    def test_canonical_order(self):
        data = np.zeros((3, 2, 10))
        time = np.linspace(0, 0.9, 10)
        bs = BinnedSpikes.from_arrays(data, time)
        self.assertEqual(bs.n_trials, 3)
        self.assertEqual(bs.n_units, 2)
        self.assertEqual(bs.n_time, 10)

    def test_axes_transpose(self):
        # Data in (unit, time, trial) order
        data = np.arange(24, dtype=float).reshape(2, 4, 3)
        time = np.array([0.0, 0.1, 0.2, 0.3])
        bs = BinnedSpikes.from_arrays(data, time, axes=("unit", "time", "trial"))
        # Should be transposed to (trial, unit, time) = (3, 2, 4)
        self.assertEqual(bs.data.shape, (3, 2, 4))
        assert_array_equal(bs.time, time)

    def test_bad_axes_raises(self):
        with self.assertRaises(ValueError):
            BinnedSpikes.from_arrays(
                np.zeros((3, 2, 10)),
                np.zeros(10),
                axes=("x", "y", "z"),
            )


if __name__ == "__main__":
    unittest.main()
