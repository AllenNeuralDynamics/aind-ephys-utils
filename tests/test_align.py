"""Tests spike alignment methods."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from aind_ephys_utils import align


class AlignSpikesTest(unittest.TestCase):
    """Tests spike alignment methods."""

    events = np.arange(10)  # 10 events
    times = events
    labels = np.arange(10) + 1  # event labels

    def test_align(self) -> None:
        """Test the `align` method."""

        ts, inds = align.align(self.times, self.events, (-0.1, 0.1))

        assert_array_equal(ts, np.zeros(self.times.shape))
        assert_array_equal(inds, np.arange(10, dtype="int"))

        df = align.align(
            self.times, self.events, (-0.1, 0.1), return_dataframe=True
        )

        assert_array_equal(df.time, np.zeros(self.times.shape))
        assert_array_equal(df.event_index, np.arange(10, dtype="int"))

        df = align.align(
            self.times,
            self.events,
            (-0.1, 0.1),
            event_labels=self.labels,
            return_dataframe=True,
        )

        assert_array_equal(df.time, np.zeros(self.times.shape))
        assert_array_equal(df.event_index, np.arange(10, dtype="int"))

        bins, counts = align.align(
            self.times, self.events, (-0.1, 0.1), bin_size=0.01
        )

        self.assertEqual(np.sum(counts), len(self.events))

        da = align.align(
            self.times,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            return_dataframe=True,
        )

        self.assertEqual(np.sum(da.data), len(self.events))

        da = align.align(
            self.times,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            event_labels=self.labels,
            return_dataframe=True,
        )

        self.assertEqual(np.sum(da.data), len(self.events))

        with self.assertRaises(Exception) as context:
            da = align.align(
                self.times,
                self.events,
                (-0.1, 0.1),
                bin_size=0.01,
                event_labels=self.labels[:5],
                return_dataframe=True,
            )

        self.assertTrue(
            "events and event_labels must be the same length."
            in str(context.exception)
        )

    def test_align_spikes(self) -> None:
        """Test the `align_spikes` alias"""

        ts, inds = align.align_spikes(self.times, self.events, (-0.1, 0.1))

        assert_array_equal(ts, np.zeros(self.times.shape))
        assert_array_equal(inds, np.arange(10, dtype="int"))

    def test_align_times(self) -> None:
        """Test the `align_times` alias"""

        ts, inds = align.align_times(self.times, self.events, (-0.1, 0.1))

        assert_array_equal(ts, np.zeros(self.times.shape))
        assert_array_equal(inds, np.arange(10, dtype="int"))


if __name__ == "__main__":
    """Run the tests"""
    unittest.main()
