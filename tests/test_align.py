"""Tests spike alignment methods."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from aind_ephys_utils.align import align_to_events, to_events


class AlignSpikesTest(unittest.TestCase):
    """Tests spike alignment methods."""

    events = np.arange(10)  # 10 events
    labels = np.arange(10) + 1  # event labels
    times = events
    unit_ids = np.arange(10)  # 10 units

    def test_align_ndarray(self) -> None:
        """Test the `align` method with ndarray as input"""

        ts, inds, units = to_events(self.times, self.events, (-0.1, 0.1))

        assert_array_equal(ts, np.zeros(self.times.shape))
        assert_array_equal(inds, np.arange(10, dtype="int"))
        assert_array_equal(units, np.zeros(self.times.shape))

        df = to_events(self.times, self.events, (-0.1, 0.1), return_df=True)

        assert_array_equal(df.time, np.zeros(self.times.shape))
        assert_array_equal(df.event_index, np.arange(10, dtype="int"))
        assert_array_equal(df.unit_id, np.zeros(self.times.shape))

        df = to_events(
            self.times,
            self.events,
            (-0.1, 0.1),
            event_labels=self.labels,
            return_df=True,
        )

        assert_array_equal(df.time, np.zeros(self.times.shape))
        assert_array_equal(df.event_index, np.arange(10, dtype="int"))
        assert_array_equal(df.unit_id, np.zeros(self.times.shape))

        bins, counts, unit_ids = to_events(
            self.times, self.events, (-0.1, 0.1), bin_size=0.01
        )

        self.assertEqual(np.sum(counts), len(self.events))

        da = to_events(
            self.times,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            return_df=True,
        )

        self.assertEqual(np.sum(da.data), len(self.events))

        da = to_events(
            self.times,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            event_labels=self.labels,
            return_df=True,
        )

        self.assertEqual(np.sum(da.data), len(self.events))

        with self.assertRaises(Exception) as context:
            to_events(
                self.times,
                self.events,
                (-0.1, 0.1),
                bin_size=0.01,
                event_labels=self.labels[:5],
                return_df=True,
            )

        self.assertTrue(
            "events and event_labels must be the same length."
            in str(context.exception)
        )

    def test_align_list(self) -> None:
        """Test the `align` method with a list of times as input"""

        times_as_list = [self.times for unit in self.unit_ids]

        ts, inds, units = to_events(times_as_list, self.events, (-0.1, 0.1))

        assert_array_equal(
            ts,
            np.concatenate(
                [np.zeros(self.times.shape) for i in self.unit_ids]
            ),
        )
        assert_array_equal(
            units, np.concatenate([self.unit_ids for i in self.events])
        )
        assert_array_equal(
            inds,
            np.concatenate(
                [np.zeros(self.unit_ids.shape) + i for i in self.events]
            ),
        )

        df = to_events(times_as_list, self.events, (-0.1, 0.1), return_df=True)

        assert_array_equal(
            df.time,
            np.concatenate(
                [np.zeros(self.times.shape) for i in self.unit_ids]
            ),
        )
        assert_array_equal(
            df.unit_id, np.concatenate([self.unit_ids for i in self.events])
        )
        assert_array_equal(
            df.event_index,
            np.concatenate(
                [np.zeros(self.unit_ids.shape) + i for i in self.events]
            ),
        )

        bins, counts, unit_ids = to_events(
            times_as_list, self.events, (-0.1, 0.1), bin_size=0.01
        )

        self.assertEqual(np.sum(counts), len(self.events) * len(self.unit_ids))

        da = to_events(
            times_as_list,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            return_df=True,
        )

        self.assertEqual(
            np.sum(da.data), len(self.events) * len(self.unit_ids)
        )

    def test_align_dict(self) -> None:
        """Test the `align` method with a dict of spike times as input"""

        times_as_dict = {}

        for i in self.unit_ids:
            times_as_dict[i] = self.times

        ts, inds, units = to_events(times_as_dict, self.events, (-0.1, 0.1))

        assert_array_equal(
            ts,
            np.concatenate(
                [np.zeros(self.times.shape) for i in self.unit_ids]
            ),
        )
        assert_array_equal(
            units, np.concatenate([self.unit_ids for i in self.events])
        )
        assert_array_equal(
            inds,
            np.concatenate(
                [np.zeros(self.unit_ids.shape) + i for i in self.events]
            ),
        )

        df = to_events(times_as_dict, self.events, (-0.1, 0.1), return_df=True)

        assert_array_equal(
            df.time,
            np.concatenate(
                [np.zeros(self.times.shape) for i in self.unit_ids]
            ),
        )
        assert_array_equal(
            df.unit_id, np.concatenate([self.unit_ids for i in self.events])
        )
        assert_array_equal(
            df.event_index,
            np.concatenate(
                [np.zeros(self.unit_ids.shape) + i for i in self.events]
            ),
        )

        bins, counts, unit_ids = to_events(
            times_as_dict, self.events, (-0.1, 0.1), bin_size=0.01
        )

        self.assertEqual(np.sum(counts), len(self.events) * len(self.unit_ids))

        da = to_events(
            times_as_dict,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            return_df=True,
        )

        self.assertEqual(
            np.sum(da.data), len(self.events) * len(self.unit_ids)
        )

    def test_align_df(self) -> None:
        """Test the `align` method with a DataFrame of spike times as input"""

        times_as_list = [self.times for unit in self.unit_ids]
        times_as_df = pd.DataFrame(
            index=self.unit_ids, data={"spike_times": times_as_list}
        )

        ts, inds, units = to_events(times_as_df, self.events, (-0.1, 0.1))

        assert_array_equal(
            ts,
            np.concatenate(
                [np.zeros(self.times.shape) for i in self.unit_ids]
            ),
        )
        assert_array_equal(
            units, np.concatenate([self.unit_ids for i in self.events])
        )
        assert_array_equal(
            inds,
            np.concatenate(
                [np.zeros(self.unit_ids.shape) + i for i in self.events]
            ),
        )

        df = to_events(times_as_df, self.events, (-0.1, 0.1), return_df=True)

        assert_array_equal(
            df.time,
            np.concatenate(
                [np.zeros(self.times.shape) for i in self.unit_ids]
            ),
        )
        assert_array_equal(
            df.unit_id, np.concatenate([self.unit_ids for i in self.events])
        )
        assert_array_equal(
            df.event_index,
            np.concatenate(
                [np.zeros(self.unit_ids.shape) + i for i in self.events]
            ),
        )

        bins, counts, unit_ids = to_events(
            times_as_df, self.events, (-0.1, 0.1), bin_size=0.01
        )

        self.assertEqual(np.sum(counts), len(self.events) * len(self.unit_ids))

        da = to_events(
            times_as_df,
            self.events,
            (-0.1, 0.1),
            bin_size=0.01,
            return_df=True,
        )

        self.assertEqual(
            np.sum(da.data), len(self.events) * len(self.unit_ids)
        )

    def test_align_to_events(self) -> None:
        """Test the `align_to_events` alias"""

        ts, inds, unit_ids = align_to_events(
            self.times, self.events, (-0.1, 0.1)
        )

        assert_array_equal(ts, np.zeros(self.times.shape))
        assert_array_equal(inds, np.arange(10, dtype="int"))


if __name__ == "__main__":
    """Run the tests"""
    unittest.main()
