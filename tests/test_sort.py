"""Tests spike metrics."""

import unittest

import numpy as np

from aind_ephys_utils.align import to_events
from aind_ephys_utils.sort import by_condition, sort_by_condition


class SortingTest(unittest.TestCase):
    """Tests DataFrame sorting methods."""

    offset = 0.002
    events = np.arange(10)  # 10 events
    times = events + offset
    labels = np.flip(np.arange(1, 11))

    df = to_events(
        times,
        events,
        interval=(-1, 1),
        event_labels=labels,
        return_dataframe=True,
    )

    def test_by_condition(self) -> None:
        """Test the `by_condition` method."""

        sorted_df = by_condition(self.df)

        self.assertTrue(np.all(np.diff(sorted_df["event_label"].values) >= 0))
        self.assertTrue(np.all(np.diff(sorted_df["event_index"].values) >= 0))

    def test_sort_by_condition(self) -> None:
        """Test the `sort_by_condition` alias."""

        sorted_df = sort_by_condition(self.df)

        self.assertTrue(np.all(np.diff(sorted_df["event_label"].values) >= 0))
        self.assertTrue(np.all(np.diff(sorted_df["event_index"].values) >= 0))


if __name__ == "__main__":
    """Run the tests"""
    unittest.main()
