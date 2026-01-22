"""Tests for plots helpers."""

import unittest

import matplotlib

from aind_ephys_utils.plots.annotations import add_scale_bars, annotate_events
from aind_ephys_utils.plots.colors import get_color_for_region


class PlotHelpersTest(unittest.TestCase):
    """Coverage for plot helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Force non-interactive backend for tests."""
        matplotlib.use("Agg", force=True)

    def test_get_color_for_region_by_id(self) -> None:
        """Look up color by Allen structure id."""
        self.assertEqual(get_color_for_region(997), "#FFFFFF")

    def test_get_color_for_region_by_name(self) -> None:
        """Look up color by Allen structure name."""
        self.assertEqual(get_color_for_region("Cerebrum"), "#B0F0FF")

    def test_get_color_for_region_by_acronym(self) -> None:
        """Look up color by Allen structure acronym."""
        self.assertEqual(get_color_for_region("CH"), "#B0F0FF")

    def test_annotate_events_and_intervals(self) -> None:
        """Annotate events and intervals with labels."""
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111)
        ax.plot([0.0, 1.0], [0.0, 1.0])
        annotate_events(
            ax=ax,
            events=[0.2, 0.8],
            intervals=[(0.1, 0.3)],
            event_labels=["cue", "go"],
            interval_labels=["stim"],
        )
        self.assertGreaterEqual(len(ax.lines), 3)
        self.assertGreaterEqual(len(ax.patches), 1)
        labels = [t.get_text() for t in ax.texts]
        self.assertIn("cue", labels)
        self.assertIn("go", labels)
        self.assertIn("stim", labels)

    def test_add_scale_bars(self) -> None:
        """Hide axes and draw scale bars with custom labels."""
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111)
        ax.plot([0.0, 1.0], [0.0, 1.0])
        add_scale_bars(
            ax=ax,
            x_unit="s",
            y_unit="Hz",
            x_length=0.5,
            y_length=2.0,
            x_text="500 ms",
            y_text="2 Hz",
        )
        self.assertEqual(len(ax.get_xticks()), 0)
        self.assertEqual(len(ax.get_yticks()), 0)
        labels = [t.get_text() for t in ax.texts]
        self.assertIn("500 ms", labels)
        self.assertIn("2 Hz", labels)
