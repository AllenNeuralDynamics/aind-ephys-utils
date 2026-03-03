"""Tests for plots helpers."""

import unittest

import matplotlib
import numpy as np
import xarray as xr

from aind_ephys_utils.plots.annotations import add_scale_bars, annotate_events
from aind_ephys_utils.plots.colors import get_color_for_region
from aind_ephys_utils.plots.raster import raster


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

    def test_raster_single_unit_many_trials_uses_full_y_range(self) -> None:
        """Single-unit rasters should span all trials on the y-axis."""
        n_trials = 200
        data = np.empty((n_trials, 1), dtype=object)
        for i in range(n_trials):
            data[i, 0] = np.array([0.1, 0.2], dtype=float)
        spikes = xr.DataArray(
            data,
            dims=("trial", "unit"),
            coords={"trial": np.arange(n_trials), "unit": [0]},
        )
        ax = raster(spikes)
        y0, y1 = ax.get_ylim()
        self.assertLessEqual(y0, -0.5)
        self.assertGreaterEqual(y1, n_trials - 0.5)
        # Should not collapse to a single center tick for unit labels.
        self.assertGreater(len(ax.get_yticks()), 1)

    def test_raster_accepts_markersize_and_alpha(self) -> None:
        """Raster should forward markersize and alpha to scatter artists."""
        n_trials = 5
        data = np.empty((n_trials, 1), dtype=object)
        for i in range(n_trials):
            data[i, 0] = np.array([0.1, 0.2], dtype=float)
        spikes = xr.DataArray(
            data,
            dims=("trial", "unit"),
            coords={"trial": np.arange(n_trials), "unit": [0]},
        )
        ax = raster(spikes, markersize=5.0, alpha=0.2)
        self.assertGreaterEqual(len(ax.collections), 1)
        sizes = ax.collections[0].get_sizes()
        self.assertTrue(np.allclose(sizes, 5.0))
        self.assertAlmostEqual(float(ax.collections[0].get_alpha()), 0.2)
