"""Tests for dataframe adapters."""

import unittest

import numpy as np
import pandas as pd

from aind_ephys_utils.adapters import from_dataframe

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


class AdapterTest(unittest.TestCase):
    """Ensure adapter ops path works end-to-end."""

    def test_from_dataframe_bin(self) -> None:
        """Use ops-based align/bin path for dataframe ingestion."""
        units_df = pd.DataFrame(
            {
                "unit_id": [0],
                "spike_times": [np.array([0.05, 0.15, 0.35], dtype=float)],
            }
        )
        trials_df = pd.DataFrame(
            {
                "trial_id": [0, 1],
                "trial_start": [0.0, 0.3],
                "trial_end": [0.2, 0.5],
            }
        )

        da = from_dataframe(
            units_df,
            trials_df,
            unit_id_col="unit_id",
            trial_id_col="trial_id",
            spike_times_col="spike_times",
            trial_start_col="trial_start",
            trial_end_col="trial_end",
            bin_size=0.1,
            window=(-0.1, 0.2),
            time_unit="s",
        )

        self.assertEqual(da.dims, ("trial", "unit", "time"))
        self.assertEqual(da.shape, (2, 1, 3))
        np.testing.assert_allclose(da.values[0, 0], [0.0, 10.0, 10.0])
        np.testing.assert_allclose(da.values[1, 0], [0.0, 10.0, 0.0])

    def test_from_dataframe_units_only(self) -> None:
        """Create ragged spikes from a units-only DataFrame."""
        units_df = pd.DataFrame(
            {
                "unit_id": [0, 1],
                "spike_times": [
                    np.array([0.1, 0.2], dtype=float),
                    np.array([0.05], dtype=float),
                ],
            }
        )
        da = from_dataframe(units_df, unit_id_col="unit_id")
        self.assertEqual(da.dims, ("unit", "trial"))
        self.assertEqual(da.sizes["trial"], 1)
        self.assertEqual(da.dtype, object)

    def test_from_dataframe_events_only(self) -> None:
        """Build events DataArray from a trials-only DataFrame."""
        trials_df = pd.DataFrame(
            {
                "trial_id": [0, 1],
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
                "go_cue_time": [0.1, 1.1],
            }
        )
        events = from_dataframe(trials_df, trial_id_col="trial_id")
        self.assertEqual(events.dims, ("trial", "event", "bound"))
        self.assertIn("go_cue", list(events["event"].values))

    def test_from_dataframe_events_wide_format(self) -> None:
        """Build events DataArray from wide-format trials DataFrame."""
        trials_df = pd.DataFrame(
            {
                "trial_id": [0, 1],
                "go_cue_time": [0.1, 1.1],
                "delay_start": [0.15, 1.15],
                "delay_end": [0.4, 1.4],
            }
        )
        events = from_dataframe(trials_df, trial_id_col="trial_id")

        # Check dimensions
        self.assertEqual(events.dims, ("trial", "event", "bound"))
        self.assertEqual(
            events.shape, (2, 2, 2)
        )  # 2 trials, 2 events, 2 bounds

        # Check event names
        event_names = list(events["event"].values)
        self.assertIn("go_cue", event_names)
        self.assertIn("delay", event_names)

        # Check instantaneous event (go_cue)
        go_cue_idx = event_names.index("go_cue")
        np.testing.assert_allclose(events.values[:, go_cue_idx, 0], [0.1, 1.1])
        np.testing.assert_allclose(events.values[:, go_cue_idx, 1], [0.1, 1.1])

        # Check epoch (delay)
        delay_idx = event_names.index("delay")
        np.testing.assert_allclose(
            events.values[:, delay_idx, 0], [0.15, 1.15]
        )
        np.testing.assert_allclose(events.values[:, delay_idx, 1], [0.4, 1.4])

    def test_from_dataframe_events_long_format(self) -> None:
        """Build events DataArray from long-format trials DataFrame.

        Long format groups multiple event rows by trial_id.
        """
        trials_df = pd.DataFrame(
            {
                "trial_id": [0, 0, 1, 1],
                "event_name": ["go_cue", "delay", "go_cue", "delay"],
                "event_time": [0.1, 0.15, 1.1, 1.15],
                "event_end": [0.1, 0.4, 1.1, 1.4],
            }
        )
        events = from_dataframe(
            trials_df,
            trial_id_col="trial_id",
            long_event_col="event_name",
            long_time_col="event_time",
            long_end_time_col="event_end",
        )

        # Check dimensions
        self.assertEqual(events.dims, ("trial", "event", "bound"))
        # 2 unique trials, 2 events, 2 bounds
        self.assertEqual(events.shape, (2, 2, 2))

        # Check event names preserved in order
        event_names = list(events["event"].values)
        self.assertIn("go_cue", event_names)
        self.assertIn("delay", event_names)

        # Check instantaneous event (go_cue)
        go_cue_idx = event_names.index("go_cue")
        np.testing.assert_allclose(events.values[:, go_cue_idx, 0], [0.1, 1.1])
        np.testing.assert_allclose(events.values[:, go_cue_idx, 1], [0.1, 1.1])

        # Check epoch (delay)
        delay_idx = event_names.index("delay")
        np.testing.assert_allclose(
            events.values[:, delay_idx, 0], [0.15, 1.15]
        )
        np.testing.assert_allclose(events.values[:, delay_idx, 1], [0.4, 1.4])

    def test_from_dataframe_events_long_vs_wide_equivalence(self) -> None:
        """Verify long and wide format produce equivalent results."""
        # Wide format
        wide_df = pd.DataFrame(
            {
                "trial_id": [0, 1],
                "go_cue_time": [0.1, 1.1],
                "delay_start": [0.15, 1.15],
                "delay_end": [0.4, 1.4],
            }
        )

        # Long format
        long_df = pd.DataFrame(
            {
                "trial_id": [0, 0, 1, 1],
                "event_name": ["go_cue", "delay", "go_cue", "delay"],
                "event_time": [0.1, 0.15, 1.1, 1.15],
                "event_end": [0.1, 0.4, 1.1, 1.4],
            }
        )

        wide_events = from_dataframe(wide_df, trial_id_col="trial_id")
        long_events = from_dataframe(
            long_df,
            trial_id_col="trial_id",
            long_event_col="event_name",
            long_time_col="event_time",
            long_end_time_col="event_end",
        )

        # Check shapes match
        self.assertEqual(wide_events.shape, long_events.shape)

        # Check event names match (may be in different order)
        wide_event_names = set(wide_events["event"].values)
        long_event_names = set(long_events["event"].values)
        self.assertEqual(wide_event_names, long_event_names)

        # Check values match for each event
        for event_name in wide_event_names:
            wide_vals = wide_events.sel(event=event_name).values
            long_vals = long_events.sel(event=event_name).values
            np.testing.assert_allclose(
                wide_vals,
                long_vals,
                err_msg=f"Mismatch for event {event_name}",
            )


class AdapterPolarsTest(unittest.TestCase):
    """Ensure polars inputs work across adapter paths."""

    def setUp(self) -> None:
        if pl is None:
            self.skipTest("polars is not installed")

    def test_from_dataframe_polars_events_long(self) -> None:
        """Build long-format events from a polars DataFrame."""
        trials_df = pl.DataFrame(
            {
                "trial_id": [0, 0, 1, 1],
                "event_name": ["go_cue", "delay", "go_cue", "delay"],
                "event_time": [0.1, 0.15, 1.1, 1.15],
                "event_end": [0.1, 0.4, 1.1, 1.4],
                "choice": ["L", "L", "R", "R"],
            }
        )
        events = from_dataframe(
            trials_df,
            trial_id_col="trial_id",
            long_event_col="event_name",
            long_time_col="event_time",
            long_end_time_col="event_end",
        )
        self.assertEqual(events.dims, ("trial", "event", "bound"))
        self.assertEqual(events.sizes["trial"], 2)
        self.assertEqual(events.coords["choice"].dims, ("trial",))

    def test_from_dataframe_polars_units_with_trials(self) -> None:
        """Build ragged spikes from polars units+trials DataFrames."""
        units_df = pl.DataFrame(
            {
                "unit_id": [0, 1],
                "spike_times": [[0.05, 0.15, 0.35], [0.02, 0.25]],
                "region": ["VISp", "MOs"],
            }
        )
        trials_df = pl.DataFrame(
            {
                "trial_id": [0, 1],
                "trial_start": [0.0, 0.3],
                "trial_end": [0.2, 0.5],
                "choice": ["L", "R"],
            }
        )
        da = from_dataframe(
            units_df,
            trials_df,
            unit_id_col="unit_id",
            trial_id_col="trial_id",
            spike_times_col="spike_times",
            trial_start_col="trial_start",
            trial_end_col="trial_end",
        )
        self.assertEqual(da.dims, ("trial", "unit"))
        self.assertEqual(da.sizes["trial"], 2)
        self.assertEqual(da.coords["region"].dims, ("unit",))
        self.assertEqual(da.coords["choice"].dims, ("trial",))
