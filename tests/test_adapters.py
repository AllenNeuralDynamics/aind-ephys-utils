"""Tests for dataframe adapters."""

import unittest

import numpy as np
import pandas as pd

from aind_ephys_utils.adapters import from_dataframe


class AdapterTest(unittest.TestCase):
    """Ensure adapter ops path works end-to-end."""

    def test_from_dataframe_bin_uses_ops(self) -> None:
        """Use ops-based align/bin path for dataframe ingestion."""
        units_df = pd.DataFrame(
            {
                "unit_id": [0],
                "spike_times": [
                    np.array([0.05, 0.15, 0.35], dtype=float)
                ],
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

        self.assertEqual(da.dims, ("unit", "trial", "time"))
        self.assertEqual(da.shape, (1, 2, 3))
        np.testing.assert_allclose(da.values[0, 0], [0.0, 10.0, 10.0])
        np.testing.assert_allclose(da.values[0, 1], [0.0, 10.0, 0.0])

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
