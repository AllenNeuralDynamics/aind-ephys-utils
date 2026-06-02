"""Tests for the epochs and spike_utils spike-train utilities."""

import unittest

import numpy as np

from aind_ephys_utils.spiketrain.epochs import (
    add_random_offset_with_wrap,
    concat_event_windows,
    concat_spikes_in_epochs,
)
from aind_ephys_utils.spiketrain.spike_utils import (
    align_to_events,
    align_to_events_exclude_trigger,
    build_synchrony_index,
    count_coincident_per_train,
    count_coincident_units,
    exclude_near,
    near_events,
)


class EpochsTest(unittest.TestCase):
    """Tests for epoch-based spike manipulation."""

    def test_concat_spikes_in_epochs_docstring_example(self) -> None:
        """Clip to epochs, relativise, and stitch into continuous time."""
        t = np.array([0.1, 0.3, 0.35, 0.9, 1.05, 1.2, 1.8, 2.1])
        epochs = [(0.25, 0.5), (1.0, 1.5), (1.9, 2.2)]
        out = concat_spikes_in_epochs(t, epochs)
        np.testing.assert_array_almost_equal(out, [0.05, 0.1, 0.3, 0.45, 0.95])

    def test_concat_spikes_in_epochs_extras(self) -> None:
        """return_extras exposes provenance arrays."""
        t = np.array([0.3, 1.05])
        epochs = [(0.25, 0.5), (1.0, 1.5)]
        out, extras = concat_spikes_in_epochs(t, epochs, return_extras=True)
        self.assertEqual(out.size, 2)
        self.assertEqual(
            set(extras),
            {
                "orig_indices",
                "epoch_ids",
                "epoch_offsets",
                "epoch_durations",
            },
        )
        np.testing.assert_array_equal(extras["epoch_ids"], [0, 1])

    def test_concat_event_windows(self) -> None:
        """Per-unit windows around events are stitched continuously."""
        st = [np.array([0.95, 1.05, 2.05])]
        events = np.array([1.0, 2.0])
        out = concat_event_windows(st, events, window=(-0.1, 0.1))
        # Window duration 0.2; event0 (0.9, 1.1) contributes 0.95, 1.05
        # -> rel 0.05, 0.15; event1 (1.9, 2.1) contributes 2.05 -> rel
        # (2.05 - 1.9) + offset 0.2 = 0.35.
        np.testing.assert_array_almost_equal(out[0], [0.05, 0.15, 0.35])

    def test_add_random_offset_with_wrap_bounds(self) -> None:
        """Offsetting with wrap keeps times in [0, total_duration)."""
        rng = np.random.default_rng(0)
        arrs = [np.array([0.1, 0.5, 0.9])]
        out = add_random_offset_with_wrap(arrs, 1.0, rng=rng)
        self.assertTrue(np.all(out[0] >= 0))
        self.assertTrue(np.all(out[0] < 1.0))
        self.assertEqual(out[0].size, 3)


class SpikeUtilsTest(unittest.TestCase):
    """Tests for proximity, alignment, and synchrony utilities."""

    def test_exclude_near_and_mask(self) -> None:
        """exclude_near drops near-event spikes; near_events masks them."""
        spikes = np.array([0.0, 1.0, 2.0, 3.0])
        events = np.array([1.0])
        kept = exclude_near(spikes, events, tol=0.1)
        np.testing.assert_array_equal(kept, [0.0, 2.0, 3.0])
        mask = near_events(spikes, events, tol=0.1)
        np.testing.assert_array_equal(mask, [False, True, False, False])

    def test_count_coincident_per_train(self) -> None:
        """Count how many other trains fire within tol of each spike."""
        target = np.array([1.0])
        others = [np.array([1.05]), np.array([5.0])]
        counts = count_coincident_per_train(target, others, tol=0.1)
        np.testing.assert_array_equal(counts, [1])

    def test_align_to_events(self) -> None:
        """Per-event relative times within the window."""
        times = np.array([0.9, 1.0, 1.1, 2.0])
        events = np.array([1.0, 2.0])
        trials = align_to_events(times, events, window=(-0.2, 0.2))
        self.assertEqual(len(trials), 2)
        np.testing.assert_array_almost_equal(trials[0], [-0.1, 0.0, 0.1])
        np.testing.assert_array_almost_equal(trials[1], [0.0])

    def test_align_to_events_exclude_trigger(self) -> None:
        """The triggering event itself is excluded from its own window."""
        ev = np.array([1.0, 1.3])
        trials = align_to_events_exclude_trigger(ev, ev, window=(-0.5, 0.5))
        np.testing.assert_array_almost_equal(trials[0], [0.3])
        np.testing.assert_array_almost_equal(trials[1], [-0.3])

    def test_synchrony_index_counts_distinct_units(self) -> None:
        """count_coincident_units counts distinct nearby other units."""
        trains = [
            np.array([1.0]),
            np.array([1.05]),
            np.array([5.0]),
        ]
        sorted_times, sorted_uids = build_synchrony_index(trains)
        self.assertEqual(sorted_times.size, 3)
        counts = count_coincident_units(
            trains[0],
            sorted_times,
            sorted_uids,
            n_units=3,
            tol=0.1,
            exclude_uid=0,
        )
        # Only unit 1 (1.05) is within 0.1 of the target spike at 1.0.
        np.testing.assert_array_equal(counts, [1])


if __name__ == "__main__":
    unittest.main()
