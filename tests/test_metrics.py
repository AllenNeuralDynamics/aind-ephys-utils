"""Tests spike metrics."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from aind_ephys_utils.metrics import spike_latency


class SpikeLatencyTest(unittest.TestCase):
    """Tests spike latency methods."""

    offset = 0.002
    events = np.arange(10)  # 10 events
    times = events + offset

    def test_latency(self) -> None:
        """Test the `latency` method."""

        first_spike, psth = spike_latency(self.times, self.events, (-0.1, 0.1))

        assert_allclose(first_spike, 0.0)

        first_spike, latencies = spike_latency(
            self.times, self.events, (-0.1, 0.1), use_psth=False
        )

        assert_allclose(first_spike, self.offset)
        assert_array_equal(latencies, self.times - self.events)


if __name__ == "__main__":
    """Run the tests"""
    unittest.main()
