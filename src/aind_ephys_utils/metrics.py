""" Module to compute response latency to a set of events
"""

import numpy as np

from . import align


def spike_latency(
    times,
    events,
    interval,
    use_psth=True,
    std_above_baseline=2,
    bin_size=0.001,
):
    """
    Computes latency of spikes to a set of events

    If `use_psth` is set to True, the latency will be computed
    using the average PSTH, rather than spikes on individual trials

    If `use_psth` is set to False, the latency will be computed
    as the median time to first spike on individual trials

    Parameters
    ----------
    times : ndarray
        1-D sequence of times to align (in seconds). Must
        be sorted in ascending order.
    events : ndarray
        1-D sequence of reference times (in seconds).
    interval : tuple
        First value = baseline interval (ignored if `use_psth` = False)
        Second value = maximum latency
    use_psth : bool, optional
        Set to True to use PSTH method, False otherwise
    std_above_baseline : float, optional
        Determines threshold for response onset
        Latency = first value above Mean + Std * T
    bin_size : float, optional
        Determines bin size (in seconds) for PSTH method

    Returns
    -------
    if use_psth = True:
    latency : float
        First spike latency in s
    psth : float
        The PSTH used to compute latency

    if use_psth = False:
    first_spike_latency : float
        First spike latency in s
    individual_latencies : ndarray
        Latency values for all trials with spikes

    """

    if use_psth:
        win = np.array([0, 0.25, 0.5, 0.25, 0])  # 5-point Hann window

        bins, counts = align.to_events(
            times, events, interval, bin_size=bin_size
        )

        psth = np.convolve(np.mean(counts, 1) / bin_size, win, mode="same")

        onset = np.searchsorted(bins, 0)

        baseline_firing_rate = np.mean(psth[:onset])
        baseline_std = np.std(psth[:onset])
        threshold = baseline_firing_rate + std_above_baseline * baseline_std

        first_spike_latency = np.argmax(psth[onset:] > threshold)

        return first_spike_latency * bin_size, psth

    else:
        df = align.to_events(
            times, events, (0, interval[1]), return_dataframe=True
        )

        latencies = np.squeeze(df.groupby("event_index").min().values)

        return np.median(latencies), latencies
