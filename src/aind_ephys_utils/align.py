""" Module to align two sorted temporal sequences
(usually spike times and events).
"""

import numpy as np
import pandas as pd
import xarray as xr


def to_events(  # noqa: C901
    times,
    events,
    interval,
    bin_size=None,
    event_labels=None,
    return_dataframe=False,
):
    """
    Aligns spikes times (sorted in ascending order) to
    a set of event times

    If the optional `bin_size` argument is supplied, the spike
    times are binned.

    If `return_dataframe` is set to True, the results will be
    returned as a pandas DataFrame (or xarray.DataArray if results
    are binned)

    Parameters
    ----------
    times : ndarray
        1-D sequence of times to align (in seconds). Must
        be sorted in ascending order.
    events : ndarray
        1-D sequence of reference times (in seconds).
    interval : tuple
        Start and end of the window around each event (in seconds).
    bin_size : float, optional
        Bin size (in seconds); if None, then individual times will be returned.
    event_labels : List[int] or List[str]
        Labels for each event (optional).
    return_dataframe : bool, optional (default = False)
        If True, returns the results as a pandas DataFrame
        (or xarray.DataArray if binning is enabled).

    Returns
    -------
    if return_dataframe = False and bin_size = None:
    aligned_times : ndarray
        1-D sequence of times relative to the events of interest.
    event_indices : ndarray
        1-D sequence of associated event index for each time in aligned_times.

    if return_dataframe = False and bin_size is not None:
    bins : ndarray
        1-D sequence of time bin left edges
    counts : ndarray
        2-D array of spike counts of size trials x bins

    if return_dataframe = True and bin_size = None:
    df : pd.DataFrame with columns:
        - time : aligned times
        - event_index : event index for each time
        - event_label : event label for each time (optional)

    if return_dataframe = True and bin_size is not None:
    da : xr.DataArray with dimensions:
        - time : time relative to each event
        - event_index or event_label : label for each event

    """

    if bin_size is not None:
        bins = np.arange(interval[0], interval[1] + bin_size, bin_size)
        counts = np.zeros((bins.size - 1, events.size))

    aligned_times = []
    event_indices = []

    if event_labels is not None:
        if len(event_labels) != len(events):
            raise ValueError(
                "events and event_labels must be the same length."
            )
        labels = []

    for i, start in enumerate(events):
        start_index = np.searchsorted(times, start + interval[0])
        end_index = np.searchsorted(times, start + interval[1])

        if bin_size is not None:
            counts[:, i] = np.histogram(
                times[start_index:end_index] - start, bins
            )[0]
        else:
            aligned_times.append(times[start_index:end_index] - start)
            event_indices.append(
                np.zeros((end_index - start_index,), dtype="int") + i
            )
            if event_labels is not None:
                labels.append([event_labels[i]] * (end_index - start_index))

    if not return_dataframe:
        if bin_size is None:
            return np.concatenate(aligned_times), np.concatenate(event_indices)
        else:
            return bins[:-1], counts
    else:
        if bin_size is None:
            if event_labels is None:
                return pd.DataFrame(
                    data={
                        "time": np.concatenate(aligned_times),
                        "event_index": np.concatenate(event_indices),
                    }
                )
            else:
                return pd.DataFrame(
                    data={
                        "time": np.concatenate(aligned_times),
                        "event_index": np.concatenate(event_indices),
                        "event_label": np.concatenate(labels),
                    }
                )
        else:
            if event_labels is None:
                return xr.DataArray(
                    data=counts,
                    coords={
                        "time": bins[:-1],
                        "event_index": np.arange(len(events)),
                    },
                )
            else:
                return xr.DataArray(
                    data=counts,
                    coords={"time": bins[:-1], "event_label": event_labels},
                )


align_to_events = to_events
""" Alias for `to_events` """
