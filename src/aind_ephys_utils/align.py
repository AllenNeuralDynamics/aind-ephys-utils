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
    unit_ids=None,
    return_df=False,
    spike_times_key="spike_times",
):
    """
    Aligns spikes times (sorted in ascending order) to
    a set of event times

    Spike times can take the following formats:
    - 1-dimensional ndarray of times for one unit
    - list of 1-dimensional ndarrays of times for multiple units
    - dict of 1-dimensional ndarrays with unit IDs as keys
    - DataFrame indexed by unit IDs, with a "spike_times" column

    If the optional `bin_size` argument is supplied, the spike
    times are binned.

    If `return_dataframe` is set to True, the results will be
    returned as a pandas DataFrame (or xarray.DataArray if results
    are binned)

    Parameters
    ----------
    times : ndarray, List[ndarrays], dict, or DataFrame
        1-D sequence(s) of times to align (in seconds). Must
        be sorted in ascending order.
    events : ndarray
        1-D sequence of reference times (in seconds).
    interval : tuple
        Start and end of the window around each event (in seconds).
    bin_size : float, optional
        Bin size (in seconds); if None, then individual times will be returned.
    unit_ids : List[int]
        Labels for each unit. If len(unit_ids) < len(times) and times is
        dict or DataFrame, then this argument will specify which units
        to align.
    event_labels : List[int] or List[str]
        Labels for each event (optional).
    return_df : bool, optional (default = False)
        If True, returns the results as a pandas DataFrame
        (or xarray.DataArray if binning is enabled).
    spike_times_key : str, optional (default = 'spike_times')
        If 'times' argument is a DataFrame, this specifies the name of the
        column containing the spike times.

    Returns
    -------
    if return_df = False and bin_size = None:
    aligned_times : ndarray
        1-D sequence of times relative to the events of interest.
    event_indices : ndarray
        1-D sequence of associated event index for each time in aligned_times.
    unit_ids : list or ndarray
        1-D sequence of unit IDs for each sequence of spike times

    if return_df = False and bin_size is not None:
    bins : ndarray
        1-D sequence of time bin left edges
    counts : ndarray
        2-D or 3-D array of spike counts of size trials x bins (x units)
    unit_ids : ndarray
        1-D sequence of unit IDs

    if return_df = True and bin_size = None:
    df : pd.DataFrame with columns:
        - time : aligned times
        - event_index : event index for each time
        - event_label : event label for each time (optional)
        - unit_id : unit label for each time (optional)

    if return_df = True and bin_size is not None:
    da : xr.DataArray with dimensions:
        - time : time relative to each event
        - event_index or event_label : label for each event
        - unit_id : label of each unit

    """

    if unit_ids is None:
        if isinstance(times, np.ndarray):
            unit_ids = [0]
        elif isinstance(times, list):
            unit_ids = np.arange(len(times))
        elif isinstance(times, dict):
            unit_ids = np.array(list(times.keys()))
        elif isinstance(times, pd.DataFrame):
            unit_ids = times.index.values

    if bin_size is not None:
        bins = np.arange(interval[0], interval[1] + bin_size, bin_size)
        counts = np.zeros((bins.size - 1, events.size, len(unit_ids)))

    aligned_times = []
    event_indices = []
    unit_labels = []

    if event_labels is not None:
        if len(event_labels) != len(events):
            raise ValueError(
                "events and event_labels must be the same length."
            )
        labels = []

    for i, start in enumerate(events):
        for j, unit in enumerate(unit_ids):
            if isinstance(times, np.ndarray):
                unit_times = times
            elif isinstance(times, list):
                unit_times = times[unit]
            elif isinstance(times, dict):
                unit_times = times[unit]
            elif isinstance(times, pd.DataFrame):
                unit_times = times.loc[unit][spike_times_key]

            start_index = np.searchsorted(unit_times, start + interval[0])
            end_index = np.searchsorted(unit_times, start + interval[1])

            if bin_size is not None:
                counts[:, i, j] = np.histogram(
                    unit_times[start_index:end_index] - start, bins
                )[0]
            else:
                aligned_times.append(unit_times[start_index:end_index] - start)
                event_indices.append(
                    np.zeros((end_index - start_index,), dtype="int") + i
                )
                unit_labels.append(
                    np.zeros((end_index - start_index,), dtype="int") + unit
                )
                if event_labels is not None:
                    labels.append(
                        [event_labels[i]] * (end_index - start_index)
                    )

    if not return_df:
        if bin_size is None:
            return (
                np.concatenate(aligned_times),
                np.concatenate(event_indices),
                np.concatenate(unit_labels),
            )
        else:
            return bins[:-1], np.squeeze(counts), unit_ids
    else:
        if bin_size is None:
            if event_labels is None:
                return pd.DataFrame(
                    data={
                        "time": np.concatenate(aligned_times),
                        "event_index": np.concatenate(event_indices),
                        "unit_id": np.concatenate(unit_labels),
                    }
                )
            else:
                return pd.DataFrame(
                    data={
                        "time": np.concatenate(aligned_times),
                        "event_index": np.concatenate(event_indices),
                        "event_label": np.concatenate(labels),
                        "unit_id": np.concatenate(unit_labels),
                    }
                )
        else:
            if event_labels is None:
                return xr.DataArray(
                    data=counts,
                    coords={
                        "time": bins[:-1],
                        "event_index": np.arange(len(events)),
                        "unit_id": unit_ids,
                    },
                )
            else:
                return xr.DataArray(
                    data=counts,
                    coords={
                        "time": bins[:-1],
                        "event_label": event_labels,
                        "unit_id": unit_ids,
                    },
                )


align_to_events = to_events
""" Alias for `to_events` """
