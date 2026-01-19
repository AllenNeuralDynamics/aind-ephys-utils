Data model
==========

The package operates on xarray DataArrays with standard dimension names and
attributes. There are three canonical kinds of data:

- Ragged spikes: dtype=object with dims (trial, unit)
- Binned spikes: numeric with dims (trial, unit, time)
- Continuous signals: numeric with a time dimension

Key attributes
--------------

- ephys.kind: "spikes_ragged", "binned", or "continuous"
- ephys.timebase: "session" or "trial"
- ephys.time_unit: usually "s"
- ephys.valid_intervals: list of (tmin, tmax)

For details see :doc:`/concepts/conventions`.
