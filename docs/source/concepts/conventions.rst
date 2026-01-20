Conventions
===========

This package relies on stable naming conventions for dimensions, coordinates,
and attributes. Following these conventions keeps analysis code composable.

Dimensions
----------

- ``trial``: trial index for trial-aligned data
- ``unit``: neuron/unit index
- ``time``: time samples in seconds (or other unit in ``ephys.time_unit``)
- ``component``: unit axis may be replaced by component axis after dimensionality reduction


Attributes
----------

- ``ephys.kind``: ``"spikes_ragged"``, ``"binned"``, or ``"continuous"``
- ``ephys.timebase``: ``"session"`` or ``"trial"``
- ``ephys.time_unit``: typically ``"s"``
- ``ephys.valid_intervals``: list of ``(tmin, tmax)`` tuples
- ``ephys.history``: list of processing steps (optional)

Defaults for adapters
---------------------

When using ``from_dataframe``, the default column names are:

- ``spike_times``: spike times in the units table
- ``start_time``: trial start time in the trials table
- ``end_time`` (or ``stop_time``): trial end time in the trials table

Events representation
---------------------

The adapter builds events as a DataArray with dims ``(trial, event, bound)``,
where ``bound`` contains ``"start"`` and ``"end"``. The ``align`` op accepts
this representation directly and will use the ``"start"`` bound by default.

Where to find definitions
-------------------------

All constants are defined in ``aind_ephys_utils.standards.conventions``.
