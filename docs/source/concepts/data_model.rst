Data model
==========

Core data structure
--------------------

.. image:: /_static/trial_unit_time_array.png
   :alt: DataArray with trial, unit, and time dimensions plus metadata coords.
   :align: center
   :width: 50%

The core representation is a ``DataArray`` with dimensions of ``(trial, unit, time)``.
The ``trial`` dimension can carry coordinates with trial metadata (e.g.,
response, stimulus). The ``unit`` dimension can carry coordinates with unit
metadata (e.g., depth, region). Each value represents a firing rate in Hz for a specific trial, unit, and time bin.



Alternate data structures
----------------------------

This package can also operate on other types of ``DataArray`` objects:

.. image:: /_static/unit_array.png
   :alt: DataArray with unit coords.
   :align: center
   :width: 30%

**Ragged spikes**: Each value represents an array of spike times (rather than a rate in a bin). This array has dimensions of ``(trial, unit)``, or simply ``(unit,)``. Use the ``bin()`` operation to add a ``time`` dimension.

.. image:: /_static/trial_array.png
   :alt: DataArray with trial, event, and bound coords.
   :align: center
   :width: 30%

**Trials**: Each value represents the time of an event within a trial. This array dimensions of ``(trial, event, bound)``. Use the ``align()`` operation to combine this with a ragged spikes object.

Return values
-------------

In most cases, the ``DataArray`` returned by each operation will have the same dimensions and coordinates as the input ``DataArray``.

The exceptions are:

- ``bin()``: Expects a ragged spikes array with dimensions of ``(trial, unit)``, returns an array of firing rates with dimensions of ``(trial, unit, time)``. 

- ``align()``: Expects a ragged spikes array and a trials array. Either returns a ragged spikes array or an array with dimensions of ``(trial, unit, time)`` (if binning is desired).

- ``psth()``: Removes one dimension (usually trials), and optionally creates a new dimension if the `group_by` input argument is specified.

- ``reduce()``: Reduces one dimension (usually units), and returns a ``Dataset`` with two entries: ``projections`` with dimensions of ``(components, trials, time)`` and ``weights`` with dimensions of ``(components, unit)``.


Dimensions and coords
---------------------

The default dimension names are:

- ``trial``: trial index
- ``unit``: unit index
- ``time``: time axis in seconds (or other units)

Trial metadata (e.g., response, stimulus) should live in coordinates with
``dims=("trial",)`` so grouping and filtering are straightforward.

Inputs with alternate dimension names are allowed, but not recommended.

Timebase
--------

The attribute ``ephys.timebase`` indicates whether time values are in
``"session"`` or ``"trial"`` time. Session time is absolute (across the
recording), while trial time is relative to a per-trial anchor.

Valid intervals
---------------

``ephys.valid_intervals`` is a list of ``(tmin, tmax)`` tuples that define the
time range where data are valid. This matters for binning and plotting when
there are no spikes in parts of the window.


Key attributes
--------------

- ``ephys.kind``: ``"spikes_ragged"``, ``"binned"``, or ``"events"``
- ``ephys.timebase``: ``"session"`` or ``"trial"``
- ``ephys.time_unit``: usually ``"s"``
- ``ephys.valid_intervals``: list of ``(tmin, tmax)``

For details see :doc:`/concepts/conventions`.
