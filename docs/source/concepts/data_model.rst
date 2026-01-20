Data model
==========

The package operates on xarray DataArrays with standard dimension names and
metadata. Understanding these conventions is the fastest way to use the API.

Core data structure
--------------------

The core representation is a ``DataArray`` with dimensions of ``(trial, unit, time)``.
The ``trial`` dimension can carry coordinates with trial metadata (e.g.,
response, stimulus). The ``unit`` dimension can carry coordinates with unit
metadata (e.g., depth, region). Each value represents a firing rate in Hz for a specific trial, unit, and time bin.

.. image:: /_static/dataarray_example.png
   :alt: DataArray with trial, unit, and time dimensions plus metadata coords.
   :align: center
   :width: 90%

Alternate data structures
----------------------------

This package can also operate on other types of ``DataArray`` objects:

- **Ragged spikes**: Each value represents an array of spike times (rather than a rate in a bin). Has dimensions of ``(trial, unit)``, or simply ``(unit,)``. Use the ``bin()`` operation to add a ``time`` dimension.
- **Trials**: Each value represents the time of an event within a trial. Has dimensions of ``(trial, event, bound)``. Use the ``align()`` operation to combine this with a ragged spikes object.

Return values
-------------

In most cases, ``DataArray`` returned by each operation will have the same dimensions and coordinates at the input ``DataArray``.

The exceptions are:

- ``bin()``: Expects a ragged spikes array with dimensions of ``(trial, unit)``, returns an array of firing rates with dimensions of ``(trial, unit, time)``. 

- ``align()``: Expects a ragged spikes array and a trials array. Either returns a ragged spikes array or an array with dimensions of ``(trial, unit, time)`` (if binning is desired).

- ``psth()``: Removes one dimension (usually trials)

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

Example shapes
--------------

Ragged spikes (trial, unit):

.. code-block:: python

   spikes.dims  # ("trial", "unit")
   spikes.dtype # object

Binned spikes (trial, unit, time):

.. code-block:: python

   rates.dims   # ("trial", "unit", "time")
   rates.dtype  # float

Key attributes
--------------

- ``ephys.kind``: ``"spikes_ragged"``, ``"binned"``, or ``"continuous"``
- ``ephys.timebase``: ``"session"`` or ``"trial"``
- ``ephys.time_unit``: usually ``"s"``
- ``ephys.valid_intervals``: list of ``(tmin, tmax)``

For details see :doc:`/concepts/conventions`.
