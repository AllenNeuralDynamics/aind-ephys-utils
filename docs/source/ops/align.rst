align
=====

Align spikes or continuous signals to event times. Works with ragged spikes,
binned spikes, or continuous signals.

Example
-------

.. code-block:: python

   aligned = da.ephys.align(events=events, to="go_cue", window=(-0.5, 1.0))

Notes
-----

- ``events`` can be a DataArray with dims ``(trial, event, bound)`` or
  ``(trial, event)``. The ``"start"`` bound is used by default.
- ``events`` can also be 1D array-like event times; in that case ``to`` is
  optional.
- The output keeps compatible trial/unit coordinates and updates
  ``ephys.valid_intervals``.

NumPy input
-----------

.. code-block:: python

   from aind_ephys_utils.ops import align

   aligned = align(
       spikes,               # list[np.ndarray], one array per unit
       events=event_times,   # 1D array-like, one event time per trial
       window=(-0.5, 1.0),
   )

Common patterns
---------------

Align to a different event:

.. code-block:: python

   aligned = da.ephys.align(events=events, to="reward", window=(-0.2, 0.8))
