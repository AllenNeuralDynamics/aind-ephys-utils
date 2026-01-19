align
=====

Align spikes or continuous signals to event times.

Example
-------

.. code-block:: python

   aligned = da.ephys.align(events=events, to="go_cue", window=(-0.5, 1.0))
