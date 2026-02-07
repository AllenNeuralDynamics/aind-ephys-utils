bin
===

Bin ragged spikes into a dense array of firing rates in Hz.

Example
-------

.. code-block:: python

   binned = da.ephys.bin(0.01, window=(-0.5, 1.0))

Notes
-----

- Input must be ragged spikes (object dtype).
- Output dims follow input order (trial/unit/time or unit/trial/time).
- Coordinates attached to trial/unit are preserved.

Common patterns
---------------

Counts instead of rates:

.. code-block:: python

   counts = da.ephys.bin(0.01, window=(-0.5, 1.0), output="count")
