bin
===

Bin ragged spikes into a dense array of firing rates in Hz.

Example
-------

.. code-block:: python

   binned = da.ephys.bin(0.01, window=(-0.5, 1.0))
   auto_binned = da.ephys.bin(bin_size="auto", window=(-0.5, 1.0))

Notes
-----

- Input must be ragged spikes (object dtype).
- Bin width may be passed as ``dt`` or ``bin_size``.
- Use ``"auto"`` for Shimazaki-Shinomoto optimal bin-width selection.
- Output dims follow input order (trial/unit/time or unit/trial/time).
- Coordinates attached to trial/unit are preserved.
- Supports NumPy/ragged-list inputs with ``return_type`` control.

Common patterns
---------------

Counts instead of rates:

.. code-block:: python

   counts = da.ephys.bin(0.01, window=(-0.5, 1.0), output="count")

NumPy/ragged-list input:

.. code-block:: python

   from aind_ephys_utils.ops import bin

   binned = bin(
       aligned_spikes,   # list[list[np.ndarray]] or list[np.ndarray]
       dt=0.01,
       window=(-0.5, 1.0),
       output="rate",
   )
