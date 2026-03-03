smooth
======

Smooth a signal along a dimension.

Example
-------

.. code-block:: python

   smoothed = da.ephys.smooth(dim="time", method="gaussian", sigma=0.05)

Notes
-----

- Requires a uniformly spaced coordinate along ``dim``.
- ``sigma`` and ``window`` are in time units.
- Methods: ``gaussian`` (default) or ``boxcar``.
- NumPy inputs are supported; provide ``dims`` and ``coords`` for the smoothed
  dimension.
