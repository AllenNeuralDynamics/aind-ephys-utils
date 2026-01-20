normalize
=========

Normalize across one or more dimensions.

Example
-------

.. code-block:: python

   norm = da.ephys.normalize(dim=("trial", "time"), method="zscore")

Notes
-----

- Methods: ``zscore``, ``minmax``, ``robust``.
- Zero-variance slices are set to 0 to avoid NaNs.
