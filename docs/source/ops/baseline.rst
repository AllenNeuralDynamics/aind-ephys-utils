baseline
========

Baseline-correct signals within a time window.

Example
-------

.. code-block:: python

   base = da.ephys.baseline(window=(-0.2, 0.0), mode="subtract")

Notes
-----

- Modes: ``subtract``, ``divide``, ``zscore``.
- The baseline window must overlap the time coordinate.
