coding_direction
================

Binary-condition coding direction.

Use ``method="coding_direction"`` to compute a signed axis separating two
classes defined by ``labels``.

When to choose this method
--------------------------

Choose coding direction when you want a simple, interpretable discrimination axis between two conditions.

- Produces a direct "difference axis" that is easy to interpret biologically.
- Useful for comparing separation strength over time across sessions/areas.
- Lightweight and fast, making it convenient for repeated analyses.

This method is intentionally simple and binary-focused; it does not model
multiclass structure or richer covariance patterns like LDA.

.. code-block:: python

   ds = da.ephys.reduce(
       method="coding_direction",
       labels="choice",
   )

Notes
-----

- ``labels`` is required.
- Optional ``window`` can restrict fitting to a specific time range.
