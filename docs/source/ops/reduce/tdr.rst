tdr
===

Targeted dimensionality reduction.

Use ``method="tdr"`` to estimate task-relevant axes from condition-averaged
activity.

When to choose this method
--------------------------

Choose TDR when you want low-dimensional projections aligned to specific task
conditions rather than only maximal variance.

- Produces interpretable condition-linked axes.
- Applies PCA denoising before regression-based axis estimation.
- Useful for visualizing condition structure over time in a compact space.

This method is condition-focused and relies on meaningful condition labels or
pre-averaged condition inputs.

.. code-block:: python

   ds = da.ephys.reduce(
       method="tdr",
       labels="choice",
       n_components=12,
   )

Accepted inputs
---------------

- Trial-based data: ``(trial, unit, time)`` with ``labels`` provided.
- Pre-averaged data: ``(condition, unit, time)``.

Key outputs
-----------

- ``projections``: ``(condition, component, time)``.
- ``weights``: ``(condition, unit)``.

Notes
-----

- For trial-based inputs, trials are condition-averaged internally.
- ``orthogonalize`` supports ``"none"``, ``"qr"``, and ``"svd"``.
