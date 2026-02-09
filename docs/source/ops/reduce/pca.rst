pca
===

Principal component analysis.

Use ``method="pca"`` to find orthogonal components that explain maximal
variance in the data.

When to choose this method
--------------------------

Choose PCA when you want a strong unsupervised baseline with minimal
assumptions.

- Best for exploratory analysis when you do not want to commit to task labels.
- Useful for denoising and quick visualization of dominant population patterns.
- Good first pass before trying more specialized methods such as dPCA or GPFA.

PCA is not task-aware: the largest-variance directions are not always the most
behaviorally relevant directions.

.. code-block:: python

   ds = da.ephys.reduce(
       method="pca",
       n_components=5,
   )

Key outputs
-----------

- ``projections``: component activity over samples/time.
- ``weights``: per-unit loading for each component.
- ``explained_variance_ratio``: variance explained by each component.
