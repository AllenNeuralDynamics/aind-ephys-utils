lda
===

Linear Discriminant Analysis projection.

Use ``method="lda"`` to find axes that maximize separation between classes in
``labels`` while accounting for within-class variance.

When to choose this method
--------------------------

Choose LDA when you want supervised projections that remain interpretable and
naturally support multiclass settings.

- Maximizes between-class separation relative to within-class spread.
- Often works well when classes are reasonably Gaussian in feature space.
- Provides compact discriminant axes that can be easier to interpret than
  generic classifier coefficients.

LDA can degrade when class covariance assumptions are badly violated or when
sample size is small relative to dimensionality.

.. code-block:: python

   ds = da.ephys.reduce(
       method="lda",
       labels="choice",
   )

Notes
-----

- ``labels`` is required.
- Works for binary or multiclass condition labels.
