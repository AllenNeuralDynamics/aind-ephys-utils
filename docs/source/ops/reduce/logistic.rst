logistic
========

Logistic-regression projection axis.

Use ``method="logistic"`` to fit a linear classifier and project neural data
onto its learned decision direction(s).

When to choose this method
--------------------------

Choose logistic reduction when predictive decoding performance is the primary
goal.

- Optimizes class separation with a discriminative objective.
- Supports regularization, which can improve robustness in high-dimensional neural feature spaces.
- Useful for linking low-dimensional projections to classifier decision
  boundaries.

Compared with coding direction, logistic can fit more flexible discriminative axes, but those axes may be less directly interpretable as condition means.

.. code-block:: python

   ds = da.ephys.reduce(
       method="logistic",
       labels="choice",
   )

Notes
-----

- ``labels`` is required.
- ``regularization`` and ``cv`` can be used to control model fitting.
