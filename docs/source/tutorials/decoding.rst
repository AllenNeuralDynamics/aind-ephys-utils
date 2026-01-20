Time-resolved decoding
======================

This tutorial walks through supervised reductions (logistic/LDA/coding direction)
with windowed fitting and time-resolved projections.

Outline
-------

- Bin and normalize
- Fit a classifier in a time window
- Project across the trial
- Interpret projections

1. Bin and normalize
--------------------

.. code-block:: python

   binned = spikes.ephys.bin(0.01, window=(-0.5, 1.0))
   smoothed = binned.ephys.smooth(sigma=0.05)
   norm = smoothed.ephys.normalize(dim=("trial", "time"), method="zscore")

2. Fit in a time window
-----------------------

.. code-block:: python

   labels = norm["response"]
   ds = norm.ephys.reduce(
       method="logistic",
       dim="unit",
       labels=labels,
       stack=("trial", "time"),
       window=(-0.2, 0.0),
       window_apply="fit_only",
       orthogonalize="qr",
   )

3. Interpret projections
------------------------

.. code-block:: python

   projections = ds["projections"]  # component x trial x time
   ax = projections.ephys.plot.trajectory(components=[0, 1])
