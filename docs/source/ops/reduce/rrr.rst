rrr
===

Reduced-rank regression.

Use ``method="rrr"`` to learn a low-rank mapping from neural activity to
target variables.

When to choose this method
--------------------------

Choose RRR when your main objective is predicting external variables from
neural activity while enforcing a low-dimensional predictive subspace.

- Directly models neural-to-target relationships (unlike unsupervised methods).
- Useful for neural encoding/decoding problems where targets are continuous or
  multidimensional.
- Low-rank structure can improve generalization and interpretability.

RRR requires meaningful target definitions and enough data to estimate the
mapping reliably.

What goes in ``target_da``?
---------------------------

``target_da`` is the ``targets=...`` argument passed to
``da.ephys.reduce(method="rrr", ...)``.

It must be an ``xarray.DataArray`` with:

- All sample dimensions used by ``reduce`` (by default, ``trial`` and ``time``
  because the default ``stack`` is ``("trial", "time")``).
- Exactly one additional non-sample dimension that indexes target variables
  (for example ``"target"``, ``"behavior"``, or ``"kinematic_feature"``).

Common choices for target variables include:

- Behavioral signals over time (running speed, pupil size, lick rate).
- Kinematics over time (position/velocity components).
- Task-variable encodings (for example one-hot vectors for stimulus/choice).

The sample coordinates in ``target_da`` should align with the neural data
samples used for fitting.

.. code-block:: python

   ds = da.ephys.reduce(
       method="rrr",
       targets=target_da,
       rank=3,
   )

Example
-------

Time-resolved targets (default stacking over ``trial`` and ``time``):

.. code-block:: python

   target_da = xr.DataArray(
       target_values,  # shape: (n_trial, n_time, n_target)
       dims=("trial", "time", "target"),
       coords={
           "trial": da.coords["trial"],
           "time": da.coords["time"],
           "target": ["x_vel", "y_vel"],
       },
   )

   ds = da.ephys.reduce(
       method="rrr",
       targets=target_da,
       rank=3,
   )

Trial-level targets only (for example per-trial choice), by stacking trials
only:

.. code-block:: python

   target_da = xr.DataArray(
       trial_targets,  # shape: (n_trial, n_target)
       dims=("trial", "target"),
       coords={"trial": da.coords["trial"], "target": ["left", "right"]},
   )

   ds = da.ephys.reduce(
       method="rrr",
       targets=target_da,
       stack=("trial",),  # now targets need only the trial sample dim
       rank=2,
   )

Notes
-----

- ``targets`` is required.
- ``rank`` controls the dimensionality of the learned predictive subspace.
