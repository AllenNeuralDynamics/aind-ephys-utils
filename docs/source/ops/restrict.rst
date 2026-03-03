restrict
========

``restrict`` crops data to a time window. It supports both dense time series
and ragged spike arrays.

Examples
--------

Dense time series:

.. code-block:: python

   cropped = rates.ephys.restrict(window=(0.0, 1.0))

Ragged spikes:

.. code-block:: python

   cropped = spikes.ephys.restrict(window=(-0.5, 0.5))

NumPy input:

.. code-block:: python

   cropped = restrict(
       rates_np,
       window=(0.0, 1.0),
       dims=("time",),
       coords={"time": time_values},
   )
