Xarray primer
=============

This section introduces the minimal Xarray concepts used in this library. The
goal is to make the docs usable even if you have never seen Xarray before.

Core ideas
----------

- ``dims``: named axes (e.g. "trial", "unit", "time")
- ``coords``: labeled values along dims (e.g. time in seconds, trial metadata)
- ``attrs``: metadata for the whole array (e.g. ephys.time_unit)

Why it matters
--------------

Xarray gives names to axes. That means you can select, group, and combine data
without remembering axis order. The library uses these names to keep all
operations composable.

Common patterns
---------------

Select by label

.. code-block:: python

   da.sel(trial=[0, 1])

Select by integer index
-----------------------

.. code-block:: python

   da.isel(trial=0)

Filter with conditions

.. code-block:: python

   da_correct = da.where(da["response"] == 1, drop=True)

Assign coordinates
------------------

.. code-block:: python

   da = da.assign_coords(response=("trial", response_values))

Group by trial metadata
-----------------------

.. code-block:: python

   for condition, sub in da.groupby("response"):
       ...

Reduce across dimensions
------------------------

.. code-block:: python

   mean_over_trials = da.mean(dim="trial")
   mean_over_time = da.mean(dim="time")

Stack and unstack
-----------------

.. code-block:: python

   da_stack = da.stack(sample=("trial", "time"))
   da_unstack = da_stack.unstack("sample")

Common gotchas
--------------

- If a dimension has no coordinate values, some operations will raise errors.
  Use ``assign_coords`` to add a time axis.
- Always check ``da.dims`` and ``da.coords`` after transformations.
