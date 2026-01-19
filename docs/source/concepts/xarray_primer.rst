Xarray primer
=============

This section introduces the minimal xarray concepts used in this library.

Core ideas
----------

- ``dims``: named axes (e.g. "trial", "unit", "time")
- ``coords``: labeled values along dims (e.g. time in seconds)
- ``attrs``: metadata (e.g. ephys.time_unit)

Common patterns
---------------

Select by label

.. code-block:: python

   da.sel(trial=[0, 1])

Filter with conditions

.. code-block:: python

   da_correct = da.where(da["response"] == 1, drop=True)

Group by trial metadata

.. code-block:: python

   for condition, sub in da.groupby("response"):
       ...

Stack and unstack

.. code-block:: python

   da_stack = da.stack(sample=("trial", "time"))
   da_unstack = da_stack.unstack("sample")
