psth
====

Reduce across trials to form a PSTH-style summary.

Example
-------

.. code-block:: python

   psth = da.ephys.psth(dim="trial", reduce="mean")
   psth_by_condition = da.ephys.psth(
       dim="trial", reduce="mean", group_by="condition"
   )

Notes
-----

- Works on binned or continuous data.
- Use ``group_by=...`` to reduce each trial group separately.
- Use ``keep_trials=True`` to append the summary to trial data.
