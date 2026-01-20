psth
====

Reduce across trials to form a PSTH-style summary.

Example
-------

.. code-block:: python

   psth = da.ephys.psth(dim="trial", reduce="mean")

Notes
-----

- Works on binned or continuous data.
- Use ``keep_trials=True`` to append the summary to trial data.
