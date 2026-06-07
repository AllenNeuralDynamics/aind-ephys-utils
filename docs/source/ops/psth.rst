psth
====

Reduce across trials to form a PSTH-style summary.

Example
-------

.. code-block:: python

   psth = da.ephys.psth(dim="trial")
   psth_by_condition = da.ephys.psth(
       dim="trial", group_by="condition"
   )
   smoothed_psth = ragged_spikes.ephys.psth(
       bin_size="auto", smooth_window=0.05, window=(-0.5, 1.0)
   )

Notes
-----

- Works on binned or continuous data, or ragged spikes when ``bin_size`` is
  provided.
- Use ``smooth_window`` to apply boxcar smoothing after trial reduction.
- Use ``group_by=...`` to reduce each trial group separately.
- Use ``keep_trials=True`` to append the summary to trial data.
- NumPy inputs are supported. Pass ``dims`` (for example ``("trial", "time")``)
  when not using xarray.
