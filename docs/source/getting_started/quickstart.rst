Quickstart
==========

This quickstart shows the minimal flow from raw spike times to plots.

Install
-------

.. code-block:: bash

   pip install aind-ephys-utils

Load and convert
----------------

.. code-block:: python

   import pandas as pd
   import xarray as xr
   from aind_ephys_utils.adapters import from_dataframe

   units_df = pd.DataFrame({
       "unit_id": [0, 1],
       "spike_times": [
           [0.05, 0.10, 0.20],
           [0.02, 0.30],
       ],
   })
   trials_df = pd.DataFrame({
       "trial_id": [0, 1],
       "start_time": [0.0, 1.0],
       "stop_time": [0.8, 1.8],
       "go_cue_time": [0.2, 1.2],
   })

   spikes = from_dataframe(units_df, trials_df, window=(-0.5, 1.0), bin_size=None)
   events = from_dataframe(trials_df)

Align, bin, plot
----------------

.. code-block:: python

   aligned = spikes.ephys.align(events=events, to="go_cue", window=(-0.5, 1.0))
   binned = aligned.ephys.bin(0.01).ephys.smooth(sigma=0.05)
   ax = binned.ephys.plot.psth()

Next steps
----------

- Read :doc:`/concepts/data_model` for the expected shapes.
- Read :doc:`/concepts/xarray_primer` for essential xarray concepts.
- Explore :doc:`/ops/index` for each operation and usage examples.
