Quickstart
==========

This quickstart shows the minimal flow from raw spike times to plots.

Install
-------

The library can be installed via ``pip``. It only has three dependencies: ``xarray`` (for core data structures), ``scikit-learn`` (for dimensionality reduction methods), and ``ipympl`` (for interactive plotting with ``matplotlib``).

.. code-block:: bash

   pip install aind-ephys-utils

Load and convert
----------------

To start working with spike data, it needs to be stored in a ``DataArray`` with the correct format. If you already have your spikes in Panda ``DataFrames`` with a ``spike_times`` columns (e.g., loaded from an NWB file), generating the appropriate data structure is straightforward:

.. code-block:: python

   import pandas as pd
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
       "outcome" : ['hit', 'miss']
   })

   spikes = from_dataframe(units_df, trials_df, window=(-0.5, 1.0), bin_size=None)

Data filtering and selection
-----------------------------

Once your data is in the right format, you can use ``xarray`` operations to filter and select data:

.. code-block:: python

   spikes_for_unit = spikes.sel(unit=1) # select unit by ID
   spikes_for_hits = spikes_for_unit.where(
        spikes.outcome == 'hit', 
        drop=True) # filter out miss trials


Bin, smooth, plot
------------------

Now you can perform operations (e.g. binning and smoothing) on the output and plot the results. All ``aind-ephys-utils`` methods are available via the ``ephys`` accessor, whereas built-in Xarray methods can be called directly on the ``DataArray`` object.

.. code-block:: python

   binned = spikes_for_hits.ephys.bin(0.01).ephys.smooth(sigma=0.05)
   ax = binned.ephys.plot.psth()

Next steps
----------

- Read :doc:`/concepts/data_model` for the expected shapes.
- Read :doc:`/concepts/xarray_primer` for essential Xarray concepts.
- Explore :doc:`/ops/index` for each operation and usage examples.
