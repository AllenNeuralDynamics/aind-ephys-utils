From dataframe to analysis
==========================

This tutorial shows how to ingest pandas DataFrames and convert them to
canonical DataArrays.

Steps
-----

1) Build ragged spikes with ``from_dataframe``
2) Align to events
3) Bin and smooth
4) Plot

1. Build ragged spikes
----------------------

.. code-block:: python

   import pandas as pd
   from aind_ephys_utils.adapters import from_dataframe

   units_df = pd.DataFrame(
       {
           "unit_id": [0, 1],
           "spike_times": [
               [0.05, 0.10, 0.20],
               [0.02, 0.30],
           ],
       }
   )
   trials_df = pd.DataFrame(
       {
           "trial_id": [0, 1],
           "start_time": [0.0, 1.0],
           "stop_time": [0.8, 1.8],
           "go_cue_time": [0.2, 1.2],
           "response": [1, 0],
       }
   )

   spikes = from_dataframe(units_df, trials_df, window=(-0.5, 1.0))

2. Align to events
------------------

.. code-block:: python

   events = from_dataframe(trials_df)
   aligned = spikes.ephys.align(events=events, to="go_cue", window=(-0.5, 1.0))

3. Bin and smooth
-----------------

.. code-block:: python

   binned = aligned.ephys.bin(0.01, window=(-0.5, 1.0))
   smoothed = binned.ephys.smooth(sigma=0.05)

4. Plot
-------

.. code-block:: python

   ax = smoothed.ephys.plot.psth(group_by="response")
