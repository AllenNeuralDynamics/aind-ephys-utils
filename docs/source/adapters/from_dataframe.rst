from_dataframe
==============

Convert pandas DataFrames into canonical DataArrays.

Example
-------

.. code-block:: python

   spikes = from_dataframe(units_df, trials_df, window=(-0.5, 1.0))
