from_dataframe
==============

Convert pandas DataFrames into canonical DataArrays.

Example
-------

.. code-block:: python

   spikes = from_dataframe(units_df, trials_df, window=(-0.5, 1.0))

Inputs
------

- ``units_df``: one row per unit, with a ``spike_times`` column.
- ``trials_df``: one row per trial, with ``start_time`` and ``end_time``.
- Optional trial metadata columns become coordinates.

Common patterns
---------------

Events only:

.. code-block:: python

   events = from_dataframe(trials_df)

Session-only spikes:

.. code-block:: python

   spikes = from_dataframe(units_df)
