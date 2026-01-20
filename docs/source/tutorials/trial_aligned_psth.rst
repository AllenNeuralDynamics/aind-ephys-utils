Trial-aligned PSTH
==================

This tutorial builds a PSTH aligned to events.

Outline
-------

- Align spikes to a cue
- Bin into rates
- Average across trials
- Plot

1. Align to events
------------------

.. code-block:: python

   events = from_dataframe(trials_df)
   aligned = spikes.ephys.align(events=events, to="go_cue", window=(-0.5, 1.0))

2. Bin into rates
-----------------

.. code-block:: python

   rates = aligned.ephys.bin(0.01, window=(-0.5, 1.0))

3. Average across trials
------------------------

.. code-block:: python

   psth = rates.ephys.psth(dim="trial", reduce="mean")

4. Plot
-------

.. code-block:: python

   ax = rates.ephys.plot.psth(group_by="response")
