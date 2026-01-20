psth
====

Plot a PSTH from binned or continuous data.

Example
-------

.. code-block:: python

   ax = binned.ephys.plot.psth(group_by="response")

Notes
-----

- If ``unit`` is present, one line per unit is drawn by default.
- Set ``compute=False`` to plot already-reduced data.
