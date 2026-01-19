psth
====

Reduce across trials to form a PSTH-style summary.

Example
-------

.. code-block:: python

   psth = da.ephys.psth(dim="trial", reduce="mean")
