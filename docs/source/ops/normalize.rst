normalize
=========

Normalize across one or more dimensions.

Example
-------

.. code-block:: python

   norm = da.ephys.normalize(dim=("trial", "time"), method="zscore")
