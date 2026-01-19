smooth
======

Smooth a signal along a dimension.

Example
-------

.. code-block:: python

   smoothed = da.ephys.smooth(dim="time", method="gaussian", sigma=0.05)
