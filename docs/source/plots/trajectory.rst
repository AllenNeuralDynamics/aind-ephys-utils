trajectory
==========

Plot low-dimensional trajectories after reduction.

Example
-------

.. code-block:: python

   ax = reduced.ephys.plot.trajectory(components=[0, 1])

Notes
-----

- Input should already be reduced (components x time).
- For 3D trajectories, pass three components.
