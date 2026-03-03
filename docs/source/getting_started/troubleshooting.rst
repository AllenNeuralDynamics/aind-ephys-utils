Troubleshooting
===============

NaNs in PCA
-----------

- Check zero-variance units; normalize handles these by returning zeros.

Missing time coord
------------------

- Assign a time coordinate before running time-based ops.
