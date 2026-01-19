reduce
======

Dimensionality reduction and supervised projections.

Example
-------

.. code-block:: python

   ds = da.ephys.reduce(method="pca_time", n_components=5)
   proj = ds["projections"] if isinstance(ds, xr.Dataset) else ds

Supervised example
-----------------

.. code-block:: python

   ds = da.ephys.reduce(
       method="logistic",
       dim="unit",
       labels=labels,
       stack=("trial", "time"),
       window=(-0.2, 0.0),
       orthogonalize="qr",
   )
