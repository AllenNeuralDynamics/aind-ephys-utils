reduce
======

Dimensionality reduction and supervised projections.

The ``reduce`` operation supports seven methods:

- ``pca``
- ``gpfa``
- ``dpca``
- ``coding_direction``
- ``logistic``
- ``lda``
- ``rrr``

.. toctree::
   :maxdepth: 1

   pca
   gpfa
   dpca
   coding_direction
   logistic
   lda
   rrr

Quick examples
--------------

Example: PCA
^^^^^^^^^^^^

PCA works on ``DataArray`` inputs with dimensions like
``(trials, units, time)``. If ``n_components`` is not specified, it defaults
to 5.

.. code-block:: python

   ds = da.ephys.reduce(method="pca")
   proj = ds["projections"]
   w = ds["weights"]

Example: GPFA
^^^^^^^^^^^^^

GPFA works on pre-binned data with dimensions
``(trials, units, time)`` and uses the same default component behavior as PCA.

.. code-block:: python

   ds = da.ephys.reduce(method="gpfa", n_components=2)
   proj = ds["projections"]
   w = ds["weights"]

Example: Supervised methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supervised methods work on ``DataArray`` inputs with dimensions like
``(trials, units, time)``. They require ``labels`` to define condition/group structure. A ``window`` can optionally restrict fitting to a time range. When multiple labels or windows are used, outputs can be orthogonalized with ``"qr"`` (Gram-Schmidt) or ``"svd"`` (Singular Value Decomposition) methods.

.. code-block:: python

   ds = da.ephys.reduce(
       method="lda",
       labels=["choice", "stimulus"],
       window=(-0.2, 0.0),
       orthogonalize="qr",
   )
