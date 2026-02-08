reduce
======

Dimensionality reduction and supervised projections.

Example: PCA
---------------------

PCA works on any ``DataArray`` with dimensions of ``(trials, units, time)``. If the ``'n_components'`` input argument is not specified, it will return 5 components by default.

.. code-block:: python

   ds = da.ephys.reduce(method="pca")
   proj = ds["projections"]
   w = ds["weights"]

Example: GPFA
---------------------

GPFA works on pre-binned data with dimensions ``(trials, units, time)`` and
uses the same default component behavior as PCA.

.. code-block:: python

   ds = da.ephys.reduce(method="gpfa")
   proj = ds["projections"]
   w = ds["weights"]

Example: Supervised methods
----------------------------

Supervised methods work on any ``DataArray`` with dimensions of ``(trials, units, time)``. They require a ``'labels'`` input argument that specifies which coordinate should be used for grouping data. A ``'window'`` can optionally be specified to restrict the computation to a certain segment of the data. If there are multiple labels or multiple windows specified, the outputs can be orthogonalized using the ``'qr'`` (Gram-Schmidt process) or ``'svd'`` (singular value decomposition) methods. 

.. code-block:: python

   ds = da.ephys.reduce(
       method="lda",
       labels=['choice', 'stimulus'],
       window=(-0.2, 0.0),
       orthogonalize="qr",
   )

Available supervised methods are:

* ``'dpca'``: Demixed principal component analysis

* ``'coding_direction'``: Coding direction

* ``'logistic'``: Logistic regression

* ``'lda'``: Linear discriminant analysis

* ``'rrr'``: Reduced rank regression
