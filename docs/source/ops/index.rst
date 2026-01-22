Operations
==========

Operations (or "ops") are pure, composable functions that transform ``DataArrays`` without side
effects. They preserve compatible coordinates and return ``DataArrays`` or
``Datasets`` so you can continue using standard Xarray methods.

.. toctree::
   :maxdepth: 2

   align
   bin
   smooth
   baseline
   normalize
   psth
   reduce
   restrict
