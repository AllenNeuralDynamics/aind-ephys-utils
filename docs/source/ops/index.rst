Operations
==========

Operations (or "ops") are pure, composable functions that transform ``DataArrays`` without side
effects. They preserve compatible coordinates and return ``DataArrays`` or
``Datasets`` so you can continue using standard Xarray methods.

NumPy compatibility
-------------------

Most core ops also accept NumPy-based inputs. For dense arrays, pass
``dims=...`` and (for time-based ops) ``coords=...`` so dimensions are
interpretable. For ragged spikes, Python list formats are supported:

- session-aligned ragged: ``list[np.ndarray]`` (unit -> spikes)
- trial-aligned ragged: ``list[list[np.ndarray]]`` (trial -> unit -> spikes)

You can control outputs with ``return_type``:

- ``"auto"`` (default): return same style as input
- ``"xarray"``: force ``DataArray`` output
- ``"numpy"``: force NumPy/ragged-list output

.. toctree::
   :maxdepth: 2

   align
   bin
   smooth
   baseline
   normalize
   psth
   pseudopop
   reduce/index
   restrict
