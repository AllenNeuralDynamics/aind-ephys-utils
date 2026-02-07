Xarray primer
=============

This section introduces the minimal Xarray concepts used in this library. The
goal is to make the docs usable even if you have never seen Xarray before.

Core ideas
----------

- ``dims``: named axes (e.g. "trial", "unit", "time")
- ``coords``: labeled values along dims (e.g. time in seconds, trial metadata)
- ``attrs``: metadata for the whole array (e.g. ephys.time_unit)

Why it matters
--------------

Xarray gives names to axes. That means you can select, group, and combine data
without remembering axis order. The library uses these names to keep all
operations composable.

Common patterns
---------------

Select by label
^^^^^^^^^^^^^^^

Use ``sel`` to access data via *meaningful labels*, rather than
array positions. This is the safest way to subset when trials/units are
reordered, dropped, or merged across preprocessing steps, because the
selection follows coordinate values instead of implicit integer offsets.

.. code-block:: python

   da.sel(trial=[0, 1])

Select by integer index
^^^^^^^^^^^^^^^^^^^^^^^

Use ``isel`` for fast positional slicing when labels are unavailable or when
you intentionally want "the first N items" regardless of their coordinate
values. This is especially useful for quick inspection, debugging, and
deterministic train/test splits based on order.

.. code-block:: python

   da.isel(trial=0)
   da.isel(unit=slice(0,10))

Filter with conditions
^^^^^^^^^^^^^^^^^^^^^^

Use ``where(..., drop=True)`` to express quality-control or behavioral filters
as explicit boolean logic. This keeps filtering readable and reproducible while preserving aligned coordinates on all remaining dimensions.

.. code-block:: python

   da_correct = da.where(da["response"] == 1, drop=True)

Assign coordinates
^^^^^^^^^^^^^^^^^^

Use ``assign_coords`` to attach semantic labels (condition IDs, brain region,
session metadata) so later operations can be written using understandable terms. In practice, many higher-level workflows
(``groupby``, condition averages, plotting labels) depend on these coordinates.

.. code-block:: python

   da = da.assign_coords(response=("trial", response_values))

Group by trial metadata
^^^^^^^^^^^^^^^^^^^^^^^

Use ``groupby`` when you need per-condition summaries while preserving the
same analysis pipeline across conditions. It replaces manual mask loops with a
single declarative step and guarantees each group stays aligned to its own
coordinate label.

.. code-block:: python

   for condition, sub in da.groupby("response"):
       ...

Reduce across dimensions
^^^^^^^^^^^^^^^^^^^^^^^^

Use named reductions (``mean``, ``median``, etc.) to collapse across a named 
dimensions (for example, trial-averaged
time courses or unit-averaged population dynamics). Because reductions are
dimension-aware, they are less error-prone than raw ``numpy`` axis numbers.

.. code-block:: python

   mean_over_trials = da.mean(dim="trial")
   mean_over_time = da.mean(dim="time")

Stack and unstack
^^^^^^^^^^^^^^^^^

Use ``stack`` to reshape multidimensional data into a 2D matrix for methods
that expect samples x features (PCA, regressions, decoders). Use ``unstack``
to map model outputs back into interpretable axes so results can be plotted
and compared in the original trial/time/condition structure.

The ``reduce`` operator in this package performs this stacking automatically.

.. code-block:: python

   da_stack = da.stack(sample=("trial", "time"))
   da_unstack = da_stack.unstack("sample")

Common gotchas
--------------

- Always check ``da.dims`` and ``da.coords`` after transformations to understand the shape of your data.
- Xarray aligns by coordinate labels, not just shape. Two arrays with the same
  length can still mismatch if coordinate values differ.
- ``where`` keeps original shape unless you pass ``drop=True``. If you expected
  fewer trials/time bins, confirm the dimension sizes after filtering.
- ``groupby`` only works on existing coordinates. If your condition is in a
  separate table or NumPy array, attach it first with ``assign_coords``.
- ``stack`` creates a MultiIndex coordinate. Some downstream tools expect plain
  coordinates, so you may need ``reset_index`` or ``unstack`` before export.
- Missing values (NaNs) propagate through many operations by default. Decide
  early whether to fill, mask, or drop NaNs to avoid silent changes in summary
  statistics.
- Coordinate dtype matters for selection. For example, string labels ``"1"``
  do not match integer labels ``1`` in ``sel``/``loc``.
