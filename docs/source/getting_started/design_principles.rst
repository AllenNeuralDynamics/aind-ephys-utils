Design principles
=================

This library is designed to be:

- **Composable**: all operations return ``DataArrays`` or ``Datasets`` and can be chained with ``.sel``, ``.mean``, ``.groupby``, and other built-in Xarray methods.
- **Minimal**: small functions, predictable outputs, no hidden
  global state.
- **Practical**: common analysis workflows are short and
  readable.

Package layers
--------------

- ``accessors``: ergonomic entry point via ``da.ephys``
- ``adapters``: convert external data into canonical ``DataArrays``
- ``standards``: conventions and validation of shapes/metadata
- ``ops``: pure, composable operations (no plotting, no I/O)
- ``plots``: code for visualization
- ``metrics``: (coming soon)

Data model assumptions
----------------------

- Ragged spikes are stored as object arrays with dims ``(trial, unit)`` or simply ``(unit,)``.
- Binned spikes are numeric with dims ``(trial, unit, time)``.

Coords and attributes are first‑class
-------------------------------------

- Trial metadata should be stored as coordinates with ``dims=("trial",)``.
- Unit metadata should be stored as coordinates with ``dims=("unit",)``.
- Ops should preserve compatible coordinates whenever possible.
- Always update or preserve ``ephys.timebase`` and ``ephys.valid_intervals``.

Error handling
--------------

- Prefer early validation with helpful error messages.
- Raise explicit errors when dims or coords are missing.

Extensibility
-------------

- New ops should take a ``DataArray`` and return a ``DataArray`` or ``Dataset``.
- Avoid side effects and mutable globals.
- Keep plotting separate from computation.

Guidelines for contributions
----------------------------

- Use the conventions in :doc:`/concepts/conventions`.
- Prefer pure functions in ``ops`` and keep side effects out.
- Preserve compatible coordinates through ops.
- Keep plotting functions separate from analysis ops.
- Add tests for new behavior and include minimal examples in the docs.
