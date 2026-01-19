Design principles
=================

This library is designed to be:

- Composable with xarray: methods return DataArrays or Datasets
- Explicit about dims and metadata
- Small, testable, and easy to extend

Guidelines for contributions
----------------------------

- Use the conventions in :doc:`/concepts/conventions`.
- Prefer pure functions in ``ops`` and keep side effects out.
- Preserve compatible coordinates through ops.
- Keep plotting functions separate from analysis ops.
