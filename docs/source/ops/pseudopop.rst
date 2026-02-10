pseudopop
=========

Build a pseudopopulation by combining units across sessions after
condition-wise trial averaging.

What this operation means
-------------------------

``pseudopop`` takes a list of session ``DataArray`` objects and, for each
session:

1. Computes a PSTH grouped by one or more trial-condition coordinates
   (``group_by``).
2. Concatenates the resulting condition-averaged units across sessions along the unit dimension.
3. Adds a ``session`` coordinate on the unit dimension so each unit retains
   session provenance.

This is useful when no single recording contains enough units, and you want a single condition-aligned population representation for downstream analyses (for example PCA, dPCA, GPFA, or decoding on condition means).

Example
-------

.. code-block:: python

   from aind_ephys_utils import pseudopop

   pp = pseudopop(
       [session_a, session_b, session_c],
       group_by="choice",
   )

   # pp has units from all sessions and a condition dimension ("choice")
   # plus a "session" coordinate attached to each unit.

Notes
-----

- Input arrays should share compatible condition definitions used in
  ``group_by``.
- Trial-level variability is intentionally removed by averaging within each condition before concatenation.
- If ``session_ids`` is omitted, session labels default to ``s0``, ``s1``,
  ...
