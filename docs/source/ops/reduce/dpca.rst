dpca
====

Demixed PCA for condition-aware low-dimensional structure.

Use ``method="dpca"`` when you want components aligned to experimental labels such as choice, stimulus, or task context.

When to choose this method
--------------------------

Choose dPCA when you need interpretable components tied to known task
variables.

- Separates shared population variance into label-specific and interaction
  components (for example stimulus, choice, and stimulus-choice interaction).
- Useful for understanding *which variable* drives each neural subspace.
- A strong choice when your dataset has clear factorial structure.

dPCA depends on reliable condition averages. It is less appropriate when trial counts per condition are very small or strongly imbalanced.

.. code-block:: python

   ds = da.ephys.reduce(
       method="dpca",
       labels=["choice", "stimulus"],
       n_components=6,
   )

Unlike other methods for dimensionality reduction, the ``projections`` ``DataArray`` will have an additional ``marginal`` axis for each label and their interactions (e.g. ``choice``, ``time``, ``choice_time``).

Details from the original dPCA paper
------------------------------------

Kobak et al. (2016) developed dPCA to address a common issue in higher cortical datasets: single neurons often show mixed selectivity, and standard PCA compresses activity without clearly separating *which task variable* drives each component.

Key ideas emphasized in the paper:

- dPCA aims for two goals at once: preserve variance (as in PCA) and improve
  interpretability with respect to task parameters.
- It decomposes population activity into marginalized components associated with specific factors (for example stimulus, decision, and interactions).
- The paper presents a formulation with an analytical solution, making the
  method practical and stable for typical neuroscience use cases.
- Across multiple datasets, dPCA produced compact visual summaries that
  separated task-related signals more clearly than PCA alone.

In practice, this makes dPCA a good default when your experiment has explicit
condition structure and your main question is how different task variables are
represented in population activity over time.

Notes
-----

- ``labels`` is required.
- By default, ``trial_average=True`` and the method averages over ``trial``
  before demixing.

Reference
---------

Kobak D, Brendel W, Constantinidis C, Feierstein CE, Kepecs A, Mainen ZF,
Qi X-L, Romo R, Uchida N, Machens CK (2016). Demixed principal component
analysis of neural population data. *eLife* 5:e10989. `doi:10.7554/eLife.10989 <https://doi.org/10.7554/eLife.10989>`__
