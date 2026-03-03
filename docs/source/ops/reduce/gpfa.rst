gpfa
====

Gaussian Process Factor Analysis for latent trajectory extraction.

Use ``method="gpfa"`` on pre-binned trial data to recover smooth latent
dynamics.

When to choose this method
--------------------------

Choose GPFA when temporal structure is central to your question.

- Designed for trajectory recovery: it models smooth latent dynamics over time,
  not just static variance structure.
- Often preferred over PCA for single-trial dynamics, state-space geometry, and
  time-evolving neural computations.
- Useful when you expect latent processes to be continuous and noisy.

Compared with PCA, GPFA is usually slower and can be more sensitive to
binning, trial length, and optimization settings.

Details from the original GPFA paper
------------------------------------

Yu et al. (2009) introduced GPFA to address a limitation of two-stage pipelines
(temporal smoothing first, then static dimensionality reduction). In GPFA,
smoothing and dimensionality reduction are solved together in one probabilistic model, which tends to produce cleaner single-trial trajectories.

Important practical points from the paper:

- GPFA is designed for *single-trial* latent trajectory recovery, not only
  condition averages.
- Latent dimensions have explicit Gaussian process timescales, which control smoothness and help preserve temporal structure.
- Model quality was evaluated with a neuron-prediction metric (predict each neuron from the others), and GPFA improved prediction relative to the tested two-stage methods.
- In the motor-cortex delayed-reach dataset, trajectories were observed to converge during motor planning, supporting a dynamical-systems  interpretation of population activity.

In the ``aind-ephys-utils`` implementation, GPFA operates on pre-binned data and returns latent projections and weights in an Xarray-friendly format.

.. code-block:: python

   ds = da.ephys.reduce(
       method="gpfa",
       n_components=3,
   )

Optional GPFA behavior can be configured through ``gpfa_options``.

.. code-block:: python

   ds = da.ephys.reduce(
       method="gpfa",
       gpfa_options={
           "max_iters": 200,
           "freq_ll": 5,
           "fast_mode": True,
       },
   )

Reference
---------

Yu BM, Cunningham JP, Santhanam G, Ryu SI, Shenoy KV, Sahani M (2009).
Gaussian-process factor analysis for low-dimensional single-trial analysis of
neural population activity. *Journal of Neurophysiology*, 102(1), 614-635. `doi:10.1152/jn.90941.2008 <https://pmc.ncbi.nlm.nih.gov/articles/PMC2712272/>`__

