.. Doc Template documentation master file, created by
   sphinx-quickstart on Wed Aug 17 15:36:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


:code:`aind-ephys-utils`
===========================================

.. image:: https://img.shields.io/badge/license-MIT-brightgreen
.. image:: https://img.shields.io/badge/code%20style-black-black
.. image:: https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release

Helpful methods for exploring *in vivo* electrophysiology data.

.. image:: /_static/aind-ephys-utils.png
   :alt: DataArray with trial, event, and bound coords.
   :width: 70%


Installation
############

.. code-block:: bash

   pip install aind-ephys-utils

Key concepts
############

Analysis happens on Xarray ``DataArray`` objects with labeled dimensions and coordinates, via the ``ephys`` accessor:
- ``da.ephys.align(...)``
- ``da.ephys.reduce(...)``
- ``da.ephys.psth(...)``
- ``da.ephys.plot.raster(...)``

This allows functions to be run in sequence and combined with built-in Xarray functions, e.g.:

.. code-block:: python

   da.ephys.align(...).sel(unit=1).mean('trial').ephys.smooth(...)


Example usage (with NWB)
##########################

.. code-block:: python

   from aind_ephys_utils import from_dataframe
   from pynwb import NWBHDF5IO

   # read the file
   nwb = NWBHDF5IO('/path/to/file.nwb', 'r').read()

   # load units and trials dataframes
   units = nwb.units.to_dataframe()
   trials = nwb.trials.to_dataframe()

   # align all units to all trials in a specific time window
   spikes = from_dataframe(units, trials, window=(-0.5, 1.0))

   # plot a spike raster for one unit, grouped by the value in the "choice" column:
   ax = spikes.sel(unit=1).ephys.plot.raster(group_by="choice")

   # bin the spikes in 0.01 s intervals and smooth
   binned = spikes.ephys.bin(0.01).ephys.smooth(window=0.05)

   # plot a PSTH for all units and conditions:
   ax = binned.ephys.plot.psth()

Next steps
##########

- :doc:`/getting_started/quickstart`
- :doc:`/concepts/index`
- :doc:`/ops/index`

Documentation
#############

.. toctree::
   :maxdepth: 2

   getting_started/quickstart
   concepts/index
   getting_started/design_principles
   adapters/index
   ops/index
   plots/index
   api/modules

* :ref:`genindex`
* :ref:`modindex`
