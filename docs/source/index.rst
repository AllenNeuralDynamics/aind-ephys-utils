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

Installation
############

.. code-block:: bash

   pip install aind-ephys-utils

Example usage (with NWB)
##########################

.. code-block:: python

   from aind_ephys_utils.adapters import from_dataframe
   from pynwb import NWBHDF5IO

   # read the file
   nwb = NWBHDF5IO('/path/to/file.nwb', 'r').read()

   # load units and trials dataframes
   units = nwb.units.to_dataframe()
   trials = nwb.trials.to_dataframe()

   # align all units to all trials in a window
   spikes = from_dataframe(units, trials, window=(-0.5, 1.0))

   # Use the `ephys` accessor to bin the spikes in 0.01 s intervals and smooth
   binned = spikes.ephys.bin(0.01).ephys.smooth(sigma=0.05)

   # plot a PSTH
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
   tutorials/index
   getting_started/troubleshooting
   getting_started/glossary
   api/modules

* :ref:`genindex`
* :ref:`modindex`
