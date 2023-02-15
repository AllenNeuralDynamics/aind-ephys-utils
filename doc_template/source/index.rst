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

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   modules


* :ref:`genindex`
* :ref:`modindex`

Motivation
##############

This library is intended to encompass a set of versatile, general-purpose methods for transforming and exploring spike trains and local field potential data. It will be developed in parallel to two other Python packages, :code:`aind-ephys-plots` and :code:`aind-ephys-widgets`, but can also be used on its own.

We hope that these libraries will reduce the amount of redundant code written for exploratory analysis performed on high-density electrophysiology data at the Allen Institute for Neural Dynamics. The code will be public from the start, so others are welcome to contribute to the development as well.

Design principles
##################

Functions added to this library should try to adhere to the following guidelines:

- Their primary inputs/output are numpy :code:`ndarrays`, pandas :code:`DataFrames`, or xarray :code:`DataArrays`. The analysis code shouldn't depend on any library-specific Python classes.
- They have minimal external dependencies. Ideally the library will only require **xarray** (which itself depends on **numpy** and **pandas**), **scipy**, and **scikit-learn**, and perhaps a few other packages.
- They have a variety of use cases. Project-specific code should live in Code Ocean capsules, or eventually separate libraries.

