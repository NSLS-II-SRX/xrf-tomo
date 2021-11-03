========
XRF Tomo
========

.. image:: https://img.shields.io/travis/dmgav/xrf-tomo.svg
        :target: https://travis-ci.org/dmgav/xrf-tomo

.. image:: https://img.shields.io/pypi/v/xrf-tomo.svg
        :target: https://pypi.python.org/pypi/xrf-tomo


XRF Tomography Reconstruction
=============================

Installation
------------

The software is expected to work with Python version 3.7-3.9. Create conda environment
with the preferable python version::

  $ conda create -n xrf-tomo-env python=3.8 -c conda-forge
  $ conda activate xrf-tomo-env

``tomopy`` and ``xraylib`` (dependency of PyXRF) are not available from PyPI and need
to be installed from `conda-forge`::

  $ conda install tomopy pyxrf -c conda-forge

``svmbir`` is an optional dependency. Install ``svmbir`` separately if needed. Instructions
are slightly different depending on OS. Linux::

  $ pip install svmbir

OSX::

  $ ln -sf /usr/local/bin/gcc-10 /usr/local/bin/gcc
  $ CC=gcc pip install --no-binary svmbir svmbir

Windows::

  $ CC=gcc pip install svmbir

Finally install this package. From PyPI::

  $ pip install xrf-tomo

From source (develop install). Clone the repository in the appropriate directory and
then install with ``pip``::

  $ git clone https://github.com/NSLS-II-SRX/xrf-tomo
  $ cd xrf-tomo
  $ pip install -e .

Using the package
-----------------

Activate the environment::

  $ conda activate xrf-tomo-env

In IPython environment or a script import necessary or all functions from the package, e.g. ::

  from xrf_tomo import *
