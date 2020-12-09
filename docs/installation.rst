Installation
============

This page will guide you through installing ``gaussianCR``.

Dependencies
------------
``gaussianCR`` operates on any given linear density field. It is possible to use this package with only ``numpy`` and ``scipy``, once you manually feed in the cosmology and the 3D linear density field as numpy array.

If one wants to quickly generate a realization of Gaussian random field to play with, I recommend to use ``nbodykit`` and ``fastpm``, as shown in the tutorial/Construct-constrained-realization.

You might need to write your own interface to output the constrained-IC for your simulation.
In the `/examples`, we output the constrained-IC compatible with ``MP-Gadget``, which need ``bigfile`` installed. 

Install
-----------------
If you intend to install ``gaussianCR`` in editable mode, clone the repository (or your fork of it):

    git clone https://github.com/yueyingn/gaussianCR.git

Move to the directory and install with:

    pip install -e .
