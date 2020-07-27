# gaussianCR

Implementation of the constrained realization to Gaussian primordial density fields.

## Usage and tutorial

Check the full documentation [here](https://gaussiancr.readthedocs.io/en/latest/tutorials.html).


Dependencies
------------
It is possible to use this package with only ``numpy`` and ``scipy``, once you manually
feed in the cosmology and the 3D linear density field as numpy array.

To generate gaussian random realization of linear density field, we require
``nbodykit`` and ``fastpm`` installed.

To output IC file compatible with ``MP-Gadget``, we need ``bigfile`` installed.

Install
-----------------
If you intend to install ``gaussianCR`` in editable mode, clone the repository (or your fork of it)::

    git clone https://github.com/yueyingn/gaussianCR.git

Move to the directory and install with::

    pip install -e .
