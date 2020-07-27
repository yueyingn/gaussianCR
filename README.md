# gaussianCR

This is an implementation of the constrained realization to Gaussian primordial density fields, the theoritical formalism is developed by van de Weygaert & Bertschinger 1996. 

Features
------------
* A general tool to impose constraints to the linear density field.
* Support the full 18 constraints simultaneously applied to the density field, so that one can impose peak, gravity and tidal field constraints to any Gaussian random field.
* Convolution-type constraints with a Gaussian kernel, one can specify the size of the Gaussian kernel $R_G$ to select the scale upon which to impose the constraints.
* For now, we only support imposing the sets of constraints to 1 position in the density field at each time.  To obtain two separate density peaks, one need to construct one after another. 


Usage and tutorial
------------

[Read the docs here.](https://gaussiancr.readthedocs.io/en/latest/tutorials.html)


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
