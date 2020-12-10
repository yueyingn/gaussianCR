# gaussianCR

This is an implementation of the constrained realization to Gaussian primordial density fields, based on the theoritical formalism introduced by van de Weygaert & Bertschinger 1996. 

Features
------------
* A general tool to impose constraints to the linear density field.
* Supports the full 18 constraints simultaneously applied to the density field. One can control the height and shape of peaks in the Gaussian random field, as well as constrain the peculiar velocity and tidal field at the site of the peak.
* Convolution-type constraints with a Gaussian kernel. One need to specify the size of the Gaussian kernel RG to select the scale upon which to impose the constraints.
* For now, we only support imposing multiple constraints on 1 density peak covariantly.  To construct separate density peaks, one might need to constrain the peaks one after another. 


Usage and tutorial
------------

[Read the docs here.](https://gaussiancr.readthedocs.io/en/latest/tutorials.html)


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


References
------------------
If you find this code useful in your research, please cite the associated paper [Ni et al. (2020)](https://arxiv.org/abs/2012.04714) and the original theoritical paper [van de Weygaert & Bertschinger (1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.281...84V/abstract). Please also consider starring the GitHub repository.

Questions?
------------------
If you have any questions or would like to contribute to this code, please open an issue or email to yueyingn@andrew.cmu.edu directly.


