"""
The 18 peak constraint kernels 
c.f. W&B 1998 Appendix F
In our one peak constraint, H1 ~ H18 share the same RG and rpk, 
therefore the factor of exp(-k^2*RG^2/2)*exp(-ik \dot rpk) is took out and 
only calculated once in gsCR.Ensemble_field
"""

# zeroth order of fG field
# ------------------------------------------- 
def H1(kgrid,k2,H0,F):
    return 1

# first order derivatives of fG
# ------------------------------------------- 
def H2(kgrid,k2,H0,F):
    return -1j*kgrid[...,0]
def H3(kgrid,k2,H0,F):
    return -1j*kgrid[...,1]
def H4(kgrid,k2,H0,F):
    return -1j*kgrid[...,2]

# second order derivatives of fG, control shape,compactness,orientation of the peak
# ------------------------------------------- 
def H5(kgrid,k2,H0,F):
    return -kgrid[...,0]*kgrid[...,0]
def H6(kgrid,k2,H0,F):
    return -kgrid[...,1]*kgrid[...,1]
def H7(kgrid,k2,H0,F):
    return -kgrid[...,2]*kgrid[...,2]
def H8(kgrid,k2,H0,F):
    return -kgrid[...,0]*kgrid[...,1]
def H9(kgrid,k2,H0,F):
    return -kgrid[...,0]*kgrid[...,2]
def H10(kgrid,k2,H0,F):
    return -kgrid[...,1]*kgrid[...,2]


# pecular velocity of the peak, in unit of km/s
# ------------------------------------------- 
def H11(kgrid,k2,H0,F):
    return H0*F*(-1j*kgrid[...,0]/k2)
def H12(kgrid,k2,H0,F):
    return H0*F*(-1j*kgrid[...,1]/k2)
def H13(kgrid,k2,H0,F):
    return H0*F*(-1j*kgrid[...,2]/k2)


# tidal field (shear field) of the peak, in km/s/Mpc
# ------------------------------------------- 
def H14(kgrid,k2,H0,F):
    return -H0*F*(kgrid[...,0]*kgrid[...,0]/k2 - 1./3)
def H15(kgrid,k2,H0,F):
    return -H0*F*(kgrid[...,1]*kgrid[...,1]/k2 - 1./3)
def H16(kgrid,k2,H0,F):
    return -H0*F*(kgrid[...,0]*kgrid[...,1]/k2)
def H17(kgrid,k2,H0,F):
    return -H0*F*(kgrid[...,0]*kgrid[...,2]/k2)
def H18(kgrid,k2,H0,F):
    return -H0*F*(kgrid[...,1]*kgrid[...,2]/k2)


Hhats = [H1,H2,H3,H4,H5,H6,H7,H8,H9,H10,H11,H12,H13,H14,H15,H16,H17,H18]

