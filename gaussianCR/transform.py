from numpy import *
import numpy as np

# -------------------------------------------------------------------------------------
# set up transformation matrix

def isclose(x, y, rtol=1e-5, atol=1e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def bound(x,lower=-1,upper=1):
    if x < lower: x = lower
    if x > upper: x = upper
    return x

def set_Aij(a,b,p):
    """
    Transformation matrix of rotation wrt Euler angle {alpha,beta,gamma} (in ZX'Z'' sequence)
    c.f. https://mathworld.wolfram.com/EulerAngles.html
    Note that Aij[1,2] has a sign difference with W&B Eq.66
    """
    Aij = [[ cos(a)*cos(p)-cos(b)*sin(a)*sin(p), sin(a)*cos(p)+cos(b)*cos(a)*sin(p),  sin(b)*sin(p)],
           [-cos(a)*sin(p)-cos(b)*sin(a)*cos(p), -sin(a)*sin(p)+cos(b)*cos(a)*cos(p), sin(b)*cos(p)],
           [     sin(b)*sin(a),                       -sin(b)*cos(a),                      cos(b)]]
    return np.array(Aij)

def Aij_angle(Aij):
    """
    Extract Euler angle {alpha,beta,gamma} from Aij matrix
    Aij_angle(set_Aij) is unitary operation
    """
    beta = np.arccos(Aij[2,2])
    
    if not isclose(sin(beta),0):        
        cosa = bound(Aij[2,1]/(-sin(beta)))
        sina = bound(Aij[2,0]/sin(beta))
        alpha = np.arccos(cosa)
        if sina < 0:      
            alpha = 2*np.pi - alpha

        cosp = bound(Aij[1,2]/(sin(beta)))
        sinp = bound(Aij[0,2]/(sin(beta)))
        phi = np.arccos(cosp)
        if sinp < 0:      
            phi = 2*np.pi - phi
            
    # beta = 0 or 2pi, then alpha,phi rotate on same plane, we can just assume phi=0        
    elif (Aij[0,0] == Aij[1,1]) or (Aij[0,0] == -Aij[1,1]):  
        phi = 0
        cosa = Aij[0,0]
        sina = Aij[0,1]
        alpha = np.arccos(cosa)
        if sina < 0:      
            alpha = 2*np.pi - alpha 
    else:
        raise ValueError(("Problem occurs when solving Aij"))
        
    return alpha,beta,phi

# -------------------------------------------------------------------------------------
# get c5 ~ c10, c14 ~ c18 from physics quantity

def set_fGij(xd,a12sq,a13sq,alpha,beta,phi):
    """
    Calculate 2nd order derivative fG(x)_ij field.
    c.f. W&B eq. 67 - 69 to construct c5~c10 from compactness,ellipcity,and orientation
    xd with xd/sigma2_RG = compactness, a12sq = a12^2 as in the text, but is not necessarily positive if not at the peak
    """
    
    lam1 = xd/(1+a12sq+a13sq)
    lam2 = lam1*a12sq
    lam3 = lam1*a13sq
    lam = np.array([lam1,lam2,lam3])

    Aij = set_Aij(alpha,beta,phi)
    fij = - np.einsum('a,ai,aj -> ij',lam,Aij,Aij)
    return np.array([fij[0,0],fij[1,1],fij[2,2],fij[0,1],fij[0,2],fij[1,2]])


def set_VGij(epsilon,omega,a2,b2,p2):
    """
    Calculate the tidal field EG,ij, Input epsilon is in unit of km/s/Mpc  
    c.f. W&B eq. 89 - 90 to construct c14~c18
    different with W&B eq. 86, I directly use a2,b2,p2 w.r.t the original coordinate \
    (instead of the principal axis of the mass ellipsoid)
    """
    L1 = np.cos((omega+2*np.pi)/3)
    L2 = np.cos((omega-2*np.pi)/3)
    L3 = np.cos((omega)/3)
    Ls = np.array([L1,L2,L3])
    lam = epsilon*Ls

    Tij = set_Aij(a2,b2,p2)
    Eij = np.einsum('k,ki,kj -> ij',lam,Tij,Tij)

    return np.array([Eij[0,0],Eij[1,1],Eij[0,1],Eij[0,2],Eij[1,2]])

# -------------------------------------------------------------------------------------
# extract physics quantify from c5 ~ c10, c14 ~ c18

def extract_ellipsoid_info(c_array):
    """
    inverse of set_fGij, xd,a12sq,a13sq,a,b,p from fG(x)_ij matrix
    recover compactness,ellipcity,and orientation from c5-c10
    """
    fGij = -np.array([[c_array[0], c_array[3], c_array[4]],
                      [c_array[3], c_array[1], c_array[5]],
                      [c_array[4], c_array[5], c_array[2]]])

    w,v = np.linalg.eig(fGij)
    idx = np.argsort(np.abs(w))
    lam = w[idx]
    v = v.transpose()
    Aij = v[idx]

    # enforce alpha,beta,phi [0,pi] and right-hand 
    if Aij[0,2]<0:
        Aij[0]*=(-1)
    if Aij[2,0]<0:
        Aij[2]*=(-1)
    if np.linalg.det(Aij)<0:
        Aij[1]*=(-1)

    xd = np.sum(lam)
    a12sq = (lam[1]/lam[0])
    a13sq = (lam[2]/lam[0])

    a,b,p = Aij_angle(Aij)
    param_recover = np.array([xd,a12sq,a13sq,a,b,p])

    return param_recover


def extract_tidal_info(c_array):
    """
    Inverse of set_VGij, recover epsilon,omega,a,b,p from the tidal field matrix
    """
    tidal_tensor = np.array([[c_array[0], c_array[2], c_array[3]],
                             [c_array[2], c_array[1], c_array[4]],
                             [c_array[3], c_array[4], - c_array[0] - c_array[1]]])

    w,v = np.linalg.eig(tidal_tensor)

    # the correct order of lambda
    epsilon = np.linalg.norm(w)/1.225
    idx = np.argsort(-np.arccos(w/epsilon))
    idx[1],idx[2] = idx[2],idx[1]

    lam = w[idx]
    v = v.transpose()
    Aij = v[idx]

    omega = np.arccos(lam[2]/epsilon)*3

    # enforce alpha,beta,phi [0,pi] and right-hand 
    if Aij[0,2]<0:
        Aij[0]*=(-1)
    if Aij[2,0]<0:
        Aij[2]*=(-1)
    if np.linalg.det(Aij)<0:
        Aij[1]*=(-1)

    a,b,p = Aij_angle(Aij)
    param_recover = np.array([epsilon,omega,a,b,p])

    return param_recover
#-------------------------------------------------------------------------------------




