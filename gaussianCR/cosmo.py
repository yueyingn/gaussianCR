import numpy as np
from scipy import interpolate
from scipy import special
from scipy.special import hyp2f1


class Cosmos(object):
    """
    Collection of some useful variables and functions dependent on cosmology. 
    
    Usage
    ----------
    * if FLRW is True, use obj from astropy.cosmology or nbodykit.cosmology to initialize, 
      and nbodykit.cosmology.LinearPower() to generate Pk_lin (at z=0)
    * if FLRW is not True, need to feed in H0, Om0, Ob0, Pk_lin (at z=0) by hand
    
    Attributes
    ----------
    : obj    : object from astropy.cosmology or nbodykit.cosmology, needed if FLRW is True
    : H0     : Hubble constant in km/s/Mpc, e.g. 69.3
    : Om0    : Total matter fraction, e.g. 0.2865
    : Ob0    : Total baryon fraction, e.g. 0.04628
    : F      : F(Omega) = Om0**0.6
    : Pk_lin : The linear P(k) function at z=0
    
    """
    
    def __init__(self,FLRW=True,obj=None,H0=None,Om0=None,Ob0=None,Pk_lin=None):
        
        if FLRW == True:
            
            if obj is None:
                raise ValueError(("give obj from astropy.cosmology ornbodykit.cosmology if FLRW is True"))
            
            try:
                from nbodykit.cosmology import LinearPower
            except:
                raise ImportError(("Need nbodykit.cosmology.LinearPower"))
                
            self.H0 = obj.h * 100   # km/s/Mpc
            self.Om0 = obj.Om0
            self.Ob0 = obj.Ob0
            self.F = (obj.Om0)**(0.6)
            
            k_space = np.logspace(-4,3,500)
            Plin = LinearPower(obj, redshift=0, transfer='EisensteinHu')
            Pk_space = Plin(k_space)
            self.Pk_lin = interpolate.interp1d(k_space,Pk_space,fill_value="extrapolate")
            
        else:
            
            if any(x is None for x in [H0,Om0,Ob0,Pk_lin]):
                raise Exception("Set H0,Om0,Ob0,Pk_lin by hand if FLRW is not True")
                
            self.H0 = H0
            self.Om0 = Om0
            self.Ob0 = Ob0
            self.F = (Om0)**(0.6)
            self.Pk_lin = Pk_lin  
    
    def Pk_smoothed(self,k,RG):
        """
        Smooth the power spectrum with Gaussian kernel exp(-(kR)^2/2)
        """
        return self.Pk_lin(k)*np.exp(-k*k*RG*RG/2)   
    
    def gs_spectral_moment(self,l,RG,lowk=-4,highk=3):
        """
        return sigma_l(RG)^2 = \int d^3k/(2pi^3) P(k) W(kR)^2 k^(2l)
        """
        kspace = np.logspace(lowk,highk,100000)
        kmid = 10**((np.log10(kspace[:-1])+np.log10(kspace[1:]))/2)
        pkspace = self.Pk_lin(kmid)
        w = np.exp(-kmid*kmid*RG*RG/2)
        integrand = kmid*kmid*pkspace*w*w*(kmid**(2*l))
        sig2 = np.sum(integrand*np.diff(kspace))/(2*np.pi*np.pi)
        return sig2
       
    def D(self,z):
        """
        The linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
        """
        Om = self.Om0
        OL = 1 - Om
        a = 1.0 / (1+z)
        return a * hyp2f1(1, 1/3, 11/6, - OL * a**3 / Om) \
                 / hyp2f1(1, 1/3, 11/6, - OL / Om)
    
    def M_RG(self,RG,kernel='Gaussian'):
        """
        Mass enclosed in a kernel filter with width RG
        RG in unit of Mpc/h, Mass in unit of Msun/h
        """
        Msolar = 1.99*10**30                                       # unit: kg
        Gconstant = 6.67408*10**(-11)                              # unit: m^3*kg^(-1)*s^(-2)
        Mpc = 3.086*10**22                                         # unit: m
        rho_c = (3*100*100*1000*1000/(Mpc*Mpc*8*np.pi*Gconstant))  # unit: h*h*kg/m^3
        rho_0 = (self.Om0*rho_c)*(Mpc**3)/Msolar                   # unit: (Msun/h)/(Mpc/h)**3
        
        if kernel == 'Gaussian':
            fac = (2*np.pi)**(3./2)*rho_0
        elif kernel == 'TopHat':
            fac = (4./3)*np.pi*rho_0
        else:
            raise NotImplementedError('kernel type `{}` not supported'.format(kernel))
        
        return fac*RG**3
    
    
    
