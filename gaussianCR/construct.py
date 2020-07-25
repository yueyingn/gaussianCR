import numpy as np
from .transform import *
from .kernel import *
from .cosmo import *


class gsCR(object):
    """
    Implementation of Gaussian Constrained Realization from van de Weygaert & Bertschinger 1996 (W&B)
    Construct ensemble mean field in simulation box with boxsize Lbox, smoothing scale RG, 
    peak position xpk and multiple CONS options.
    """
    
    CONS_avaliable = ['full','f0','f1','f2','vx','vy','vz','TG']
    
    def __init__(self, 
                 cosmology, 
                 Lbox=20, 
                 Nmesh=128,
                 RG=0.9, 
                 xpk=[0,0,0],
                 CONS = ['full']):
        
        """
        Parameters
        ----------
        cosmology : Cosmos object
        Lbox      : Boxsize in Mpc/h
        RG        : Gaussian kernel radius in Mpc/h
        Nmesh     : Number of mesh per side 
        xpk       : peak position in Mpc/h
        
        
        CONS options
        ----------
        full      : enable all the 18 constraints at position xpk
        f0        : H1, constrain zeroth order of fG field
        f1        : H2~H4, the three 1st order derivatives of fG field, 
        f2        : H5~H10, 2nd order derivatives of fG field, fG2,ij is 3x3 symetric matrix
        vx,vy,vz  : H11~H13, constrain the three peculiar velocities of fG field
        TG        : H14~H18, the tidal field of fG field, TG,ij is 3x3 traceless matrix
        
        """
        
        self.attrs = {}
        self.attrs['Lbox'] = Lbox
        self.attrs['Nmesh'] = Nmesh
        self.attrs['xcellsize'] = self.attrs['Lbox']/self.attrs['Nmesh']
        self.attrs['kcellsize'] = 2*np.pi/self.attrs['Lbox']
        
        self.RG = RG
        self.xpk = xpk
        
        if not isinstance(cosmology,Cosmos):
            raise ValueError(("{} should be object of Cosmos".format(cosmology)))
        self.cosmo = cosmology
        
        if np.any([n not in self.CONS_avaliable for n in CONS]):
            raise ValueError(("Contain available flags for peak constraint"))
        self.CONS = CONS
        
        self.xij_tensor_inv = None
        
    
    def __str__(self):
        s1 = "  This is a gsCR object: \n"
        s2 = "  Lbox = %.1f Mpc/h \n" % self.attrs['Lbox']
        s3 = "  Nmesh = %d\n" % self.attrs['Nmesh']
        s4 = "  RG = %.1f Mpc/h \n" % self.RG
        s5 = "  Sigma0_RG = {:.2f}, Sigma2_RG = {:.2f} \n".format(self.sigma0_RG,self.sigma2_RG)
        s6 = "  xpk = {} \n".format(self.xpk)
        s7 = "  CONS = {} \n".format(self.CONS)
        s8 = "  xij_tensor_inv = \n {} ".format(self.xij_tensor_inv)
        return "{}{}{}{}{}{}{}{}".format(s1, s2, s3, s4, s5, s6, s7, s8)
    
    
    @property
    def xpk(self):
        return self._xpk
    
    @xpk.setter
    def xpk(self,value):
        if np.any([(x<0 or x>self.attrs['Lbox']) for x in value]):
            raise ValueError("peak position exceed boxsize")
        self._xpk = value  
        
    @property
    def sigma0_RG(self): 
        """
        Variance of the peak height, the unit of peak parameter nu
        """
        sigma = np.sqrt(self.cosmo.gs_spectral_moment(l=0,RG=self.RG))
        return sigma
    
    @property
    def sigma2_RG(self): 
        """
        Variance of the \laplace f_G, the unit of peak parameter xd
        """
        sigma = np.sqrt(self.cosmo.gs_spectral_moment(l=2,RG=self.RG))
        return sigma
    
    @property
    def cmask(self):
        """
        mask to select a subset of the full 18 constraints
        """
        mask = np.zeros(18)
        if 'full' in self.CONS: mask[:] = 1
        if 'f0' in self.CONS: mask[0] = 1
        if 'f1' in self.CONS: mask[1:4] = 1
        if 'f2' in self.CONS: mask[4:10] = 1
        if 'vx' in self.CONS: mask[10] = 1
        if 'vy' in self.CONS: mask[11] = 1
        if 'vz' in self.CONS: mask[12] = 1
        if 'TG' in self.CONS: mask[13:18] = 1
        return mask>0
    

    def build_Xij_inv_matrix(self,Nmesh=64):
        """
        Construct the covariance matrix for a subset of the 18-constriants.
        Used to construct the Ensemble mean field
        : Nmesh : Nmesh to calculate xij, can be smaller than self.Nmesh
                  actually Nmesh > 2*Lbox/RG is good enough for xij calculation
        """
        H0, F = self.cosmo.H0, self.cosmo.F
        Lbox = self.attrs['Lbox']
        kgrid = initialize_kgrid(Nmesh,Lbox)
        kmag_grid = np.linalg.norm(kgrid,axis=3)
        w_grid = self.cosmo.Pk_lin(kmag_grid)*(1/Lbox**3)*np.exp(-kmag_grid*kmag_grid*self.RG*self.RG)
        k2 = kmag_grid**2
        k2[0,0,0] = 1 
        #----------------------------------------------------
        cspace = np.arange(0,18)
        
        xij_tensor = [[np.sum(np.conj(Hhats[i](kgrid,k2,H0,F))*Hhats[j](kgrid,k2,H0,F)*w_grid)
                   for j in cspace[self.cmask]] for i in cspace[self.cmask]]
        
        xij_tensor = np.array(xij_tensor)
        self.xij_tensor_inv = np.linalg.inv(xij_tensor.real)
    
    
    def Ensemble_field(self,c_value):
        """
        populate the k space according to W&B Eq 44 and return the dx_field, 
        Note that the phase has a minus sign (opposite to W&B) because np.ifft has exp(+ikx) convention
        : c_value : the subset of c1~c18, should have the same length as cmask
        : return  : dx_field of the ensemble mean field
        """
        if (np.sum(self.cmask) != len(c_value)):
            raise ValueError("weights should have the same size as cmask")
        
        Lbox,RG,xpk = self.attrs['Lbox'],self.RG,self.xpk
        H0,F = self.cosmo.H0,self.cosmo.F
        
        if self.xij_tensor_inv is None:
            print ("Building Xij matrix ...")
            self.build_Xij_inv_matrix()   
        
        kgrid = initialize_kgrid(self.attrs['Nmesh'],Lbox)
        kmag_grid = np.linalg.norm(kgrid,axis=3)
        k2 = kmag_grid**2
        k2[0,0,0]=1
        
        # ----------------------------------------------------
        phase = -np.sum(kgrid*xpk,axis=3) 
        ampl_field = self.cosmo.Pk_smoothed(kmag_grid,RG)*(1/Lbox**3)
        dk_field = np.complex128(np.zeros_like(ampl_field))
        
        weights = np.einsum('ij,j->i',self.xij_tensor_inv,c_value)
        
        cspace = np.arange(0,18)
        for w,i in zip(weights,cspace[self.cmask]):
            dk_field += w * Hhats[i](kgrid,k2,H0,F) * ampl_field * np.exp(1j*phase)

        dk_field[0,0,0] = 0.0  # Note that the mean overdensity of the box should be zero
        
        #----------------------------------------------------       
        dx_field = (self.attrs['Nmesh']**3)*np.fft.ifftn(dk_field).real

        return dx_field
    
    
    def set_c_values(self,nu=3,xd=2,a12sq=1,a13sq=1,a1=0,b1=0,p1=0,
                          vx=0,vy=0,vz=0,epsilon=0.,omega=1.5*np.pi,
                          a2=np.pi,b2=0*np.pi,p2=0.5*np.pi,silent=False):
        """
        Set up c_values transformed from the peak parameters and masked by CONS flags
        : nu       : peak height, in unit of sigma0_RG
        : xd       : peak compactness, in unit of sigma2_RG
        : a12sq    : axial ratio (a1/a2)**2
        : a13sq    : axial ratio (a1/a3)**2
        : a1,b1,p1 : Euler angle to transform to principal axis of mass ellipsoid
        : vx,vy,vz : Peculiar velocity of the peak in unit km/s
        : epsilon  : Shear magnitude in unit km/s/Mpc
        : omega    : Shear angle to distribute the shear magnitude between three axes, [pi,2pi]
        : a2,b2,p2 : Euler angle to transform to principal axis of tidal tensor
        : silent   : flag to print the relevant peak parameters
        : return   : c_values in same size as cmask
        """
        
        c_target = np.zeros(18)
        c_target[0] = nu*self.sigma0_RG  
        xd_abs = xd*self.sigma2_RG
        c_target[4:10] = set_fGij(xd_abs,a12sq,a13sq,a1,b1,p1)
        c_target[10:13] = np.array([vx,vy,vz])
        c_target[13:18] = set_VGij(epsilon,omega,a2,b2,p2)
        
        if silent == False:
            print ("Constrain peak parameters: ")
            if 'f0' in self.CONS or 'full' in self.CONS: 
                print ("f0: ","nu = %.1f"%nu, "$\sigma_0$")
            if 'f1' in self.CONS or 'full' in self.CONS: 
                print ("f1: ","{f1,x = f1,y = f1,z = 0")
            if 'f2' in self.CONS or 'full' in self.CONS: 
                print ("f2: ",r"xd = {:.1f} $\sigma_2$, a12sq = {:.1f}, a13sq = {:.1f},a1={:.2f}, b1={:.2f}, p1={:.2f}".format(xd,a12sq,a13sq,a1,b1,p1))
            if 'vx' in self.CONS or 'full' in self.CONS: 
                print ("vx = {:.1f} km/s".format(vx))  
            if 'vy' in self.CONS or 'full' in self.CONS: 
                print ("vy = {:.1f} km/s".format(vy))
            if 'vz' in self.CONS or 'full' in self.CONS: 
                print ("vz = {:.1f} km/s".format(vz))
            if 'TG' in self.CONS or 'full' in self.CONS: 
                print ("TG: ","epsilon = {:.1f} km/s/Mpc, omega = {:.2f}, a2={:.2f}, b2={:.2f}, p2={:.2f}".format(epsilon,omega,a2,b2,p2))
        
        return c_target[self.cmask]
        
        
    
    def read_out_c18(self,dx_field,rpos=None):
        """
        c.f. W&B Eq.38, convolve the field with Hhats and read out the c values at rpos
        : dx_field  : with shape (Ng,Ng,Ng), the density contrast field (averaged = 0), Lbox should match self.Lbox
        : rpos      : the position to read out c values, if not set, then use self.xpk
        : return    : [c1~c18], c1 with c1/sigma0_RG = nu, c11~13 in unit of km/s, c14~c18 in unit of km/s/Mpc
        """  
        
        reps = np.shape(dx_field)
        if not (reps[0]==reps[1]==reps[2]):
            raise Exception( "dx field should in shape of (Ng,Ng,Ng)" )
        
        if rpos is None:
            rpos = self.xpk
        
        Lbox,RG = self.attrs['Lbox'],self.RG
        H0,F = self.cosmo.H0, self.cosmo.F
            
        xcellsize = Lbox/reps[0]
        pkidx = (int(rpos[0]/xcellsize),int(rpos[1]/xcellsize),int(rpos[2]/xcellsize))
        
        kgrid = initialize_kgrid(reps[0],Lbox)
        kmag_grid = np.linalg.norm(kgrid,axis=3)
        k2 = kmag_grid**2
        k2[0,0,0]=1

        dk_field = (1/reps[0]**3)*np.fft.fftn(dx_field)
        wdk_field = np.exp(-kmag_grid*kmag_grid*RG*RG*0.5)*dk_field

        cs = np.zeros(18)

        for i in range(0,18):
            dk_smoothed = np.conj(Hhats[i](kgrid,k2,H0,F))*wdk_field
            cs[i] = reps[0]**3*np.fft.ifftn(dk_smoothed).real[pkidx]

        return cs
    
    
    def find_xpk(self,dx_field):
        """
        convolve dx_field with H1 and find the peak position
        : dx_field  : with shape (Ng,Ng,Ng), the density contrast field (averaged = 0), Lbox should match self.Lbox
        : return    : xpk in Mpc/h
        """
        
        reps = np.shape(dx_field)
        if not (reps[0]==reps[1]==reps[2]):
            raise Exception( "dx field should in shape of (Ng,Ng,Ng)" )
            
        Lbox,RG = self.attrs['Lbox'],self.RG 
        xcellsize = Lbox/reps[0]
        
        dx_smoothed = H1_smooth(dx_field,Lbox,RG)
        pkidx = np.unravel_index(np.argmax(dx_smoothed),np.shape(dx_field))
        xpk = np.asarray(pkidx)*xcellsize
        
        return xpk

    

#------------------------------------------- 
def initialize_kgrid(Nmesh,Lbox):
    """
    prepare k_grid in shape of (Nmesh,Nmesh,Nmesh,3), with kcellsize = 2pi/Lbox
    """
    ksize = (2*np.pi/Lbox)
    reps = (Nmesh,Nmesh,Nmesh)

    kx_space = np.arange(0,reps[0])
    kx_space -= (kx_space>reps[0]/2)*reps[0]     # see also pmpfft.c/pm_init/pm->MeshtoK
    kx_space = kx_space*ksize

    ky_space = np.arange(0,reps[1])
    ky_space -= (ky_space>reps[1]/2)*reps[1]
    ky_space = ky_space*ksize

    kz_space = np.arange(0,reps[2])
    kz_space -= (kz_space>reps[2]/2)*reps[2]
    kz_space = kz_space*ksize

    # ----------------------------------------------------
    kgrid = np.zeros((reps[0],reps[1],reps[2],3))
    kgrid[...,0] += kx_space.reshape(-1,1,1)
    kgrid[...,1] += ky_space.reshape(-1,1)
    kgrid[...,2] += kz_space

    return kgrid


#------------------------------------------- 
def H1_smooth(dx_field,Lbox,RG):
    """
    Smooth the dx_field with Gaussian kernel H1, 
    Can be used to quickly read out the original peak position at scale RG.
    """
    reps = np.shape(dx_field)
    kgrid = initialize_kgrid(reps[0],Lbox)
    kmag_grid = np.linalg.norm(kgrid,axis=3)
    k2 = kmag_grid**2
    k2[0,0,0]=1

    dk_field = (1/reps[0]**3)*np.fft.fftn(dx_field)
    wdk_field = np.exp(-kmag_grid*kmag_grid*RG*RG*0.5)*dk_field
    
    dk_smoothed = 1*wdk_field
    dx_smooth = (reps[0]**3)*np.fft.ifftn(dk_smoothed).real

    return dx_smooth
    
