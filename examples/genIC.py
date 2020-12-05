import numpy as np
from nbodykit.lab import *
from fastpm.core import leapfrog, Solver, autostages
from pmesh.pm import ParticleMesh
import nbodykit.cosmology as nbcosmos

from gaussianCR.construct import *
from gaussianCR.cosmo import *
from args import get_args
from saveIC import *


np.set_printoptions(precision=3,linewidth=150,suppress=True)

# ------------------------------------------- 
def print_info(peak_data):    
    print ("Significance = %.2f"%peak_data['nu'])
    print ("dfG/dx, dfG/dy, dfG/dz = ",peak_data['f1'])
    print ("xd = %.2f"%peak_data['xd'],"a12sq = %.2f"%peak_data['a12sq'],"a13sq = %.2f"%peak_data['a13sq'])
    print ("Euler1: a1, b1, p1 = ",peak_data['Euler1'])
    print ("vx,vy,vz (peak velocity in km/s) :",peak_data['v_peculiar'])
    print ("epsilon = %.2f"%peak_data['epsilon'],"omega = %.2f"%peak_data['omega'])
    print ("Euler2: a2, b2, p2 = ",peak_data['Euler2'])


# load args   
# -------------------------------------------
args = get_args()

Rdm_seed = args.Rdm_seed
RG = args.RG
Ng = args.Ng
Lbox = args.Lbox
print ("Seed = ",Rdm_seed)
print ("Lbox = %.1f"%Lbox,"Ng = %d"%Ng,"RG = %.2f"%RG)


# Choose cosmology
# -------------------------------------------
cosmology = nbcosmos.WMAP9
mycosmo = Cosmos(FLRW=True,obj=cosmology)

# generate linear density field at z=0 
# -------------------------------------------
pm = ParticleMesh(BoxSize=Lbox, Nmesh=[Ng,Ng,Ng])
Q = pm.generate_uniform_particle_grid(shift=0)
solver = Solver(pm,cosmology,B=1)

wn = solver.whitenoise(seed = Rdm_seed)
dlin = solver.linear(wn, lambda k: mycosmo.Pk_lin(k))
dx_field = dlin.c2r().value # dx_field is the density contrast field centered at 0


# initialize gsCR object, build xij^{-1} matrix for full 18 constraints
# -------------------------------------------
fg = gsCR(mycosmo,Lbox=Lbox,Nmesh=Ng,RG=RG,CONS=['full'])
fg.build_Xij_inv_matrix()
print ("xij^{-1}:")
print (fg.xij_tensor_inv)
print ("*********************************************")


# set peak position
# -------------------------------------------
if args.xpk_rel[0] != -1:
    print ("read peak position from script:")
    xpk = np.array(args.xpk_rel)*Lbox   # input the peak position
else:
    print ("load the original peak position:")
    xpk = fg.find_xpk(dx_field)

fg.xpk = xpk
print ("Peak position [Mpc/h] ", xpk)


# extract the original pk info
# -------------------------------------------
c_original,peak_original = fg.read_out_c18(dx_field,rpos=xpk)
print ("Before constraint:")
print_info(peak_original)
print ("*********************************************")


# set the target value of the peak
# -----------------------------------------------------
nu = args.significance
xd,a12sq,a13sq = args.compactness,args.a12sq,args.a13sq
a1,b1,p1 = args.Euler1 
vx,vy,vz = args.vp
epsilon,omega = args.epsilon,args.omega
a2,b2,p2 = args.Euler2

c_target = fg.set_c_values(nu,xd,a12sq,a13sq,a1,b1,p1,vx,vy,vz,
                           epsilon,omega,a2,b2,p2,silent=False)


# Get ensemble mean and the constrained field
# -----------------------------------------------------
dc = c_target - c_original
dx_ensemble = fg.Ensemble_field(dc)
dx_constraint = dx_field + dx_ensemble


# Verify from dx_constrained
# ----------------------------------------------------- 
c_result,peak_result = fg.read_out_c18(dx_constraint,rpos=xpk)
print ("*********************************************")
print ("After constraint:")
print_info(peak_result)
print ("*********************************************")


# save IC
# -----------------------------------------------------
state = saveIC_gadget3(args.IC_path,dx_constraint,Lbox,Ng,cosmology,redshift=args.redshift)


# re-Verify by re-sampling the particles to grid
# -----------------------------------------------------
cat1 = ArrayCatalog({'Position': state.X})
mesh = cat1.to_mesh(resampler='pcs',compensated=True,interlaced=True,Nmesh=Ng,position='Position',BoxSize=Lbox)
one_plus_delta = mesh.paint(mode='real')
od = one_plus_delta.value
delta_x_reconstruct = ((od - 1)/mycosmo.D(z=args.redshift)) 

print ("Resample the dx_field from IC:")
c_result,peak_result = fg.read_out_c18(delta_x_reconstruct,rpos=xpk)
print_info(peak_result)
print ("*********************************************")

