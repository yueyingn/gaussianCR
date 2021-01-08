"""
This module provides the interface to convert the density field to the IC output which is compatitable with MP-Gadget IC in BigFile format.

"""

import numpy as np
from bigfile import FileMPI
from nbodykit.lab import *
from fastpm.core import leapfrog, Solver, autostages
from pmesh.pm import ParticleMesh

#-----------------------------------------------------------------------------------------
def periodic_wrap(pos,Lbox):
    pos[pos>Lbox] -= Lbox
    pos[pos<0] += Lbox
    return pos


#-----------------------------------------------------------------------------------------
def saveIC_gadget3(IC_path,dx_field,Lbox,Ng,cosmology,redshift=99):
    """
    Use Zel-dovich approximation to back-scale the linear density field to initial redshift,
    paint baryon and dm particles from the grid,
    and save initial condition in MP-Gadget3 format
    
    
    Parameters
    ---------
    : dx_field  : linear density field at z=0
    : Lbox      : BoxSize, in Mpc/h, will be converted to kpc/h in the IC output
    : Ng        : Number of grid on which to paint the particle
    : cosmology : object from nbodykit.cosmology, or astropy.cosmology.FLRW
    : redshift  : redshift of the initial condition
    """
    
    mesh = ArrayMesh(dx_field, BoxSize=Lbox)  # density contrast field centered at zero
    dk_field = mesh.compute(mode='complex')

    shift_gas =  - 0.5 * (cosmology.Om0 - cosmology.Ob0) / cosmology.Om0
    shift_dm = 0.5 * cosmology.Ob0 / cosmology.Om0
    
    pm = ParticleMesh(BoxSize=Lbox, Nmesh=[Ng,Ng,Ng])
    solver = Solver(pm,cosmology,B=1)
    
    Q_gas = pm.generate_uniform_particle_grid(shift=shift_gas)
    Q_dm = pm.generate_uniform_particle_grid(shift=shift_dm)
    
    scale_a = 1./(1+redshift)
    state_gas = solver.lpt(dk_field, Q_gas, a=scale_a, order=1) # order = 1 : Zel-dovich
    state_dm = solver.lpt(dk_field, Q_dm, a=scale_a, order=1)
    
    state = state_dm
    with FileMPI(state.pm.comm, IC_path, create=True) as ff:

        m0 = state.cosmology.rho_crit(0)*state.cosmology.Omega0_b*(Lbox**3)/state.csize
        m1 = state.cosmology.rho_crit(0)*(state.cosmology.Om0-state.cosmology.Omega0_b)*(Lbox**3)/state.csize

        with ff.create('Header') as bb:
            bb.attrs['BoxSize'] = Lbox*1000
            bb.attrs['HubbleParam'] = state.cosmology.h
            bb.attrs['MassTable'] = [m0, m1, 0, 0, 0, 0]

            bb.attrs['Omega0'] = state.cosmology.Om0
            bb.attrs['OmegaBaryon'] = state.cosmology.Omega0_b
            bb.attrs['OmegaLambda'] = state.cosmology.Omega0_lambda

            bb.attrs['Time'] = state.a['S']
            bb.attrs['TotNumPart'] = [state.csize, state.csize, 0, 0, 0, 0]
            bb.attrs['UsePeculiarVelocity'] = 1  

        ff.create_from_array('1/Position', 1000*periodic_wrap(state.X,Lbox)) # in kpc/h
        ff.create_from_array('1/Velocity', state.V) # Peculiar velocity in km/s
        dmID = np.arange(state.csize)
        ff.create_from_array('1/ID', dmID)

        # save gas
        ff.create_from_array('0/Position', 1000*periodic_wrap(state_gas.X,Lbox))
        ff.create_from_array('0/Velocity', state_gas.V)    
        gasID = np.arange(state.csize,2*state.csize)
        ff.create_from_array('0/ID', gasID) 

    print ("IC generated!")
    print ("*********************************************")
    return state
    
    
#-----------------------------------------------------------------------------------------
def saveIC_bt2(IC_path,dx_field,Lbox,Ng,cosmology,redshift=99):
    """
    Use Zel-dovich approximation to back-scale the linear density field to initial redshift,
    paint baryon and dm particles from the grid,
    and save initial condition in MP-Gadget/bluetides-ii format
    
    
    Parameters
    ---------
    : dx_field  : linear density field at z=0
    : Lbox      : BoxSize, in Mpc/h, will be converted to kpc/h in the IC output
    : Ng        : Number of grid on which to paint the particle
    : cosmology : nbodykit.cosmology, or astropy.cosmology.FLRW
    : redshift  : redshift of the initial condition
    """
    
    mesh = ArrayMesh(dx_field, BoxSize=Lbox)  # density contrast field centered at zero
    dk_field = mesh.compute(mode='complex')

    shift_gas =  - 0.5 * (cosmology.Om0 - cosmology.Ob0) / cosmology.Om0
    shift_dm = 0.5 * cosmology.Ob0 / cosmology.Om0
    
    pm = ParticleMesh(BoxSize=Lbox, Nmesh=[Ng,Ng,Ng])
    solver = Solver(pm,cosmology,B=1)
    
    Q_gas = pm.generate_uniform_particle_grid(shift=shift_gas)
    Q_dm = pm.generate_uniform_particle_grid(shift=shift_dm)
    
    scale_a = 1./(1+redshift)
    state_gas = solver.lpt(dk_field, Q_gas, a=scale_a, order=1) # order = 1 : Zel-dovich
    state_dm = solver.lpt(dk_field, Q_dm, a=scale_a, order=1)
    
    state = state_dm
    with FileMPI(state.pm.comm, IC_path, create=True) as ff:

        m0 = state.cosmology.rho_crit(0)*state.cosmology.Omega0_b*(Lbox**3)/state.csize
        m1 = state.cosmology.rho_crit(0)*(state.cosmology.Om0-state.cosmology.Omega0_b)*(Lbox**3)/state.csize

        with ff.create('Header') as bb:
            bb.attrs['BoxSize'] = Lbox*1000
            bb.attrs['HubbleParam'] = state.cosmology.h
            bb.attrs['MassTable'] = [m0, m1, 0, 0, 0, 0]

            bb.attrs['OmegaM'] = state.cosmology.Om0
            bb.attrs['OmegaB'] = state.cosmology.Omega0_b
            bb.attrs['OmegaL'] = state.cosmology.Omega0_lambda

            bb.attrs['Time'] = state.a['S']
            bb.attrs['TotNumPart'] = [state.csize, state.csize, 0, 0, 0, 0]

        ff.create_from_array('1/Position', 1000*periodic_wrap(state.X,Lbox)) # in kpc/h
        ff.create_from_array('1/Velocity', state.V/np.sqrt(state.a['S'])) # old gadget convention for IC
        dmID = np.arange(state.csize)
        ff.create_from_array('1/ID', dmID)

        #######################################################
        ff.create_from_array('0/Position', 1000*periodic_wrap(state_gas.X,Lbox))
        ff.create_from_array('0/Velocity', state_gas.V/np.sqrt(state.a['S']))
        gasID = np.arange(state.csize,2*state.csize)
        ff.create_from_array('0/ID', gasID)

    print ("IC generated!")
    print ("*********************************************")
    return state


#-----------------------------------------------------------------------------------------
def saveIC_dmo(IC_path,dx_field,Lbox,Ng,cosmology,redshift=99):
    """
    Use Zel-dovich approximation to back-scale the linear density field to initial redshift,
    paint only dm particles from the grid,
    and save initial condition in Mpc/h and use Constrained package to transfer to bt2 format
    
    Parameters
    ---------
    : dx_field  : linear density field at z=0
    : Lbox      : BoxSize, in Mpc/h, will be converted to kpc/h in the IC output
    : Ng        : Number of grid on which to paint the particle
    : cosmology : nbodykit.cosmology, or astropy.cosmology.FLRW
    : redshift  : redshift of the initial condition
    """
    
    mesh = ArrayMesh(dx_field, BoxSize=Lbox)  # density contrast field centered at zero
    dk_field = mesh.compute(mode='complex')
    
    scale_a = 1./(1+redshift)
    state = solver.lpt(dk_field, Q, a=scale_a, order=1)
    
    with FileMPI(state.pm.comm, IC_path, create=True) as ff:
        m1 = state.cosmology.rho_crit(0)*state.cosmology.Om0*(Lbox**3)/state.csize
        with ff.create('Header') as bb:
            bb.attrs['Time'] = state.a['S']
            bb.attrs['HubbleParam'] = state.cosmology.h
            bb.attrs['Omega0'] = state.cosmology.Om0
            bb.attrs['OmegaM'] = state.cosmology.Omega0_m
            bb.attrs['OmegaLambda'] = state.cosmology.Omega0_lambda
            bb.attrs['BoxSize'] = Lbox
            bb.attrs['NC'] = int(rint((state.csize)**(1/3)))
            bb.attrs['TotNumPart'] = [0, state.csize, 0, 0, 0, 0]
            bb.attrs['MassTable'] = [0, m1, 0, 0, 0, 0]

        ff.create_from_array('1/Position', state.X)
        # Peculiar velocity in km/s
        ff.create_from_array('1/Velocity', state.V)

        stateID = np.arange(state.csize)
        ff.create_from_array('1/ID', stateID)

    print ("IC generated!")
    print ("*********************************************")
    return state
    
