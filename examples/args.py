import argparse

parser = argparse.ArgumentParser(description='prameters to genIC and the Constrained Peak parameters')

def get_args():
    parser.add_argument('--IC-path',required=True,type=str,help='the path to store IC')
    parser.add_argument('--redshift',default=99,type=float,help='the redshift of the output IC')
    
    parser.add_argument('--Rdm-seed',default=181170,type=int,help='random seed to generate IC')
    
    parser.add_argument('--Lbox',required=True,type=float,help='The box size') 
    parser.add_argument('--Ng',required=True,type=int,help='number of particles') 
    parser.add_argument('--RG',required=True,type=float,help='The Gaussian smoothing scale')
    
    parser.add_argument('--xpk-rel',default=[-1,-1,-1],nargs=3,type=float,help='the relative position of the peak to the Boxsize, if set to -1 then load the original peak position')  
    
    parser.add_argument('--significance',required=True,type=float,help='significance in unit of sigma0_RG')   
    parser.add_argument('--CONS-f1',action='store_true',help='if set,constrain c1 - c4 to be 0')     
    parser.add_argument('--compactness',required=True,type=float,help='compactness in unit of sigma2_RG')
    parser.add_argument('--a12sq',required=True,type=float,help='a12sq')
    parser.add_argument('--a13sq',required=True,type=float,help='a13sq')
    parser.add_argument('--Euler1',nargs=3,default=[0,0,0],type=float,help='Euler angle of mass ellipsoid principal axes')

    parser.add_argument('--vp',nargs=3,default=[0,0,0],type=float,help='peculiar velocity of the peak in km/s')
    
    parser.add_argument('--epsilon',required=True,type=float,help='magnitude of the tidal field, in unit of km/s/Mpc')
    parser.add_argument('--omega',required=True,type=float,help='angle of epsilon wrt 3 principal axes')
    parser.add_argument('--Euler2',nargs=3,default=[3.14,1.57,1.57],type=float,help='Euler angles of tidal field principal axes')
    
    args = parser.parse_args()
    
    return args

