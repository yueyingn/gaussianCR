#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --job-name=genIC
#SBATCH --output=%x-%j.out
#SBATCH --time=00:10:00

hostname; pwd; date
ROOT="/home/yueying/source/gaussianCR/examples"
outpath="IC"

module load intel/19.5
mpirun -n 1 python3  $ROOT/genIC.py --IC-path "$outpath" --Lbox 20 --Ng 128 --RG 0.9 \
                                    --significance 5.0 --compactness 3.6 \
                                    --a12sq 1.3 --a13sq 2.1 --Euler1 0 0 0 \
                                    --epsilon 34 --omega 4.71 --Euler2 3.14 1.57 1.57 \
                                    --vp '0.0' '0.0'  '0.0'   || exit 1


