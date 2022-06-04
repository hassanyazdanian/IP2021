#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J MySerialPython
# -- choose queue --
#BSUB -q hpc
# -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
# -- specify that we need 100GB of memory per core/slot -- 
#BSUB -R "rusage[mem=100GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
# -- Output File --
#BSUB -o Output_%J.txt
# -- Error File --
#BSUB -e Error_%J.txt
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
# -- Number of cores requested -- 
#BSUB -n 1 
# -- end of LSF options --
# OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
# export OMP_NUM_THREADS

# load module needed by MySerialPython
#module load mpi4py/3.0.0-python-3.6.2 
#module load FEniCS/2019.1.0-with-petsc-3.10.5-and-numpy-1.16.5-and-slow-blas
module load FEniCS/2019.1.0
#module load FEniCS/2018.1.0-with-petsc-and-slepc-and-scotch-and-newmpi

# Run my program 
#mpirun -np $LSB_DJOB_NUMPROC python3 Generate_data.py -logfile MySerialPythonOut
python3 Generate_data.py -logfile MySerialPythonOut
