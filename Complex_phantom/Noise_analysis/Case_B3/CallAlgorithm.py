#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This program is part of codes written for doi: 10.1088/1361-6420/ac1e81
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    This is a calling routine to reconstruct conductivities for the complex
    phantom in Case_B3 (noise_study).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
from dolfin import *
from forward_solver import *
from NewtonAlgorithm import *
import time

# Initial FEniCS setting 
parameters["allow_extrapolation"] =True
parameters["linear_algebra_backend"]='PETSc'
parameters["form_compiler"]["quadrature_degree"] = 5 

alpha =0        # regularization parameter
K = 15          # iteration number
level=0.8       # relative noise level 

# lower and upper bounds for conductivity
nu=1e-7;        # lower bound [S/mm]
mu=3e-3;        # upper bound [S/mm]  


#-------------- Generate Mesh
n1= [118, 118, 118]        # number of cells [nx, ny, nz] in each direction       
x0=[-25, -25, -25]            
x1=[25, 25, 25]
# Generate tetrahedral mesh of the 3D rectangular prism 
# spanned by two points x0 and x1.
mesh1 = BoxMesh(MPI.comm_world, Point(x0), Point(x1),n1[0],n1[1],n1[2])


#-------------- Define spaces
V1 = FunctionSpace (mesh1 ,"CG",1)                  #First order Continuous Galerkin (linear Lagrange element)
V1FE = FiniteElement("CG", mesh1.ufl_cell(), 1)     

V1v = FunctionSpace (mesh1 , "RT", 1)               #First order Raviart-Thomas element
V1FEv = FiniteElement("RT", mesh1.ufl_cell(), 1)

V1b = VectorFunctionSpace (mesh1 , "DG", 0)         #Constant Discontinuous Lagrange element
dof=np.size(V1b.dofmap().dofs())                    #Get degree of freedoms (dofs)

 
#-------------- Define Boundary condition(s)
f1 = Expression ('x[0]',degree=1)
f2 = Expression ('x[1]',degree=1)



#-------------- Conductivity function
a0 = Constant(0.5E-3)
class sigmafun(UserExpression):
    def eval(self,values,x):
        values[0] = a0
            
#-------------- Load (simulated) data (and true conductivity)
Htrue1 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue1.h5", "r")
input_file.read(Htrue1, "Htrue1")
input_file.close()

Htrue2 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue2.h5", "r")
input_file.read(Htrue2, "Htrue2")
input_file.close()

sigma_trueV1 = Function(V1)
input_file = HDF5File(mesh1.mpi_comm(), "sigma_trueV1.h5", "r")
input_file.read(sigma_trueV1, "sigma_trueV1")
input_file.close()

#-------------- Adding noise to data
np.random.seed(0)                   #set the seed for genrating same randn
Htrue1norm=norm(Htrue1,'L2',mesh1)
noise=np.random.normal(0, 1, dof)
NOISE= Function(V1b)
NOISE.vector()[:]=noise
noiseNorm= norm(NOISE,'L2',mesh1);
NOISE.vector()[:]=level*Htrue1norm*noise/noiseNorm
H1noise=Function(V1b)
H1noise.vector()[:]=Htrue1.vector().get_local()+NOISE.vector().get_local()

Htrue2norm=norm(Htrue2,'L2',mesh1)
noise=np.random.normal(0, 1, dof)
NOISE= Function(V1b)
NOISE.vector()[:]=noise
noiseNorm= norm(NOISE,'L2',mesh1);
NOISE.vector()[:]=level*Htrue2norm*noise/noiseNorm
H2noise=Function(V1b)
H2noise.vector()[:]=Htrue2.vector().get_local()+NOISE.vector().get_local()

#-------------- Constrain sigma
def ConsSig(sigma):
        value=sigma.vector().get_local()
        value[np.where(value < nu)]=nu
        value[np.where(value > mu)]=mu
        sigma.vector()[:]=value
        return sigma
            
sigma_trueV1=ConsSig(sigma_trueV1)
            

#-------------- Initilizing
sigma0 =sigmafun()
sigma0 = project(sigma0,V1, solver_type='bicgstab',preconditioner_type='sor')

err=np.zeros(K+1)
norm_sigma_true = norm(sigma_trueV1,"L2")
err[0]= norm (project((sigma_trueV1 - sigma0 ),V1, solver_type='bicgstab',preconditioner_type='sor'),"L2")/norm_sigma_true
print ("err0=",err[0])

sw=0                #define a switch for checking semi-convergence
sigmaTemp=sigma0   #Hold sigma from previous iteration
time_start = time . time () 


#-------------- Start algorithm iteration
for k in range (1,K+1):
    
    #-------------- Forward problem
    timeforward = time . time ()
    u1 = EITForwardSolve (mesh1 ,sigma0,f1 ,V1)
    u2 = EITForwardSolve (mesh1 ,sigma0,f2 ,V1)
    w1=grad(u1)
    w2=grad(u2)
    H01 = ComputeH (sigma0 , w1)
    H02 = ComputeH (sigma0 , w2)
    timeforward=time . time ()-timeforward
    print (" forward time :",timeforward)
    
    #-------------- Inverse problem
    timeinverse = time . time ()
    Hdiff1 = H1noise -H01
    Hdiff1=project(Hdiff1,V1b,solver_type='bicgstab',preconditioner_type='sor')
    Hdiff2 = H2noise -H02
    Hdiff2=project(Hdiff2,V1b,solver_type='bicgstab',preconditioner_type='sor')
    w1=project(w1,V1v,solver_type='bicgstab',preconditioner_type='sor')
    w2=project(w2,V1v,solver_type='bicgstab',preconditioner_type='sor')
    dsigma = NewtonIterFull (mesh1 ,w1 ,sigma0,alpha ,Hdiff1 ,V1FE,V1FEv ,w2 , Hdiff2)
    sigma0 = sigma0 + dsigma
    sigma0=project (sigma0 ,V1,solver_type='bicgstab',preconditioner_type='sor')
    sigma0=ConsSig(sigma0); # constrain sigma    
    timeinverse=time . time ()-timeinverse
    print (" inverse time :",timeinverse)
       
    err[k]= norm (project((sigma_trueV1 - sigma0 ),V1,solver_type='bicgstab',preconditioner_type='sor'),"L2")/norm_sigma_true
    if err[k-1]<err[k]:
        if sw==0:
            print ("Semi-Convergence")
            sigma1 = project (sigmaTemp ,V1,solver_type='bicgstab',preconditioner_type='sor')
            File('Sigma_Case_B3_noisy_SC.pvd')<<sigmaTemp
            sw=1
    else:
        sigmaTemp=sigma0
            

    print ( "k=",k, "err=", err[k])

time_stop = time . time ()
print (" Elapsed time :", time_stop-time_start)
print ("err=",err)

#-------------- Save reconstructed conductivity
sigma0 = project (sigma0 ,V1,solver_type='bicgstab',preconditioner_type='sor')
File('Sigma_Case_B3_noisy.pvd.pvd')<<sigma0    
