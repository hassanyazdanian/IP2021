#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This program is part of codes written for doi: 10.1088/1361-6420/ac1e81
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    This is a calling routine to reconstruct conductivities for the complex
    phantom in Case_B3 and using J-substitution method.

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
import time

# Initial FEniCS setting 
parameters["allow_extrapolation"] =True
parameters["linear_algebra_backend"]='PETSc'
parameters["form_compiler"]["quadrature_degree"] = 5 

alpha =0        # regularization parameter
K = 25          # iteration number

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

sw=0                                    # Define a switch for checking semi-convergence
sigmaTemp=sigma0                        # Hold sigma from previous iteration
time_start = time . time ()  

dof=np.size(V1b.dofmap().dofs())        # Get degree of freedoms (dofs)
temp1=np.zeros([1,dof])                 # Temporary variable 1
temp1[:,::3]=Htrue1.vector()[::3]       # Start from element 0 and take by step=3. Put the first component of data (x-component) to temp1. 
temp1[:,1::3]=Htrue1.vector()[1::3]     # Start from element 1 and take by step=3. Put the second component of data (y-component) to temp1. 

temp2=np.zeros([1,dof])                 # temporary variable 2
temp2[:,::3]=Htrue2.vector()[::3]       # Start from element 0 and take by step=3. Put 1st component of data (x-component) to temp2. 
temp2[:,1::3]=Htrue2.vector()[1::3]     # Start from element 1 and take by step=3. Put 2nd component of data (y-component) to temp2. 

Hp1 = Function(V1b)                     # Projected current density 1
Hp2 = Function(V1b)                     # Projected current density 2

H01 = Function(V1b)                     # Current density 1 obtained from forward problem
H02 = Function(V1b)                     # Current density 2 obtained from forward problem

#-------------- Start algorithm iteration
for k in range (1,K+1):
    
    #-------------- Forward problem
    timeforward = time . time ()
    u1 = EITForwardSolve (mesh1 ,sigma0,f1 ,V1)
    u2 = EITForwardSolve (mesh1 ,sigma0,f2 ,V1)
    w1 = grad(u1)
    w2 = grad(u2)
    H01 = ComputeH (sigma0 , w1)
    H01 = project (H01 ,V1b,solver_type='bicgstab',preconditioner_type='sor')
    H02 = ComputeH (sigma0 , w2)
    H02 = project (H02 ,V1b,solver_type='bicgstab',preconditioner_type='sor')
    timeforward=time . time ()-timeforward
    print (" forward time :",timeforward)

    # replace the 3rd component of the projeted current density with forward problem output
    temp1[:,2::3]=H01.vector()[2::3]
    temp1=temp1.ravel()
    Hp1.vector()[:]=temp1

    temp2[:,2::3]=H02.vector()[2::3]
    temp2=temp2.ravel()
    Hp2.vector()[:]=temp2
    
    #-------------- Inverse problem (J-substitution)
    timeinverse = time . time ()
    # sigma0=-(assemble(dot(Hp1,w1)*dx)+assemble(dot(Hp2,w2)*dx))/(assemble(dot(w1,w1)*dx)+assemble(dot(w2,w2)*dx))
    sigma0=-(inner(Hp1,w1)+inner(Hp2,w2))/(inner(w1,w1)+inner(w2,w2))
    timeinverse=time . time ()-timeinverse
    print (" inverse time :",timeinverse)
    
        
    err[k]= norm (project((sigma_trueV1 - sigma0 ),V1,solver_type='bicgstab',preconditioner_type='sor'),"L2")/norm_sigma_true
    if err[k-1]<err[k]:
        if sw==0:
            print ("Semi-Convergence")
            sigma1 = project (sigmaTemp ,V1,solver_type='bicgstab',preconditioner_type='sor')
            File('Sigma_Case_B3_SC.pvd')<<sigma1
            sw=1
    else:
        sigmaTemp=sigma0
    
    print ( "k=",k, "err=", err[k])
    
    #Rewrite temporary variables
    temp1=np.zeros([1,dof])
    temp1[:,::3]=Htrue1.vector()[::3]
    temp1[:,1::3]=Htrue1.vector()[1::3]
    
    temp2=np.zeros([1,dof])
    temp2[:,::3]=Htrue2.vector()[::3]
    temp2[:,1::3]=Htrue2.vector()[1::3]

time_stop = time . time ()
print (" Elapsed time :", time_stop-time_start)
print ("err=",err)

#-------------- Save reconstructed conductivity
sigma0 = project (sigma0 ,V1,solver_type='bicgstab',preconditioner_type='sor')
File('Sigma_Case_B3.pvd.pvd')<<sigma0