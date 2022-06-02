#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This program is part of codes written for doi: 10.1088/1361-6420/ac1e81
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    This is a calling routine to reconstruct conductivities for the simple
    phantom in Case_C1.

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

# lower and upper bounds for conductivity
nu=1e-7;        # lower bound [S/mm]
mu=3e-3;        # upper bound [S/mm]  


#-------------- Generate Mesh
n1= [88, 88, 88]        # number of cells [nx, ny, nz] in each direction       
x0 = [-25, -25, -25]            
x1 = [25, 25, 25]
# Generate tetrahedral mesh of the 3D rectangular prism 
# spanned by two points x0 and x1.
mesh1 = BoxMesh(MPI.comm_world, Point(x0), Point(x1),n1[0],n1[1],n1[2])


#-------------- Define spaces
V1 = FunctionSpace (mesh1 ,"CG",1)                  #First order Continuous Galerkin (linear Lagrange element)
V1FE = FiniteElement("CG", mesh1.ufl_cell(), 1)     

V1v = FunctionSpace (mesh1 , "RT", 1)               #First order Raviart-Thomas element
V1FEv = FiniteElement("RT", mesh1.ufl_cell(), 1)

V1b = VectorFunctionSpace (mesh1 , "DG", 0)         #Constant Discontinuous Lagrange element
V1bm = FunctionSpace (mesh1 , "DG", 0)

#-------------- Define Boundary condition(s)
f1 = Expression ('x[0]',degree=1)
f2 = Expression ('x[1]',degree=1)
f3 = Expression ('x[2]',degree=1)

#-------------- Conductivity function
a0 = Constant(0.5E-3)
class sigmafun(UserExpression):
    def eval(self,values,x):
        values[0] = a0
            
#-------------- Load (simulated) data (and true conductivity)
Htrue11 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue1.h5", "r")
input_file.read(Htrue11, "Htrue1")
input_file.close()

Htrue1 = Function(V1bm)
Htrue1=project(dot(Htrue11 , Htrue11 )**(0.5),V1bm,solver_type='bicgstab',preconditioner_type='sor')

Htrue22 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue2.h5", "r")
input_file.read(Htrue22, "Htrue2")
input_file.close()

Htrue2 = Function(V1bm)
Htrue2=project(dot(Htrue22 , Htrue22 )**(0.5),V1bm,solver_type='bicgstab',preconditioner_type='sor')


Htrue33 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue3.h5", "r")
input_file.read(Htrue33, "Htrue3")
input_file.close()

Htrue3 = Function(V1bm)
Htrue3=project(dot(Htrue33 , Htrue33 )**(0.5),V1bm,solver_type='bicgstab',preconditioner_type='sor')


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

sw=0                #define a switch for checking semi-convergence
sigmaTemp=sigma0   #Hold sigma from previous iteration
time_start = time . time () 
for k in range (1,K+1):
    
    timeforward = time . time ()
    u1 = EITForwardSolve (mesh1 ,sigma0,f1 ,V1)
    u2 = EITForwardSolve (mesh1 ,sigma0,f2 ,V1)
    u3 = EITForwardSolve (mesh1 ,sigma0,f3 ,V1)
    w1=grad(u1)
    w2=grad(u2)
    w3=grad(u3)
    H01 = ComputeH (sigma0 , w1)
    H02 = ComputeH (sigma0 , w2)
    H03 = ComputeH (sigma0 , w3)
    timeforward=time . time ()-timeforward
    print (" forward time :",timeforward)
    
    #-------------- Inverse problem
    timeinverse = time . time ()
    Hdiff1 = Htrue1 -H01
    Hdiff1=project(Hdiff1,V1bm,solver_type='bicgstab',preconditioner_type='sor') 
    Hdiff2 = Htrue2 -H02
    Hdiff2=project(Hdiff2,V1bm,solver_type='bicgstab',preconditioner_type='sor')  
    Hdiff3 = Htrue3 -H03
    Hdiff3=project(Hdiff3,V1bm,solver_type='bicgstab',preconditioner_type='sor')
    w1=project(w1,V1v,solver_type='bicgstab',preconditioner_type='sor')
    w2=project(w2,V1v,solver_type='bicgstab',preconditioner_type='sor')
    w3=project(w3,V1v,solver_type='bicgstab',preconditioner_type='sor')  
    dsigma = NewtonIterFull (mesh1 ,w1 ,sigma0,alpha ,Hdiff1 ,V1FE,V1FEv ,w2 , Hdiff2, w3 , Hdiff3)
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
            File('Sigma_Case_C2_SC.pvd')<<sigma1
            sw=1
    else:
        sigmaTemp=sigma0
            
    print ( "k=",k, "err=", err[k])

time_stop = time . time ()
print (" Elapsed time :", time_stop-time_start)
print ("err=",err)

#-------------- Save reconstructed conductivity
sigma0 = project (sigma0 ,V1,solver_type='bicgstab',preconditioner_type='sor')
File('Sigma_Case_C2.pvd.pvd')<<sigma0
