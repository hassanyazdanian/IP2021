#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This program is part of codes written for doi: 10.1088/1361-6420/ac1e81
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    Functions for generating simulated data (interior current density which is
    generated on a different mesh from reconstruction mesh to avoid inverse crime)

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
import time

"""
Solver for the forward problem with a given conductivity sigma
and a current density f on the boundary of the domain .

div( sigma * grad (u)) = 0
u = f on boundary

Input :
Mesh 'mesh ' the finite element mesh
Function 'sigma ' the conductivity defined in the domain
Function 'f' current density on the boundary

Output :
Function 'u' solution of the forward problem
"""

def EITForwardSolve (mesh ,sigma ,f,V):
  
    u = TrialFunction(V)
    v = TestFunction(V)

    a = sigma*inner(grad(u),grad(v))*dx     # lhs of the weak formulation
    uleft = Constant("0.00")                # rhs of weak formulation
    L = uleft*v*dx

    # Solve the system
    bcs = DirichletBC(V,f,"on_boundary")
    u = Function(V)
    problem = LinearVariationalProblem(a, L, u,[bcs])
    solver = LinearVariationalSolver(problem)
    prm = solver.parameters
    prm["linear_solver"] = 'bicgstab'
    prm["preconditioner"] = 'sor'
    prm["krylov_solver"]["absolute_tolerance"]= 1E-6
#    prm["krylov_solver"]["relative_tolerance"] = 1E-2
#    prm["krylov_solver"]["maximum_iterations"] = 100
    solver.solve()
    
    
    return u
#-------------- Compute interior current density    
def ComputeH (sigma ,grad_u):
    # solve (a == L, H)
    H = -sigma *grad_u
    return H

# Initial FEniCS setting 
parameters["linear_algebra_backend"]='PETSc'
parameters["form_compiler"]["quadrature_degree"] = 5 

# lower and upper bounds for conductivity
nu=1e-7;        # lower bound [S/mm]
mu=3e-3;        # upper bound [S/mm] 


#-------------- Generate Mesh
n= [90,90,90]       # Data mesh number of cells
n1= [88,88,88]      # Reconstruction mesh number of cells

x0=[-25,-25,-25];x1=[25,25,25]
# Generate tetrahedral mesh of the 3D rectangular prism 
# spanned by two points x0 and x1.

mesh1 = BoxMesh(Point(x0), Point(x1),n1[0],n1[1],n1[2])     # Reconstruction mes
mesh_true = BoxMesh(Point(x0), Point(x1),n[0],n[1],n[2])    # Data mesh


#-------------- Define spaces
#reconstruction_spaces
V1 = FunctionSpace (mesh1 ,"CG",1) #First order Continuous Galerkin (linear Lagrange element)
V1b = VectorFunctionSpace (mesh1 , "DG", 0) #Constant Discontinuous Lagrange element
#data_dpaces
Vtrue = FunctionSpace ( mesh_true ,"CG",1)
Vtrueb = VectorFunctionSpace (mesh_true , "DG", 0)


#-------------- Define Boundary condition(s)
f1 = Expression ('x[0]',degree=1)
f2 = Expression ('x[1]',degree=1)
f3 = Expression ('x[2]',degree=1)


#-------------- True conductivity function
a0 = Constant(1.0E-3)
a1 = Constant(1.5E-3)
class sigmafun(UserExpression):
    def eval(self,values,x):
        if (between(x[2], (5,15)) and between(x[1], (5,15)) and between(x[0], (5,15))):
            values[0] = a1
        else:
            values[0] = a0
            
#-------------- Constrain sigma
def ConsSig(sigma):
        value=sigma.vector().get_local()
        value[np.where(value < nu)]=nu
        value[np.where(value > mu)]=mu
        sigma.vector()[:]=value
        return sigma
          
sigma_true = sigmafun()
sigma_true = project(sigma_true,Vtrue,solver_type='bicgstab',preconditioner_type='sor')
sigma_true=ConsSig(sigma_true)

#-------------- Interpolate true sigma on reconstruction mesh
sigma_trueV1=interpolate(sigma_true,V1)    
sigma_trueV1=ConsSig(sigma_trueV1)


time_start = time . time ()
u_true1  = EITForwardSolve (mesh_true ,sigma_true,f1 ,Vtrue)
u_true2  = EITForwardSolve (mesh_true ,sigma_true,f2 ,Vtrue)
u_true3  = EITForwardSolve (mesh_true ,sigma_true,f3 ,Vtrue)

w_true1=grad(u_true1)
w_true2=grad(u_true2)
w_true3=grad(u_true3)

Htrue1 = ComputeH (sigma_true , w_true1)
Htrue2 = ComputeH (sigma_true , w_true2)
Htrue3 = ComputeH (sigma_true , w_true3)

#-------------- Interpolate from data mesh to reconstruction mesh
parameters["allow_extrapolation"] = True
Htrue1 = project (Htrue1 , Vtrueb,solver_type='bicgstab',preconditioner_type='sor')
Htrue2 = project (Htrue2 , Vtrueb,solver_type='bicgstab',preconditioner_type='sor')
Htrue3 = project (Htrue3 , Vtrueb,solver_type='bicgstab',preconditioner_type='sor')

HV1 = interpolate(Htrue1,V1b)
HV2 = interpolate(Htrue2,V1b)
HV3 = interpolate(Htrue3,V1b)
parameters["allow_extrapolation"] =False
Htrue1 = HV1
Htrue2 = HV2
Htrue3 = HV3    

time_stop=time . time ()
print (" Elapsed time :", time_stop-time_start)
   
# Save data for later use
output_file = HDF5File(mesh1.mpi_comm(), "Htrue1.h5", "w")
output_file.write(Htrue1, "Htrue1")
output_file.close()

output_file = HDF5File(mesh1.mpi_comm(), "Htrue2.h5", "w")
output_file.write(Htrue2, "Htrue2")
output_file.close()

output_file = HDF5File(mesh1.mpi_comm(), "Htrue3.h5", "w")
output_file.write(Htrue3, "Htrue3")
output_file.close()

output_file = HDF5File(mesh1.mpi_comm(), "sigma_trueV1.h5", "w")
output_file.write(sigma_trueV1, "sigma_trueV1")
output_file.close()

#-------------- Save true sigma in paraview format
sigmatrueV1 = project (sigma_trueV1 ,V1,solver_type='bicgstab',preconditioner_type='sor')
File('SigmaTrueCG.pvd')<<sigma_trueV1