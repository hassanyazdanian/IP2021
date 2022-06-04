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
    
    H = -sigma *grad_u
    return H

# Initial FEniCS setting 
parameters["linear_algebra_backend"]='PETSc'
parameters["form_compiler"]["quadrature_degree"] = 5 

# lower and upper bounds for conductivity
nu=1e-7;        # lower bound [S/mm]
mu=3e-3;        # upper bound [S/mm] 

#-------------- Generate Mesh
n= [120,120,120]       # Data mesh number of cells
n1= [118,118,118]      # Reconstruction mesh number of cells

x0=[-25,-25,-25];x1=[25,25,25]
# Generate tetrahedral mesh of the 3D rectangular prism 
# spanned by two points x0 and x1.

mesh1 = BoxMesh(Point(x0), Point(x1),n1[0],n1[1],n1[2])     # Reconstruction mes
mesh_true = BoxMesh(Point(x0), Point(x1),n[0],n[1],n[2])    # Data mesh


#-------------- Define spaces
#reconstruction_spaces
V1 = FunctionSpace (mesh1 ,"CG",1) #First order Continuous Galerkin (linear Lagrange element)
V1b = VectorFunctionSpace (mesh1 , "DG", 0) #Constant Discontinuous Lagrange element
#data_spaces
Vtrue = FunctionSpace ( mesh_true ,"CG",1)
Vtrueb = VectorFunctionSpace (mesh_true , "DG", 0)


#-------------- Define Boundary condition(s)
f1 = Expression ('x[0]',degree=1)
f2 = Expression ('x[1]',degree=1)
f3 = Expression ('x[2]',degree=1)


#-------------- True conductivity function
condbg = Constant(0.5E-3)      #Background
condcy=  Constant(1.0E-3)      #Elliptic Cylinder
cond13 = Constant(1.5E-3)      #Ellipsoid 13
cond12 = Constant(1.0E-3)      #Ellipsoid 12
cond11 = Constant(0.1E-3)      #Ellipsoid 11
cond10 = Constant(0.1E-3)      #Ellipsoid 10
cond9 = Constant(1.5E-3)       #Ellipsoid 9
cond8 = Constant(2.0E-3)       #Ellipsoid 8
cond7 = Constant(2.0E-3)       #Ellipsoid 7
cond6 = Constant(2.0E-3)       #Ellipsoid 6
cond5 = Constant(1.5E-3)       #Ellipsoid 5
cond4 = Constant(2.0E-3)       #Ellipsoid 4
cond3 = Constant(2.0E-3)       #Ellipsoid 3
cond2 = Constant(2.0E-3)       #Ellipsoid 2
cond1 = Constant(2.0E-3)       #Ellipsoid 1



xccy=[0.0,10]            #Elliptic Cylinder center
rcy=[7.5,11.3]           #Elliptic Cylinder radius

xc13=[0.0,0.0,-25]
r13=18.8

xc12=[0.0,0.0,25]
r12=11.3

xc11=[-7.5,-2.5,-5.0]
r11=[15.5,5.3,17.7]
theta11=105*pi/180

xc10=[7.5,-2.5,7.5]
r10=[13.4,3.4,13.4]
theta10=75*pi/180

xc9=[23.8,0.0,0.0]; xc8=[-23.8,2.5,0.0]; xc7=[-23.8,-2.5,0.0]
xc6=[-2.5,-23.8,0.0]; xc5=[2.5,-23.8,0.0]; xc4=[-15.0,-20.0,0.0]
xc3=[15.0,-20.0,0.0]; xc2=[0.0,5.0,0.0]; xc1=[-2.5,-2.5,0.0]

r9=1.2; r8=1.7; r7=1.2 
r6=1.2; r5=1.2; r4=1.7
r3=1.7; r2=1.9; r1=1.9

class sigmafun(UserExpression):
    def eval(self,values,x):         
            
        if((x[0]-xc1[0])**2+(x[1]-xc1[1])**2+(x[2]-xc1[2])**2) <=r1**2:
            values[0] = cond1
        
        elif((x[0]-xc2[0])**2+(x[1]-xc2[1])**2+(x[2]-xc2[2])**2) <=r2**2:
            values[0] = cond2
        
        elif((x[0]-xc3[0])**2+(x[1]-xc3[1])**2+(x[2]-xc3[2])**2) <=r3**2:
            values[0] = cond3
            
        elif((x[0]-xc4[0])**2+(x[1]-xc4[1])**2+(x[2]-xc4[2])**2) <=r4**2:
            values[0] = cond4
            
        elif((x[0]-xc5[0])**2+(x[1]-xc5[1])**2+(x[2]-xc5[2])**2) <=r5**2:
            values[0] = cond5
            
        elif((x[0]-xc6[0])**2+(x[1]-xc6[1])**2+(x[2]-xc6[2])**2) <=r6**2:
            values[0] = cond6
            
        elif((x[0]-xc7[0])**2+(x[1]-xc7[1])**2+(x[2]-xc7[2])**2) <=r7**2:
            values[0] = cond7
            
        elif((x[0]-xc8[0])**2+(x[1]-xc8[1])**2+(x[2]-xc8[2])**2) <=r8**2:
            values[0] = cond8
            
        elif((x[0]-xc9[0])**2+(x[1]-xc9[1])**2+(x[2]-xc9[2])**2) <=r9**2:
            values[0] = cond9
        
        elif (((x[0]-xc10[0])*cos(theta10)+(x[1]-xc10[1])*sin(theta10))**2/r10[0]**2\
            +((x[1]-xc10[1])*cos(theta10)-(x[0]-xc10[0])*sin(theta10))**2/r10[1]**2\
            +(x[2]-xc10[2])**2/r10[2]**2)<=1:
            values[0] = cond10
            
        elif (((x[0]-xc11[0])*cos(theta11)+(x[1]-xc11[1])*sin(theta11))**2/r11[0]**2\
              +((x[1]-xc11[1])*cos(theta11)-(x[0]-xc11[0])*sin(theta11))**2/r11[1]**2\
              +(x[2]-xc11[2])**2/r11[2]**2)<=1:
            values[0] = cond11
            
        elif((x[0]-xc12[0])**2+(x[1]-xc12[1])**2+(x[2]-xc12[2])**2) <=r12**2:
            values[0] = cond12
            
        elif((x[0]-xc13[0])**2+(x[1]-xc13[1])**2+(x[2]-xc13[2])**2) <=r13**2:
            values[0] = cond13 
            
        elif ((x[0]-xccy[0])**2/rcy[0]**2+(x[1]-xccy[1])**2/rcy[1]**2)<=1:
            values[0] = condcy   
        
        else:
            values[0] = condbg

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