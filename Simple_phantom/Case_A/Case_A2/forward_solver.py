#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This program is part of codes written for doi: 10.1088/1361-6420/ac1e81
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    Functions for solving the forward problem and computing interior current 
    density

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

from dolfin import *
import numpy as np

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
    H = sigma *dot(grad_u , grad_u )**0.5
    return H
