#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This program is part of codes written for doi: 10.1088/1361-6420/ac1e81
    If you find the program useful in your research, please cite our work.

    Copyright (C) 2022 Hassan Yazdanin, Kim Knudsen
    
    Function for solving the inverse problem based on Newton algorithm

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
Solver for the inverse problem based on the Newton algorithm.

Input :
Mesh 'mesh ' the finite element mesh
Function 'w0' grad(u1)
Function 'sigma ' the conductivity defined in the domain
Paramete 'alpha' the regularization parameter
Function 'Hdiff' difference between data and forward problem output
FiniteElement CG1 'V' and FiniteElement RT1 'Vg'
Function 'w2' grad(u2)
Function 'Hdiff2' difference between data and forward problem output for 2nd bc

Output :
Function 'x1' delta_sigma, solution of the inverse problem
"""

def NewtonIterFull (mesh ,w0 ,sigma ,alpha ,Hdiff ,V,Vg, w2, Hdiff2):  #nzig: none_zero_intial_guess

    # Define function space
    W = FunctionSpace (mesh,MixedElement([V, V, V,Vg ,V,Vg]))
    # Define trial and test functions
    (deltas ,deltau ,deltaz ,deltaw , deltau2 , deltaw2)= split ( TrialFunction (W))
    (v1 ,v2 ,v3 ,v4 ,v5 ,v6) = split ( TestFunction (W))
        
    #Define boundary condition(s)
    bc1 = DirichletBC (W.sub(1), 0.0, "on_boundary")
    bc4 = DirichletBC (W.sub(4), 0.0, "on_boundary" )
    
    
    #Define different part of weak formulation
    def Q1(deltaz ,deltaw ,w0 ):
        q1 = inner ( grad ( deltaz ),w0 )+ deltaz *div(w0) \
        + inner ( grad ( sigma ), deltaw ) \
        + sigma *div( deltaw )
        return q1
    
    def Q2(deltaz ,deltawxy ,w0xy ,new= False ):
        
        q2 = -deltaz *w0xy - sigma *deltawxy
        return q2

    def G(deltau ,deltaw ,v2 ,v4 ):
        g = inner ( grad ( deltau )-deltaw , grad (v2)-v4)
        return g

    def Reg (deltas , deltau1 ):
        reg = alpha *( deltas *v1 + deltau1 *v2)
        return reg

    def Lfun (Hdiffxy ,v3 ,v4xy ,w0xy):
        l1 = Hdiffxy *( -v3 *w0xy - sigma*v4xy)
        return l1
    
    a= Q1(deltaz ,deltaw ,w0 )* Q1(v3 ,v4 ,w0)
    a=a+Q2(deltaz ,deltaw[0] ,w0[0])* Q2(v3 ,v4[0] ,w0[0])
    a=a+Q2(deltaz ,deltaw[1] ,w0[1])* Q2(v3 ,v4[1] ,w0[1])
    a=a+(deltas - deltaz )*(v1 -v3)
    a=a+G(deltau ,deltaw ,v2 ,v4)
    a=a+Reg(deltas , deltau )
    a=a+Q1(deltaz , deltaw2 ,w2 )* Q1(v3 ,v6 ,w2)
    a=a+Q2(deltaz , deltaw2[0] ,w2[0])* Q2(v3 ,v6[0] ,w2[0])
    a=a+Q2(deltaz , deltaw2[1] ,w2[1])* Q2(v3 ,v6[1] ,w2[1])
    a=a+G( deltau2 , deltaw2 ,v5 ,v6)
    a=a+alpha * deltau2 *v5
    a=a*dx
    
    L=Lfun (Hdiff[0] ,v3 ,v4[0] ,w0[0])+Lfun (Hdiff[1] ,v3 ,v4[1] ,w0[1])
    L=L+Lfun ( Hdiff2[0] ,v3 ,v6[0] ,w2[0])+Lfun ( Hdiff2[1] ,v3 ,v6[1] ,w2[1])
    L=L*dx
    
    # Solve the system
    w = Function (W)
    problem = LinearVariationalProblem(a, L, w, [bc1,bc4])
    solver = LinearVariationalSolver(problem)
    prm = solver.parameters
    prm["linear_solver"] = 'bicgstab'
    prm["preconditioner"] = 'sor'
    prm["krylov_solver"]["absolute_tolerance"]= 1E-4
    #prm["krylov_solver"]["relative_tolerance"] = 1E-2
    #prm["krylov_solver"]["maximum_iterations"] = 1000
    prm["krylov_solver"]['monitor_convergence'] = True
    prm["krylov_solver"]['nonzero_initial_guess'] = False
    prm["krylov_solver"]['report'] = True
    solver.solve()
    
       
    (x1 ,x2 ,x3 ,x4 ,x5 ,x6) = w. split ( deepcopy = True )
    
    return x1