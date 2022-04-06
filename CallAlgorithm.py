#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:23:18 2019

#@author: hassan
#"""
#
import numpy as np
from dolfin import *
from forward_solver import *
from NewtonAlgorithm import *
import time
#

alpha =0  #regularization parameter
K = 10          #iterattion numbers

#conductivtiy upper and lower bounds
nu=1E-7;  # S/mm
mu=3E-3;  # S/mm

parameters["allow_extrapolation"] =True
parameters["linear_algebra_backend"]='PETSc'
parameters["form_compiler"]["quadrature_degree"] = 5 

#Mesh
n1= [88,88,88]
x0=[-25,-25,-25];x1=[25,25,25]
mesh1 = BoxMesh(MPI.comm_world, Point(x0), Point(x1),n1[0],n1[1],n1[2])
# mesh1 = BoxMesh(Point(x0), Point(x1),n1[0],n1[1],n1[2])



V1 = FunctionSpace (mesh1 ,"CG",1)
V1FE = FiniteElement("CG", mesh1.ufl_cell(), 1)

V1v = FunctionSpace (mesh1 , "RT", 1)
V1FEv = FiniteElement("RT", mesh1.ufl_cell(), 1)

V1b = VectorFunctionSpace (mesh1 , "DG", 0)


 
f1 = Expression ('x[0]',degree=1)
f1str = 'x'

f2 = Expression ('x[1]',degree=1)
f2str = 'y'

f3 = Expression ('x[2]',degree=1)
f3str = 'z'

# Conductivity
a0 = Constant(0.5E-3)   # S/mm
class sigmafun(UserExpression):
    def eval(self,values,x):
        values[0] = a0
            
# Load solution
Htrue1 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue1.h5", "r")
input_file.read(Htrue1, "Htrue1")
input_file.close()

Htrue2 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue2.h5", "r")
input_file.read(Htrue2, "Htrue2")
input_file.close()

Htrue3 = Function(V1b)
input_file = HDF5File(mesh1.mpi_comm(), "Htrue3.h5", "r")
input_file.read(Htrue3, "Htrue3")
input_file.close()

sigma_trueV1 = Function(V1)
input_file = HDF5File(mesh1.mpi_comm(), "sigma_trueV1.h5", "r")
input_file.read(sigma_trueV1, "sigma_trueV1")
input_file.close()

#constrain sigma
def ConsSig(sigma):
        value=sigma.vector().get_local()
        # sigma.vector()[:]=value*(np.sign(value)+1)/2+1e-2
        # print(value[value<=nu])
        value[np.where(value < nu)]=nu
        value[np.where(value > mu)]=mu
        sigma.vector()[:]=value
        # print("newsigma",value[value<=0])
        return sigma
            
sigma_trueV1=ConsSig(sigma_trueV1)
            

norm_sigma_true = norm(sigma_trueV1,"L2")

# ======================================================
# Newton - type algorithm for inverse problem
# ======================================================
sigmaFun =sigmafun()
sigma0 = project(sigmaFun,V1, solver_type='bicgstab',preconditioner_type='sor')

err=np.zeros(K+1)
err[0]= norm (project((sigma_trueV1 - sigma0 ),V1, solver_type='bicgstab',preconditioner_type='sor'),"L2")/norm_sigma_true
print ("err0=",err[0])

sw=0  #define a switch
time1 = time . time () 
sigmaTemp=sigma0   
for k in range (1,K+1):
    
    timeforward = time . time ()
    u1 = EITForwardSolve (mesh1 ,sigma0,f1 ,V1)
    u2 = EITForwardSolve (mesh1 ,sigma0,f2 ,V1)
    u3 = EITForwardSolve (mesh1 ,sigma0,f3 ,V1)
    timeforward=time . time ()-timeforward
    print (" forward time :",timeforward)

    w1=grad(u1)
    w2=grad(u2)
    w3=grad(u3)
    
    
    # Newton iteration
    H01 = ComputeH (sigma0 , w1)
    Hdiff1 = Htrue1 -H01
    
    Hdiff1=project(Hdiff1,V1b,solver_type='bicgstab',preconditioner_type='sor')
    
    H02 = ComputeH (sigma0 , w2)
    Hdiff2 = Htrue2 -H02
    
    Hdiff2=project(Hdiff2,V1b,solver_type='bicgstab',preconditioner_type='sor')
        
    H03 = ComputeH (sigma0 , w3)
    Hdiff3 = Htrue3 -H03
    
    Hdiff3=project(Hdiff3,V1b,solver_type='bicgstab',preconditioner_type='sor')
     
    w1=project(w1,V1v,solver_type='bicgstab',preconditioner_type='sor')
    w2=project(w2,V1v,solver_type='bicgstab',preconditioner_type='sor')
    w3=project(w3,V1v,solver_type='bicgstab',preconditioner_type='sor')
    
          
    timeinverse = time . time ()
    
   
    dsigma = NewtonIterFull (mesh1 ,w1 ,sigma0,alpha ,Hdiff1 ,V1FE,V1FEv ,w2 , Hdiff2, w3 , Hdiff3)

    timeinverse=time . time ()-timeinverse
    print (" inverse time :",timeinverse)
   
           
    sigma0 = sigma0 + dsigma
    sigma0=project (sigma0 ,V1,solver_type='bicgstab',preconditioner_type='sor')
    sigma0=ConsSig(sigma0); # constrain sigma
    
    print ("k=", k)
       
    err[k]= norm (project((sigma_trueV1 - sigma0 ),V1,solver_type='bicgstab',preconditioner_type='sor'),"L2")/norm_sigma_true
    
    if err[k-1]<err[k]:
        if sw==0:
            print ("Semi-Convergence")
            sigma1 = project (sigmaTemp ,V1,solver_type='bicgstab',preconditioner_type='sor')
            File('Sigma3MFullJ5itrSC.pvd')<<sigma1
            sw=1
    else:
        sigmaTemp=sigma0
            

    print ( "err=", err[k])

time2 = time . time ()
elapsed = time2 - time1
print (" Elapsed time :")
print ( elapsed )

print ("err=",err)


sigma1 = project (sigma0 ,V1,solver_type='bicgstab',preconditioner_type='sor')
File('Sigma3MFullJ5itrAl0.pvd.pvd')<<sigma1
