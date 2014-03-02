# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Driver script for solving the 1D Euler equations

import globalVar as glb
import numpy
import functions
import eulerFuncs
import matplotlib.pyplot as plt

# Polynomial order used for approximation 
glb.N = 6

# Generate simple mesh
[glb.Nv, glb.VX, glb.K, EToV] = functions.MeshGen1D(0.0, 1.0, 250)

# Initialize solver and construct grid and metric
execfile("initiate.py")
gamma = 1.4

# Set up initial conditions -- Sod's problem

rho = (((-1)*numpy.vectorize(functions.step)(glb.x))+1)+((0.125)*numpy.vectorize(functions.step)(glb.x))
rhou = numpy.zeros([glb.Np,glb.K])
Ener = ((((-1)*numpy.vectorize(functions.step)(glb.x))+1)+((0.1)*numpy.vectorize(functions.step)(glb.x)))/(gamma-1.0)
FinalTime = 0.2

# Solve Problem
[rho,rhou,Ener] = eulerFuncs.Euler1D(rho,rhou,Ener,FinalTime)

#Exact solution
[rhoEx,uEx,pEx,rhouEx,EnerEx]=eulerFuncs.exactSod(glb.x,FinalTime)

#Calculate error L2 Norm
rhoL2=(numpy.average(((rho-rhoEx)**2.).flatten()))**0.5
rhouL2=(numpy.average(((rhou-rhouEx)**2.).flatten()))**0.5
EnerL2=(numpy.average(((Ener-EnerEx)**2.).flatten()))**0.5
