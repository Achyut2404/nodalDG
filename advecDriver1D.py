# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Driver script for solving the 1D advection equations
# Speed is 2*pi

import globalVar as glb
glb.globalInit()
import functions
import math
import runpy
import matplotlib.pyplot as plt

# Order of polymomials used for approximation 
glb.N = 1

# Generate simple mesh
[Nv, glb.VX, glb.K, EToV] = functions.MeshGen1D(0.0,2.0*math.pi,200);

# Initialize solver and construct grid and metric
execfile("initiate.py")

# Set initial conditions
#u = numpy.vectorize(math.sin)(glb.x)
u = numpy.vectorize(functions.step)(glb.x)
uEx=u.copy()

# Solve Problem
FinalTime = 1.
u = functions.advec1D(u,FinalTime)

#Calculate Error
uL2=(numpy.average(((u-uEx)**2.).flatten()))**0.5
print uL2
