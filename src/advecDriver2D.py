# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Driver script for solving 2D advection
# This currently does not work properly! Work on boundary condition implementation
import globalVar2D as glb
import math
import numpy
import matplotlib.pyplot as plt
import functions2D

glb.globalInit()
# Polynomial order used for approximation 
glb.N = 10

# Read in Mesh
import mesh2D
[glb.Nv, glb.VX, glb.VY, glb.K, glb.EToV, glb.BCType] = mesh2D.create('Grid/neu/Maxwell2D/Maxwell1.neu')

# Initialize solver and construct grid and metric
execfile("initiate2D.py")

# Set initial conditions
u=numpy.vectorize(math.sin)(math.pi*glb.x)*numpy.vectorize(math.sin)(math.pi*glb.y)
#u=numpy.array([(abs(glb.x.flatten()[i])<0.1) and (abs(glb.y.flatten()[i])<0.1) for i in range(len(glb.x.flatten()))]).reshape(glb.x.shape)
uinit=u.copy()
functions2D.plot2D(uinit)

#Apply boundaries
[glb.mapInB,glb.vmapInB]=functions2D.findBoundary(functions2D.inBoundary)
[glb.mapOutB,glb.vmapOutB]=functions2D.findBoundary(functions2D.outBoundary)

# Solve Problem
ax=0.2
ay=0.2
a=[ax,ay]
FinalTime = 1.0
u = functions2D.advec2D(u,FinalTime,a)

#Postprocess
functions2D.plot2D(u)
