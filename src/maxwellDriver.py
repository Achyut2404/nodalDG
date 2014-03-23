# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Driver script for solving the 2D vacuum Maxwell's equations on TM form

import globalVar2D as glb
import math
import numpy
import maxwellCodes as maxw
glb.globalInit()
# Polynomial order used for approximation 
glb.N = 12

# Read in Mesh
import mesh2D
[glb.Nv, glb.VX, glb.VY, glb.K, glb.EToV,glb.BCType] = mesh2D.create('Grid/neu/Maxwell2D/Maxwell025.neu')

# Initialize solver and construct grid and metric
execfile("initiate2D.py")

# Set initial conditions
mmode = 1.0
nmode = 1.0
Ez = numpy.vectorize(math.sin)(mmode*math.pi*glb.x)*numpy.vectorize(math.sin)(nmode*math.pi*glb.y)
Hx = numpy.zeros([glb.Np, glb.K])
Hy = numpy.zeros([glb.Np, glb.K])

Ezinit=Ez.copy()
#functions2D.plot2D(Ezinit)

# Solve Problem
FinalTime = 1.5
[Hx,Hy,Ez,time,errors] = maxw.Maxwell2D(Hx,Hy,Ez,FinalTime);

#Postprocess
#functions2D.plot2D(Ez)

#Calculate error
omega=math.pi*(2)**0.5
#Exact solutions
EzExac = numpy.vectorize(math.sin)(mmode*math.pi*glb.x)*numpy.vectorize(math.sin)(nmode*math.pi*glb.y)*math.cos(omega*FinalTime)
HxExac=(-math.pi*nmode/omega)*numpy.vectorize(math.sin)(mmode*math.pi*glb.x)*numpy.vectorize(math.cos)(nmode*math.pi*glb.y)*math.sin(omega*FinalTime)
HyExac=(math.pi*mmode/omega)*numpy.vectorize(math.cos)(mmode*math.pi*glb.x)*numpy.vectorize(math.sin)(nmode*math.pi*glb.y)*math.sin(omega*FinalTime)
errEz=(numpy.average((Ez-EzExac).flatten()**2))**0.5
errHx=(numpy.average((Hx-HxExac).flatten()**2))**0.5
errHy=(numpy.average((Hy-HyExac).flatten()**2))**0.5
