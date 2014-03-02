# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Purpose : Setup script, building operators, grid, metric and connectivity for 1D solver.     

import globalVar as glb
import functions
import numpy
# Definition of constants

glb.NODETOL = 1e-18
glb.Np = glb.N+1
glb.Nfp = 1
glb.Nfaces=2
# Compute basic Legendre Gauss Lobatto grid

glb.r = functions.JacobiGL(0,0,glb.N)

# Build reference element matrices
glb.V  = functions.Vandermonde1D(glb.N, glb.r)
glb.invV = numpy.linalg.inv(glb.V)
glb.Dr = functions.Dmatrix1D(glb.N, glb.r, glb.V)

# Create surface integral terms
glb.LIFT = functions.Lift1D()

# build coordinates of all the nodes
va = EToV[:,0].transpose()
vb = EToV[:,1].transpose()
glb.x = numpy.ones([glb.N+1,1]).dot([glb.VX[va]]) + 0.5*(numpy.array([(glb.r+1)]).transpose().dot([(glb.VX[vb]-glb.VX[va])]))

# calculate geometric factors
[glb.rx,glb.J] = functions.GeometricFactors1D(glb.x,glb.Dr)

# Compute masks for edge nodes
fmask1 = numpy.array(numpy.nonzero( abs(glb.r+1) < glb.NODETOL)).transpose()
fmask2 = numpy.array(numpy.nonzero( abs(glb.r-1) < glb.NODETOL)).transpose()
glb.Fmask  = numpy.array([fmask1,fmask2]).transpose()[0][0]
glb.Fx = glb.x[glb.Fmask[:],:]

#Build surface normals and inverse metric at surface
glb.nx = functions.Normals1D()
glb.Fscale = 1/(glb.J[glb.Fmask,:])

# Build connectivity matrix
[glb.EToE, glb.EToF] = functions.Connect1D(EToV)

# Build connectivity maps
[glb.vmapM, glb.vmapP, glb.vmapB, glb.mapB] = functions.BuildMaps1D()
