# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Purpose : Setup script, building operators, grid, metric, and connectivity tables.

import globalVar2D as glb
import functions2D
import numpy

# Definition of constants
glb.Nfp = glb.N+1
glb.Np = (glb.N+1)*(glb.N+2)/2
glb.Nfaces=3
glb.NODETOL = 1e-12

# Compute nodal set
[glb.x,glb.y] = functions2D.Nodes2D(glb.N)
[glb.r,glb.s] = functions2D.xytors(glb.x,glb.y)

# Build reference element matrices
glb.V = functions2D.Vandermonde2D(glb.N,glb.r,glb.s)
glb.invV = numpy.linalg.inv(glb.V)
glb.MassMatrix = glb.invV.transpose().dot(glb.invV)
[glb.Dr,glb.Ds] = functions2D.Dmatrices2D(glb.N, glb.r, glb.s, glb.V)

# build coordinates of all the nodes
va = glb.EToV[:,0]
vb = glb.EToV[:,1]
vc = glb.EToV[:,2]
glb.x = 0.5*(-numpy.array([(glb.r+glb.s)]).transpose().dot(numpy.array([glb.VX[va]]))+numpy.array([(1+glb.r)]).transpose().dot(numpy.array([glb.VX[vb]]))+numpy.array([(1+glb.s)]).transpose().dot(numpy.array([glb.VX[vc]])))
glb.y = 0.5*(-numpy.array([(glb.r+glb.s)]).transpose().dot(numpy.array([glb.VY[va]]))+numpy.array([(1+glb.r)]).transpose().dot(numpy.array([glb.VY[vb]]))+numpy.array([(1+glb.s)]).transpose().dot(numpy.array([glb.VY[vc]])))

# find all the nodes that lie on each edge
fmask1   = numpy.nonzero( abs(glb.s+1) < glb.NODETOL)[0]
fmask2   = numpy.nonzero( abs(glb.r+glb.s) < glb.NODETOL)[0]
fmask3   = numpy.nonzero( abs(glb.r+1) < glb.NODETOL)[0]
glb.Fmask  = numpy.array([fmask1,fmask2,fmask3]).transpose()
glb.Fx = glb.x[glb.Fmask[:].transpose(), :] 
glb.Fy = glb.y[glb.Fmask[:].transpose(), :]

# Create surface integral terms
glb.LIFT = functions2D.Lift2D()

# calculate geometric factors
[glb.rx,glb.sx,glb.ry,glb.sy,glb.J] = functions2D.GeometricFactors2D(glb.x,glb.y,glb.Dr,glb.Ds)

# calculate geometric factors
[glb.nx, glb.ny, glb.sJ] = functions2D.Normals2D()
glb.Fscale = glb.sJ/(glb.J[glb.Fmask.transpose().flatten()])

# Build connectivity matrix
[glb.EToE, glb.EToF] = functions2D.Connect2D(glb.EToV)

# Build connectivity maps
[glb.mapM, glb.mapP, glb.vmapM, glb.vmapP, glb.vmapB, glb.mapB]=functions2D.BuildMaps2D()

# Compute weak operators (could be done in preprocessing to save time)
[glb.Vr, glb.Vs] = functions2D.GradVandermonde2D(glb.N, glb.r, glb.s)
glb.Drw = (glb.V.dot(glb.Vr.transpose())).dot(numpy.linalg.inv((glb.V.dot(glb.V.transpose()))))
glb.Dsw = (glb.V.dot(glb.Vs.transpose())).dot(numpy.linalg.inv((glb.V.dot(glb.V.transpose()))))

# Build boundary maps
functions2D.BuildBCMaps2D()
