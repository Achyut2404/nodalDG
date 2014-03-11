# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Initialize cubature variables for higher order volume integration

import numpy
def cubatureInitiate(CubatureOrder):
	
	"""function cub = CubatureVolumeMesh2D(CubatureOrder)
	purpose: build cubature nodes, weights and geometric factors for all elements"""
	
	import globalVar2D as glb
	import functions2D as func

	# set up cubature nodes
	global R,S,W,Ncub
	[R,S,W, Ncub] = func.cubature2D(CubatureOrder)

	#evaluate generalized Vandermonde of Lagrange interpolant functions at cubature nodes
	global V
	V  = func.InterpMatrix2D(R, S)

	# evaluate local derivatives of Lagrange interpolants at cubature nodes
	global Dr,Ds
	[Dr,Ds] = func.Dmatrices2D(glb.N,R,S,glb.V)

	# evaluate the geometric factors at the cubature nodes
	global rx, sx, ry, sy, J
	[rx,sx,ry,sy,J] = func.GeometricFactors2D(glb.x,glb.y, Dr,Ds)

	# custom mass matrix per element
	# This is only needed if there are any curved elements
	if len(glb.curved) != 0:
		global mmCHOL,mm
		mmCHOL = numpy.zeros([glb.Np, glb.Np, glb.K])
		mm = numpy.zeros([glb.Np, glb.Np, glb.K])
		for k in range(glb.K):
			mm[:,:,k] = V.transpose().dot(numpy.diag(J[:,k]*W).dot(V))
			mmCHOL[:,:,k] = numpy.linalg.cholesky(mm[:,:,k])

	# incorporate weights and Jacobian
	global w
	w = W.copy()
	W = numpy.array([W]).transpose().dot(numpy.ones([1,glb.K]))
	W = W*J 

	# compute coordinates of cubature nodes
	global x,y
	x = V.dot(glb.x)
	y = V.dot(glb.y)
	return()
