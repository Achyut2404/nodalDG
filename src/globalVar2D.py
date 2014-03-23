# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Define global variables for inter module communication

import numpy

def globalInit():	
	"""Purpose: declare global variables"""
	
	global Np,Nfp,N,K
	global r,s
	global Dr,Ds,LIFT,Drw,Dsw,MassMatrix
	global Fx,Fy,nx,ny,jac,Fscale,J
	global vmapM,vmapP,vmapB,mapB,Fmask
	global BCType,mapI,mapO,mapW,mapF,mapC,mapS,mapM,mapP,mapD,mapN
	global vmapI,vmapO,vmapW,vmapO,vmapC,vmapS,vmapD,vmapN
	global rx,ry,sx,sy,J,sJ
	global rk4a,rk4b,rk4c
	global Nfaces,EToE,EToF,EToV
	global V,invV
	global x,y,NODETOL,VX,VY
	global mapSpB,vmapSpB
	# Some curved mesh and cubature quadrature specific data
	global cub,gauss,straight,curved
	global In,Out,Wall,Far,Cyl,Drichlet,Neuman,Slip
	In = 1
	Out = 2
	Wall = 3
	Far = 4
	Cyl = 5
	Drichlet = 6
	Neuman = 7
	Slip = 8
	
	[Np,Nfp,N,K,
	r,s,
	Dr,Ds,LIFT,Drw,Dsw,MassMatrix,
	Fx,Fy,nx,ny,jac,Fscale,J,
	vmapM,vmapP,vmapB,mapB,Fmask,
	BCType,mapI,mapO,mapW,mapF,mapC,mapS,mapM,mapP,mapD,mapN,
	vmapI,vmapO,vmapW,vmapO,vmapC,vmapS,vmapD,vmapN,
	rx,ry,sx,sy,J,sJ,
	rk4a,rk4b,rk4c,
	Nfaces,EToE,EToF,EToV,
	V,invV,
	x,y,NODETOL,VX,VY,
	cub,gauss]=numpy.zeros(65)
	
	straight=[]
	curved=[]
	[mapInB,vmapInB]=[0.,0.]
	[mapOutB,vmapOutB]=[0.,0.]
	# Low storage Runge-Kutta coefficients
	rk4a = [            0.0,
	        -567301805773.0/1357537059087.0,
	        -2404267990393.0/2016746695238.0,
	        -3550918686646.0/2091501179385.0,
	        -1275806237668.0/842570457699.0]
	rk4b = [ 1432997174477.0/9575080441755.0,
	         5161836677717.0/13612068292357.0,
	         1720146321549.0/2090206949498.0,
	         3134564353537.0/4481467310338.0,
	         2277821191437.0/14882151754819.0]
	rk4c = [             0.0,
	         1432997174477.0/9575080441755.0,
	         2526269341429.0/6820363962896.0,
	         2006345519317.0/3224310063776.0,
	         2802321613138.0/2924317926251.0]
	return()
