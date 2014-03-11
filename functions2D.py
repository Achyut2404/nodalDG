# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Initialization functions for 2-Dimensional Nodal DG formulation

import functions
import numpy
import math

def Simplex2DP(a,b,i,j):
	
	"""function [P] = Simplex2DP(a,b,i,j);
	Purpose : Evaluate 2D orthonormal polynomial
	          on simplex at (a,b) of order (i,j)."""
	
	h1 = functions.JacobiP(a,0,0,i)
	h2 = functions.JacobiP(b,2*i+1,0,j)
	P = (2.0**0.5)*h1*h2*((1-b)**i)
	return(P)

def rstoab(r,s):
	"""function [a,b] = rstoab(r,s)
	Purpose : Transfer from (r,s) -> (a,b) coordinates in triangle"""
	
	Np = len(r.flatten())
	a = numpy.copy(r)
	for i in range(Np):
		if(s.flatten()[i] != 1):
			a.ravel()[i] = (2*(1+r.flatten()[i])/(1-s.flatten()[i]))-1
		else:
			a.ravel()[i] = -1
	b = s.copy()
	return(numpy.array([a,b]))
	
def Warpfactor(N, rout):
	
	"""function warp = Warpfactor(N, rout)
	Purpose  : Compute scaled warp function at order N based on rout interpolation nodes"""
	
	# Compute LGL and equidistant node distribution
	LGLr = functions.JacobiGL(0.,0.,N)
	req  = numpy.linspace(-1.,1.,N+1).transpose()
	
	# Compute V based on req
	Veq = functions.Vandermonde1D(N,req)
	
	# Evaluate Lagrange polynomial at rout
	Nr = len(rout)
	Pmat = numpy.zeros([N+1,Nr])
	for i in range(N+1):
		Pmat[i] = functions.JacobiP(rout, 0, 0, i).transpose()
	
	Lmat = numpy.linalg.inv(Veq.transpose()).dot(Pmat)
	
	# Compute warp factor
	warp = Lmat.transpose().dot(LGLr - req)
	
	# Scale factor
	zerof = (abs(rout)<1.0-1.0e-10)
	sf = 1.0 - (zerof*rout)**2
	warp = warp/sf + warp*(zerof-1)
	return(warp)

def Nodes2D(N):
	"""function [x,y] = Nodes2D(N);
	Purpose  : Compute (x,y) nodes in equilateral triangle for polynomial of order N"""
	
	alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
				  1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
	          
	# Set optimized parameter, alpha, depending on order N
	if (N<16):
	  alpha = alpopt[N]
	else:
	  alpha = 5./3
	
	# total number of nodes
	Np = (N+1)*(N+2)/2
	
	# Create equidistributed nodes on equilateral triangle
	L1 = numpy.zeros(Np)
	L2 = numpy.zeros(Np)
	L3 = numpy.zeros(Np)
	sk = 0
	for n in range(N+1):
	  for m in range(N+2-n-1):
	    L1[sk] = float(n)/N
	    L3[sk] = float(m)/N
	    sk = sk+1
	
	L2 = 1.0-L1-L3
	x = -L2+L3
	y = (-L2-L3+2*L1)/(3.0**0.5)
	
	# Compute blending function at each node for each edge
	blend1 = 4*L2*L3
	blend2 = 4*L1*L3
	blend3 = 4*L1*L2;
	
	# Amount of warp for each node, for each edge
	warpf1 = Warpfactor(N,L3-L2)
	warpf2 = Warpfactor(N,L1-L3)
	warpf3 = Warpfactor(N,L2-L1)
	
	# Combine blend & warp
	warp1 = blend1*warpf1*(1 + (alpha*L1)**2);
	warp2 = blend2*warpf2*(1 + (alpha*L2)**2);
	warp3 = blend3*warpf3*(1 + (alpha*L3)**2);
	
	# Accumulate deformations associated with each edge
	x = x + 1.0*warp1 + math.cos(2*math.pi/3)*warp2 + math.cos(4*math.pi/3)*warp3;
	y = y + 0.0*warp1 + math.sin(2*math.pi/3)*warp2 + math.sin(4*math.pi/3)*warp3;
	return(numpy.array([x,y]))

def xytors(x,y):	
	"""function [r,s] = xytors(x, y)
	Purpose : From (x,y) in equilateral triangle to (r,s) coordinates in standard triangle"""
	
	L1 = (math.sqrt(3.0)*y+1.0)/3.0
	L2 = (-3.0*x - math.sqrt(3.0)*y + 2.0)/6.0
	L3 = ( 3.0*x - math.sqrt(3.0)*y + 2.0)/6.0
	
	r = -L2 + L3 - L1
	s = -L2 - L3 + L1
	return(numpy.array([r,s]))

def Vandermonde2D(N, r, s):
	"""function [V2D] = Vandermonde2D(N, r, s)
	Purpose : Initialize the 2D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i)"""
	
	V2D = numpy.zeros([len(r),(N+1)*(N+2)/2])
	
	# Transfer to (a,b) coordinates
	[a, b] = rstoab(r, s)
	
	# build the Vandermonde matrix
	sk = 0
	for i in range(N+1):
	  for j in range(0,N-i+1):
	    V2D[:,sk] = Simplex2DP(a,b,i,j)
	    sk = sk+1
	return(V2D)

def GradSimplex2DP(a,b,i,j):
	"""function [dmodedr, dmodeds] = GradSimplex2DP(a,b,id,jd)
	Purpose: Return the derivatives of the modal basis (id,jd) on the 2D simplex at (a,b)."""
	
	fa = functions.JacobiP(a, 0, 0, i)
	dfa = functions.GradJacobiP(a, 0, 0, i);
	gb = functions.JacobiP(b, 2*i+1,0, j)
	dgb = functions.GradJacobiP(b, 2*i+1,0, j);
	
	# r-derivative
	# d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
	dmodedr = dfa*gb
	if(i>0):
	  dmodedr = dmodedr*((0.5*(1-b))**(i-1))
	
	# s-derivative
	# d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
	dmodeds = dfa*(gb*(0.5*(1+a)))
	if(i>0):
	 dmodeds = dmodeds*((0.5*(1-b))**(i-1))
	
	tmp = dgb*((0.5*(1-b))**i)
	if(i>0):
	  tmp = tmp-0.5*i*gb*((0.5*(1-b))**(i-1))
	dmodeds = dmodeds+fa*tmp
	
	# Normalize
	dmodedr = 2**(i+0.5)*dmodedr
	dmodeds = 2**(i+0.5)*dmodeds
	return([dmodedr, dmodeds])

def GradVandermonde2D(N,r,s):
	"""function [V2Dr,V2Ds] = GradVandermonde2D(N,r,s)
	Purpose : Initialize the gradient of the modal basis (i,j) at (r,s) at order N"""
	
	V2Dr = numpy.zeros([len(r),(N+1)*(N+2)/2])
	V2Ds = numpy.zeros([len(r),(N+1)*(N+2)/2])
	
	# find tensor-product coordinates
	[a,b] = rstoab(r,s)
	
	# Initialize matrices
	sk = 0
	for i in range(N+1):
	  for j in range(N-i+1):
	    [V2Dr.transpose()[sk],V2Ds.transpose()[sk]] = GradSimplex2DP(a,b,i,j)
	    sk = sk+1
	return([V2Dr,V2Ds])

def Dmatrices2D(N,r,s,V):
	"""function [Dr,Ds] = Dmatrices2D(N,r,s,V)
	Purpose : Initialize the (r,s) differentiation matrices
	    on the simplex, evaluated at (r,s) at order N"""
	
	[Vr, Vs] = GradVandermonde2D(N, r, s)
	invV=numpy.linalg.inv(V)
	Dr = Vr.dot(invV)
	Ds = Vs.dot(invV)
	return([Dr,Ds])

def Lift2D():
	"""function [LIFT] = Lift2D()
	Purpose  : Compute surface to volume lift term for DG formulation"""
	
	import globalVar2D as glb
	import functions as f1D
	Emat = numpy.zeros([glb.Np, glb.Nfaces*glb.Nfp])
	
	# face 1
	faceR = glb.r[glb.Fmask[:,0]]
	V1D = f1D.Vandermonde1D(glb.N, faceR) 
	massEdge1 = numpy.linalg.inv(V1D.dot(V1D.transpose()))
	Emat[glb.Fmask[:,0],0:glb.Nfp] = massEdge1

	# face 2
	faceR = glb.r[glb.Fmask[:,1]]
	V1D = f1D.Vandermonde1D(glb.N, faceR) 
	massEdge2 = numpy.linalg.inv(V1D.dot(V1D.transpose()))
	Emat[glb.Fmask[:,1],glb.Nfp:2*glb.Nfp] = massEdge2

	# face 3
	faceS = glb.s[glb.Fmask[:,2]]
	V1D = f1D.Vandermonde1D(glb.N, faceS) 
	massEdge3 = numpy.linalg.inv(V1D.dot(V1D.transpose()))
	Emat[glb.Fmask[:,2],2*glb.Nfp:3*glb.Nfp] = massEdge3

	# inv(mass matrix)*\I_n (L_i,L_j)_{edge_n}
	LIFT = glb.V.dot(glb.V.transpose().dot(Emat))
	return(LIFT)

def Grad2D(u):
	"""function [ux,uy] = Grad2D(u);
	Purpose: Compute 2D gradient field of scalar u"""
	
	import globalVar2D as glb
	
	ur = glb.Dr.dot(u)
	us = glb.Ds.dot(u)
	ux = glb.rx*ur + glb.sx*us
	uy = glb.ry*ur + glb.sy*us
	return([ux,uy])

def Div2D(u,v):
	
	"""function [divu] = Div2D(u,v);
	Purpose: Compute the 2D divergence of the vectorfield (u,v)"""
	
	import globalVar2D as glb
	
	ur = glb.Dr.dot(u)
	us = glb.Ds.dot(u)
	vr = glb.Dr.dot(v)
	vs = glb.Ds.dot(v)
	divu = glb.rx*ur + glb.sx*us + glb.ry*vr + glb.sy*vs
	return(divu)	

def Curl2D(ux,uy,uz=[]):
	
	"""function [vx,vy,vz] = Curl2D(ux,uy,uz);
	Purpose: Compute 2D curl-operator in (x,y) plane"""
	
	import globalVar2D as glb
	
	uxr = glb.Dr.dot(ux)
	uxs = glb.Ds.dot(ux)
	uyr = glb.Dr.dot(uy)
	uys = glb.Ds.dot(uy)
	vz =  glb.rx*uyr + glb.sx*uys - glb.ry*uxr - glb.sy*uxs
	vx=[]
	vy=[]
	
	if (len(uz)!=0):
	  uzr = glb.Dr.dot(uz)
	  uzs = glb.Ds.dot(uz)
	  vx =  glb.ry*uzr + glb.sy*uzs
	  vy = -glb.rx*uzr - glb.sx*uzs
	return([vx,vy,vz])	

def GeometricFactors2D(x,y,Dr,Ds):
	"""function [rx,sx,ry,sy,J] = GeometricFactors2D(x,y,Dr,Ds)
	Purpose  : Compute the metric elements for the local mappings of the elements"""
	
	# Calculate geometric factors
	xr = Dr.dot(x)
	xs = Ds.dot(x)
	yr = Dr.dot(y)
	ys = Ds.dot(y)
	J = -xs*yr + xr*ys
	rx = ys/J
	sx =-yr/J
	ry =-xs/J
	sy = xr/J
	return([rx,sx,ry,sy,J])

def Normals2D():
	
	"""function [nx, ny, sJ] = Normals2D()
	Purpose : Compute outward pointing normals at elements faces and surface Jacobians"""
	import globalVar2D as glb
	xr = glb.Dr.dot(glb.x)
	yr = glb.Dr.dot(glb.y)
	xs = glb.Ds.dot(glb.x)
	ys = glb.Ds.dot(glb.y)
	glb.J = xr*ys-xs*yr
	
	# interpolate geometric factors to face nodes
	fxr = xr[glb.Fmask,:]
	fxs = xs[glb.Fmask,:]
	fyr = yr[glb.Fmask,:]
	fys = ys[glb.Fmask,:]
	
	# build normals
	nx = numpy.zeros([3*glb.Nfp, glb.K])
	ny = numpy.zeros([3*glb.Nfp, glb.K])
	fid1 = range(0,glb.Nfp)
	fid2 = range(glb.Nfp,2*glb.Nfp)
	fid3 = range(2*glb.Nfp,3*glb.Nfp)
	
	# face 1
	nx[fid1, :] =  fyr[:,0,:]
	ny[fid1, :] = -fxr[:,0,:]
	
	# face 2
	nx[fid2, :] =  fys[:,1,:]-fyr[:,1,:]
	ny[fid2, :] = -fxs[:,1,:]+fxr[:,1,:]
	
	# face 3
	nx[fid3, :] = -fys[:,2,:]
	ny[fid3, :] =  fxs[:,2,:]
	
	# normalise
	sJ = (nx*nx+ny*ny)**0.5
	nx = nx/sJ
	ny = ny/sJ
	return([nx, ny, sJ])

def Connect2D(EToV):
	"""function [EToE, EToF] = Connect2D(EToV)
	Purpose  : Build global connectivity arrays for grid based on
	           standard EToV input array from grid generator"""
	
	Nfaces = 3
	
	# Find number of elements and vertices
	K = EToV.shape[0]
	Nv = max(EToV.flatten())+1
	
	# Create face to node connectivity matrix
	TotalFaces = Nfaces*K
	
	# List of local face to local vertex connections
	vn = numpy.array([[0,1],[1,2],[0,2]])
	
	# Build global face to node sparse array
	SpFToV = numpy.zeros([TotalFaces, Nv])
	sk = 0
	for k in range(K):
	  for face in range(Nfaces):
		  SpFToV[sk, EToV[k, vn[face,:]]] = 1
		  sk = sk+1
	
	# Build global face to global face sparse array
	SpFToF = SpFToV.dot(SpFToV.transpose())- 2*numpy.eye(TotalFaces)
	
	# Find complete face to face connections
	[faces1, faces2] = numpy.nonzero(SpFToF==2)
	
	# Convert face global number to element and face numbers
	element1 = (faces1)/ Nfaces
	face1    = (faces1)% Nfaces
	element2 = (faces2)/ Nfaces
	face2    = (faces2)% Nfaces
	
	# Rearrange into Nelements x Nfaces sized arrays
	ind=numpy.ravel_multi_index((element1,face1), dims=(K,Nfaces), order='C')
	
	EToE = (numpy.array([range(K)]).transpose()).dot(numpy.ones([1,Nfaces]))
	EToF = numpy.ones([K,1]).dot(numpy.array([range(Nfaces)]))
	EToE.ravel()[ind] = element2 
	EToF.ravel()[ind] = face2
	return([EToE.astype(int),EToF.astype(int)])

def BuildMaps2D():
	"""function [mapM, mapP, vmapM, vmapP, vmapB, mapB] = BuildMaps2D
	Purpose: Connectivity and boundary tables in the K # of Np elements"""
	
	from globalVar2D import K,Nfp,Np,EToV,VX,VY,NODETOL,Nfaces,Fmask,EToE,EToF,x,y
	
	# number volume nodes consecutively
	nodeids = numpy.reshape(range(K*Np), [Np, K]).astype(int)
	vmapM   = numpy.zeros([Nfp, Nfaces, K]).astype(int)
	vmapP   = numpy.zeros([Nfp, Nfaces, K]).astype(int)
	mapM    = numpy.array(range(K*Nfp*Nfaces)).astype(int)
	mapP = numpy.reshape(mapM, [Nfaces,Nfp, K]).astype(int)
	 
	# find index of face nodes with respect to volume node ordering
	for k1 in range(K):
	  for f1 in range(Nfaces):
	    vmapM[:,f1,k1] = nodeids[Fmask[:,f1], k1]
	
	one = numpy.ones([1, Nfp])
	for k1 in range(K):
	  for f1 in range(Nfaces):
	    # find neighbor
	    k2 = EToE[k1,f1]
	    f2 = EToF[k1,f1]
	    
	    # reference length of edge
	    v1 = EToV[k1,f1]
	    v2 = EToV[k1, (f1+1)%3]
	    refd = ( (VX[v1]-VX[v2])**2 + (VY[v1]-VY[v2])**2 )**0.5
	
	    # find find volume node numbers of left and right nodes 
	    vidM = vmapM[:,f1,k1]
	    vidP = vmapM[:,f2,k2]

	    x1 = x.flatten()[vidM]; y1 = y.flatten()[vidM]
	    x2 = x.flatten()[vidP]; y2 = y.flatten()[vidP]
	    x1 = numpy.array([x1]).transpose().dot(one)
	    y1 = numpy.array([y1]).transpose().dot(one)
	    x2 = numpy.array([x2]).transpose().dot(one)
	    y2 = numpy.array([y2]).transpose().dot(one)

	    # Compute distance matrix
	    D = (x1 -x2.transpose())**2 + (y1-y2.transpose())**2
	    [idM, idP] = numpy.nonzero((abs(D)**0.5)<NODETOL*refd)
	    vmapP[idM,f1,k1] = vidP[idP]
	    mapP[f1,idM,k1] = mapM.reshape([Nfaces,Nfp,K])[f2,idP,k2]
	
	# reshape vmapM and vmapP to be vectors and create boundary node list
	newvmapM=numpy.zeros([Nfaces,Nfp,K]).astype(int)
	newvmapP=numpy.zeros([Nfaces,Nfp,K]).astype(int)
	oldvmapM=vmapM.reshape([Nfp,Nfaces,K])
	oldvmapP=vmapP.reshape([Nfp,Nfaces,K])
	for i in range(Nfaces):
		for j in range(Nfp):
			for k in range(K):
				newvmapM[i][j][k]=oldvmapM[j][i][k]
				newvmapP[i][j][k]=oldvmapP[j][i][k]

	vmapP = newvmapP.flatten()
	vmapM = newvmapM.flatten()
	mapP = mapP.flatten()
	mapM = mapM.flatten()
	mapB = numpy.nonzero(vmapP==vmapM)[0]
	vmapB = vmapM[mapB]
	return([mapM, mapP, vmapM, vmapP, vmapB, mapB])	

def dtscale2D():
	"""function dtscale = dtscale2D;
	Purpose : Compute inscribed circle diameter as characteristic
	          for grid to choose timestep"""
	
	import globalVar2D as glb
	
	# Find vertex nodes
	vmask1   = numpy.nonzero(abs(glb.s+glb.r+2.0) < glb.NODETOL)[0][0]
	vmask2   = numpy.nonzero( abs(glb.r-1.0)   < glb.NODETOL)[0][0]
	vmask3   = numpy.nonzero( abs(glb.s-1.0)   < glb.NODETOL)[0][0]
	vmask  = numpy.array([vmask1,vmask2,vmask3])
	vx = glb.x[vmask]
	vy = glb.y[vmask]
	
	# Compute semi-perimeter and area
	len1 = ((vx[0]-vx[1])**2+(vy[0]-vy[1])**2)**0.5
	len2 = ((vx[2]-vx[1])**2+(vy[2]-vy[1])**2)**0.5
	len3 = ((vx[0]-vx[2])**2+(vy[0]-vy[2])**2)**0.5
	sper = (len1 + len2 + len3)/2.0
	Area = (sper*(sper-len1)*(sper-len2)*(sper-len3))**0.5
	
	# Compute scale using radius of inscribed circle
	dtscale = Area/sper
	return(dtscale)

def BuildBCMaps2D():
	"""function BuildMaps2DBC
	Purpose: Build specialized nodal maps for various types of
	         boundary conditions, specified in BCType."""
	
	import globalVar2D as glb
	
	# create label of face nodes with boundary types from BCType
	#BCType is a Kx3 matrix containing boundary type at each point
	bct    = glb.BCType.transpose()
	bnodes=[bct for i in range(glb.Nfp)]
	bnodes = numpy.array(bnodes)
	
	bnodes=bnodes.reshape([glb.Nfp,glb.Nfaces,glb.K])
	newbnodes=numpy.zeros([glb.Nfaces,glb.Nfp,glb.K])
	for i in range(glb.Nfaces):
		for j in range(glb.Nfp):
			for k in range(glb.K):
				newbnodes[i][j][k]=bnodes[j][i][k]
	bnodes=newbnodes.flatten()

	
	# find location of boundary nodes in face and volume node lists
	glb.mapI = numpy.nonzero(bnodes==glb.In)[0];           glb.vmapI = glb.vmapM[glb.mapI];
	glb.mapO = numpy.nonzero(bnodes==glb.Out)[0];          glb.vmapO = glb.vmapM[glb.mapO];
	glb.mapW = numpy.nonzero(bnodes==glb.Wall)[0];         glb.vmapW = glb.vmapM[glb.mapW];
	glb.mapF = numpy.nonzero(bnodes==glb.Far)[0];          glb.vmapF = glb.vmapM[glb.mapF];
	glb.mapC = numpy.nonzero(bnodes==glb.Cyl)[0];          glb.vmapC = glb.vmapM[glb.mapC];
	glb.mapD = numpy.nonzero(bnodes==glb.Drichlet)[0];    glb.vmapD = glb.vmapM[glb.mapD];
	glb.mapN = numpy.nonzero(bnodes==glb.Neuman)[0];       glb.vmapN = glb.vmapM[glb.mapN];
	glb.mapS = numpy.nonzero(bnodes==glb.Slip)[0];         glb.vmapS = glb.vmapM[glb.mapS];
	return()

def advecRHS2D(u,a,timelocal):
	
	"""calculates Advection RHS for 2 dimensions"""
	
	import globalVar2D as glb
	[ax,ay]=a
	# Define field differences at faces
	du = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	du = (u.flatten()[glb.vmapM]-u.flatten()[glb.vmapP]).reshape([glb.Nfp*glb.Nfaces,glb.K])
	
	# Impose Boundaries
	#du.ravel()[glb.mapB]=0.0
	uin = [advecSinInlet(glb.x.flatten()[glb.vmapInB[i]],glb.y.flatten()[glb.vmapInB[i]],a,timelocal) for i in range(len(glb.vmapInB))]
	du.ravel()[glb.mapInB] = (u.flatten()[glb.vmapInB]- uin )
	du.ravel()[glb.mapOutB]=0.0

	# evaluate upwind fluxes
	adotn=ax*glb.nx+ay*glb.ny
	alpha = 0.0
	fluxu = du*(adotn-(1-alpha)*abs(adotn))/2
	# local derivatives of field
	divf=Div2D(ax*u,ay*u)
	
	# compute right hand sides of the PDE's
	rhsu =  divf + glb.LIFT.dot(glb.Fscale*fluxu)/2.0
	return(rhsu)

def advec2D(u, FinalTime,a):
	"""function [u] = Advec2D(u, FinalTime)
	Purpose  : Integrate 2D advection until FinalTime starting with
	           initial the condition, u"""
	
	import globalVar2D as glb
	time = 0.0
	
	# Runge-Kutta residual storage  
	resu = numpy.zeros([glb.Np,glb.K])
	
	# compute time step size
	rLGL = functions.JacobiGQ(0,0,glb.N); rmin = abs(rLGL[0]-rLGL[1]).min();
	dtscale = dtscale2D(); dt = (dtscale.min())*rmin*2/3
	
	# outer time step loop 
	while (time<FinalTime):
		if(time+dt>FinalTime):
			dt = FinalTime-time
		for INTRK in range(5):
			timelocal = time + glb.rk4c[INTRK]*dt
			rhsu = advecRHS2D(u,a,timelocal)
			resu = glb.rk4a[INTRK]*resu + dt*rhsu
			u = u+glb.rk4b[INTRK]*resu
		# Increment time
		time = time+dt
		print "time=%f"%time
	return(u)

def eval2D(x,y,u):
	"""evaluate u at point(x,y)"""
	import globalVar2D as glb
	va = glb.EToV[:,0]
	vb = glb.EToV[:,1]
	vc = glb.EToV[:,2]
	#Find which element it is in
	k=findElement(x,y)
	
	# If there is no element containing this value of x and y, return a constant = 0 value
	if type(k) == str:
		return(0.)

	umodal=glb.invV.dot(u).transpose()[k]
	v1=numpy.array([glb.VX[va[k]],glb.VY[va[k]]])
	v2=numpy.array([glb.VX[vb[k]],glb.VY[vb[k]]])
	v3=numpy.array([glb.VX[vc[k]],glb.VY[vc[k]]])
	#Convert to r,s
	[r,s]=changeFrame(x,y,v1,v2,v3)
	#Convert to a,b
	[a,b]=rstoab(r,s)
	unew=0.0
	N=glb.N
	sk=0
	for i in range(N+1):
	  for j in range(0,N-i+1):
	    unew = unew + umodal[sk]*Simplex2DP(a,b,i,j)[0]
	    sk = sk+1
	return(unew)

def findElement(x,y):
	"""find the element in which (x,y) belongs"""
	import globalVar2D as glb
	va = glb.EToV[:,0]
	vb = glb.EToV[:,1]
	vc = glb.EToV[:,2]
	for k in range(glb.K):
		v1=numpy.array([glb.VX[va][k],glb.VY[va][k]])
		v2=numpy.array([glb.VX[vb][k],glb.VY[vb][k]])
		v3=numpy.array([glb.VX[vc][k],glb.VY[vc][k]])
		if (inTriangle(x,y,v1,v2,v3)):
			return(k)
	Val = "Point(%f,%f) Not found" %(x, y)
	return(Val)

def inTriangle(x,y,v1,v2,v3):
	"""Check wheather (x,y) is in triangle (v1,v2,v3)"""
	import globalVar2D as glb
	tol=glb.NODETOL
	mat=numpy.array([[1.,1.,1.],[v3[0],v1[0],v2[0]],[v3[1],v1[1],v2[1]]])
	b=numpy.array([1,x,y])
	[L1,L2,L3]=numpy.linalg.solve(mat,b)
	return((0-tol<=L1<=1+tol) and (0-tol<=L2<=1+tol) and (0-tol<=L3<=1+tol))
	
def changeFrame(x,y,v1,v2,v3):
	"""return (r,s) for (x,y) in vnew vortices frame"""
	mat=numpy.array([[1.,1.,1.],[v3[0],v1[0],v2[0]],[v3[1],v1[1],v2[1]]])
	b=numpy.array([1,x,y])
	[L1,L2,L3]=numpy.linalg.solve(mat,b)
	r = -L2 + L3 - L1
	s = -L2 - L3 + L1
	return(numpy.array([r,s]))

def findBoundary(f):
	"""" this is just a temporary function in order to find a specific boundary map.
	f(x,y) is a function which tells whether a point is on a particular boundary or not"""
	import globalVar2D as glb
	mapSpB=[]
	for i in range(len(glb.vmapM)):
		if f(glb.x.flatten()[glb.vmapM[i]],glb.y.flatten()[glb.vmapM[i]]):
			mapSpB.append(i)
	vmapSpB=glb.vmapM[mapSpB]
	return(mapSpB,vmapSpB)

def outBoundary(x,y):
	""""Returns true if the point is on a specific boundary"""
	import globalVar2D as glb
	if abs(x-1)<glb.NODETOL:
		return(True)
	if abs(y-1)<glb.NODETOL:
		return(True)
	else:
		return(False)

def inBoundary(x,y):
	""""Returns true if the point is on a specific boundary"""
	import globalVar2D as glb
	if abs(x+1)<glb.NODETOL:
		return(True)
	if abs(y+1)<glb.NODETOL:
		return(True)
	else:
		return(False)

def advecSinInlet(x,y,a,t):
	"""Boundary condition for advection with sin initial condition"""
	ax=a[0]
	ay=a[1]
	val=math.sin(math.pi*(x-ax*t))*math.sin(math.pi*(y-ay*t))
	return(val)

def plot2Da(u):
	""" Plots for 2D variables. These are quick plots, with no interpolation, for better quality plots see plot2Db(u,N)"""
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	import globalVar2D as glb
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(glb.x,glb.y,u,c=u,cmap=cm.jet)
	plt.show()

def plot2Db(u,N):
	"""Plots for 2D variables, this are good quality plots and might take some time depending on N value. Go for plot2Da(u) if u need a quick plot, or decrease N value"""
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import matplotlib.pyplot as plt
	import globalVar2D as glb
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	dx=(glb.x.max()-glb.x.min())/N
	dy=(glb.x.max()-glb.x.min())/N
	X = numpy.arange(glb.x.min(), glb.x.max()+dx, dx)
	Y = numpy.arange(glb.y.min(), glb.y.max()+dy, dy)
	X, Y = numpy.meshgrid(X, Y)
	Z=numpy.copy(X)
	for i in range(len(X.flatten())):
		x=X.flatten()[i]
		y=Y.flatten()[i]
		Z.ravel()[i]=eval2D(float(x),float(y),u)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	
	fig.colorbar(surf, shrink=0.5, aspect=5)
	
	plt.show()
	return()

def contour2Da(u):
	""" Plots 2D contour plots for 2D variables. These are quick plots, with no interpolation, for better quality plots see contour2Db(u,N)"""
	import matplotlib.pyplot as plt
	plt.figure()
	import globalVar2D as glb
	plt.scatter(glb.x.flatten(),glb.y.flatten(),c = u.flatten())
	plt.show()

def contour2Db(u,N,Nu):
	"""Plots contours for 2D variables, this are good quality plots and might take some time depending on N value. Go for contour2Da(u) if u need a quick plot, or decrease N value"""
	import matplotlib
	import matplotlib.cm as cm
	import matplotlib.mlab as mlab
	import matplotlib.pyplot as plt
	import globalVar2D as glb

	global dx, dy, X, Y	
	# Define meths grid and find values at all the points
	dx=(glb.x.max()-glb.x.min())/N
	dy=(glb.x.max()-glb.x.min())/N
	X = numpy.arange(glb.x.min(), glb.x.max()+dx, dx)
	Y = numpy.arange(glb.y.min(), glb.y.max()+dy, dy)
	X, Y = numpy.meshgrid(X, Y)
	Z=numpy.copy(X)
	for i in range(len(X.flatten())):
		x=X.flatten()[i]
		y=Y.flatten()[i]
		Z.ravel()[i]=eval2D(float(x),float(y),u)
	
	# Plot X,Y,Z in contour plot
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'

	# Define levels for contours
	du = (u.max()-u.min())/Nu
	levels = numpy.arange(u.min(), u.max()+du, du)
	
	plt.figure()
	CS = plt.contourf(X, Y, Z, levels)
	plt.colorbar(CS)

	# Plot boundary points
	plt.scatter(glb.x.flatten()[glb.vmapB],glb.y.flatten()[glb.vmapB],s=8)
	plt.show()

	return()

def cutOffFilter2D(Nc,frac):
	
	""" function [F] = CutOffFilter2D(Nc,frac)
	 Purpose : Initialize 2D cut off filter matrix of cutoff Nc and frac"""
	
	import globalVar2D as glb
	
	filterdiag = numpy.ones(glb.Np)
	
	# build exponential filter
	sk = 0
	for i in range(glb.N+1):
		for j in range(glb.N-i+1):
			if (i+j>=Nc):
				filterdiag[sk] = frac
			sk = sk+1
	F = glb.V.dot(numpy.diag(filterdiag).dot(glb.invV))
	return(F)

def cubData():
	"""Reads cubature data from file 'cubData2D.m' """
	import oct2py
	oc=oct2py.Oct2Py()
	oc.addpath('/home/achyut/ddp/galerkin/newCodes')
	cub2D=oc.cubData2D()
	
	#change structure for python compatibility
	cub2D[0]=numpy.array([cub2D[0]])
	return(cub2D)

def cubature2D(Corder):
	
	"""function [cubR,cubS,cubW, Ncub] = Cubature2D(Corder)
		Purpose: provide multidimensional quadrature (i.e. cubature) 
		rules to integrate up to Corder polynomials"""
	 
	import functions
	 
	cub2D=cubData() 
	if(Corder<=28):
		cubR = cub2D[Corder-1][:,0]
		cubS = cub2D[Corder-1][:,1]
		cubW = cub2D[Corder-1][:,2]
	else:
		cubNA = int(numpy.ceil( (Corder+1.)/2.))
		[cubA,cubWA] = functions.JacobiGQ(0.,0., cubNA-1)
		cubNB = int(numpy.ceil( (Corder+1.)/2.))
		[cubB,cubWB] = functions.JacobiGQ(1.,0., cubNB-1)
		cubA1 = numpy.ones([cubNB,1]).dot(numpy.array([cubA]))
		cubB1 = numpy.array([cubB]).transpose().dot(numpy.ones([1,cubNA]))
		
		cubR = 0.5*(1+cubA1)*(1-cubB1)-1.
		cubS = cubB1.copy()
		cubW = 0.5*numpy.array([cubWB]).transpose().dot((numpy.array([cubWA])))
		
		cubR = cubR.transpose().flatten()
		cubS = cubS.transpose().flatten()
		cubW = cubW.transpose().flatten()
	
	Ncub = len(cubW)
	return([cubR,cubS,cubW, Ncub])
	
def InterpMatrix2D(rout, sout):
	
	"""function [IM] = InterpMatrix2D(rout, sout)
	Purpose: Compute local elemental interpolation matrix"""
	
	import globalVar2D as glb
	 
	# compute Vandermonde at (rout,sout)
	Vout = Vandermonde2D(glb.N, rout, sout)
	
	# build interpolation matrix
	IM = Vout.dot(glb.invV)
	return(IM)
