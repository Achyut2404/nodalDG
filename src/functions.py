# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Initialization functions for 1-dimensional Nodal DG formulation

import numpy
import math

def JacobiGQ(alpha,beta,N):
	"""function [x,w] = JacobiGQ(alpha,beta,N)
	Purpose: Compute the N'th order Gauss quadrature points, x, 
	     and weights, w, associated with the Jacobi 
	          polynomial, of type (alpha,beta) > -1 ( <> -0.5)."""
	
	alpha=float(alpha)
	beta=float(beta)
	eps=2.2204e-16
	
	if (N==0):
		x=numpy.zeros(1)
		w=numpy.zeros(1)
		x[0]= -(alpha-beta)/(alpha+beta+2)
		w[0] = 2.0
		return([x,w])
	
	# Form symmetric matrix from recurrence.
	J = numpy.zeros(N+1)
	h1 = 2.0*numpy.array(range(0,N+1))+(alpha*numpy.ones(N+1))+(beta*numpy.ones(N+1))
	J = (numpy.diag(-1/2.0*(alpha**2-beta**2)/(h1+2)/h1)+
	numpy.diag(2.0/(h1[0:N]+2)*((numpy.array(range(1,N+1))*(numpy.array(range(1,N+1))+alpha+beta)*
	(numpy.array(range(1,N+1))+alpha)*(numpy.array(range(1,N+1))+beta)/(h1[0:N]+1)/(h1[0:N]+3.0))**0.5),1))
	
	if (alpha+beta<10*eps):
		J[0,0]=0.0
	J = J + J.transpose()

	#Compute quadrature by eigenvalue solve
	[D,V] = numpy.linalg.eig(J) 
	D_sorted = numpy.sort(D)
	V_sorted = V[:,D.argsort()]
	D=D_sorted
	V=V_sorted
	x = D
	w = ((V[0,:].transpose())**2)*2**(alpha+beta+1)/(alpha+beta+1)*math.gamma(alpha+1)*math.gamma(beta+1)/math.gamma(alpha+beta+1)
	return([x,w])
	
def JacobiP(x,alpha,beta,N):
	""" function [P] = JacobiP(x,alpha,beta,N)
	Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
	          (alpha+beta <> -1) at points x for order N and returns P[1:length(xp))]
	Note   : They are normalized to be orthonormal."""
	
	# Turn points into row if needed.
	xp = numpy.copy(x) 
	xp = xp.flatten()
	
	alpha=float(alpha)
	beta=float(beta)

	PL = numpy.zeros([N+1,numpy.size(xp)])
	
	# Initial values P_0(x) and P_1(x)
	gamma0 = 2.0**(alpha+beta+1)/(alpha+beta+1)*math.gamma(alpha+1)*math.gamma(beta+1)/math.gamma(alpha+beta+1)
	PL[0,:] = 1.0/math.sqrt(gamma0)
	if (N==0):
		P=PL.transpose()
		return(P[0])
	gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
	PL[1,:] = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/math.sqrt(gamma1)
	if (N==1):
		P=PL[N,:].transpose()
		return(P)
	
	# Repeat value in recurrence.
	aold = 2/(2+alpha+beta)*math.sqrt((alpha+1)*(beta+1)/(alpha+beta+3));
	
	# Forward recurrence using the symmetry of the recurrence.
	for i in range(1,N):
		h1 = 2*i+alpha+beta
		anew = 2/(h1+2)*math.sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
		bnew = - (alpha**2-beta**2)/h1/(h1+2)
		PL[i+1,:] = 1/anew*( -aold*PL[i-1,:] + (xp-bnew)*PL[i,:])
		aold =anew
	P = PL[N,:].transpose();	
	return(P)

def JacobiGL(alpha,beta,N):
	"""function [x] = JacobiGL(alpha,beta,N)
	Purpose: Compute the N'th order Gauss Lobatto quadrature 
	         points, x, associated with the Jacobi polynomial,
	         of type (alpha,beta) > -1 ( <> -0.5)."""
	
	x = numpy.zeros([N+1,1])
	if (N==1):
		x[0]=-1.0
		x[1]=1.0
		return(x.transpose()[0])
	
	[xint,w] = JacobiGQ(alpha+1,beta+1,N-2);
	x = numpy.concatenate([numpy.array([-1.0]), xint.transpose(), numpy.array([1.0])]).transpose();
	return(x)

def GradJacobiP(r, alpha, beta, N):
	"""function [dP] = GradJacobiP(r, alpha, beta, N);
	Purpose: Evaluate the derivative of the Jacobi polynomial of type (alpha,beta)>-1,
	       at points r for order N and returns dP[1:length(r))]"""
	r=numpy.array(r)
	dP = numpy.zeros([len(r),1])
	if(N == 0):
		dP[:,:] = 0.0 
		dP=dP[:,0]
	else:
		dP = math.sqrt(N*(N+alpha+beta+1))*JacobiP(r[:],alpha+1,beta+1, N-1)
	return(dP)	

def Vandermonde1D(N,r):
	"""function [V1D] = Vandermonde1D(N,r)
	Purpose : Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i);"""
	
	r=numpy.array(r)
	V1D = numpy.zeros([r.size,N+1])
	for j in range(N+1):
		V1D[:,j] = JacobiP(r.flatten(), 0, 0, j)
	return(V1D)

def GradVandermonde1D(N,r):
	"""function [DVr] = GradVandermonde1D(N,r)
	Purpose : Initialize the gradient of the modal basis (i) at (r) at order N"""
	
	r=numpy.array(r)
	DVr = numpy.zeros([len(r),(N+1)]);
	
	# Initialize matrix
	for i in range(N+1):
		DVr[:,i] = GradJacobiP(r[:],0.,0.,i)
	if N==0:
		DVr=DVr[:,0]
	return(DVr)
	
def Dmatrix1D(N,r,V):
	"""function [Dr] = Dmatrix1D(N,r,V)
	Purpose : Initialize the (r) differentiation matrices on the interval,
	        evaluated at (r) at order N"""
	
	Vr = GradVandermonde1D(N, r)
	Dr = Vr.dot((numpy.linalg.inv(V)))
	return(Dr)

def Lift1D():
	"""function [LIFT] = Lift1D
	Purpose  : Compute surface integral term in DG formulation"""
	
	import globalVar
	Emat = numpy.zeros([globalVar.Np,(globalVar.Nfaces)*(globalVar.Nfp)])
	# Define Emat
	Emat[0,0] = 1.0
	Emat[globalVar.Np-1,1] = 1.0
	
	# inv(mass matrix)*\s_n (L_i,L_j)_{edge_n}
	LIFT = globalVar.V.dot((globalVar.V.transpose().dot(Emat)))
	return(LIFT)

def GeometricFactors1D(x,Dr):
	"""function [rx,J] = GeometricFactors1D(x,Dr)
	Purpose  : Compute the metric elements for the local mappings of the 1D elements"""
	xr  = Dr.dot(x) 
	J = xr
	rx = 1/J
	return([rx,J])

def Normals1D():
	"""function [nx] = Normals1D
	Purpose : Compute outward pointing normals at elements faces"""
	
	import globalVar
	nx = numpy.zeros([globalVar.Nfp*globalVar.Nfaces, globalVar.K])
	
	# Define outward normals
	nx[0,:] = -1.0
	nx[1,:] = 1.0
	return(nx)

def Connect1D(EToV):
	"""function [EToE, EToF] = Connect1D(EToV)
	Purpose  : Build global connectivity arrays for 1D grid based on standard 
	           EToV input array from grid generator"""
	
	Nfaces = 2
	# Find number of elements and vertices
	K = EToV.shape[0] 
	TotalFaces = Nfaces*K 
	Nv = K+1
	
	# List of local face to local vertex connections
	vn = numpy.array([0,1])
	
	# Build global face to node sparse array
	SpFToV = numpy.zeros([TotalFaces, Nv])
	sk = 0
	for k in range(K):
	  for face in range(Nfaces):
	     SpFToV[ sk, EToV[k, vn[face]]] = 1
	     sk = sk+1
	     
	# Build global face to global face sparse array
	SpFToF = SpFToV.dot(SpFToV.transpose()) - numpy.eye(TotalFaces)
	
	# Find complete face to face connections
	[faces1, faces2] = numpy.nonzero(SpFToF==1)
	
	# Convert face global number to element and face numbers
	element1 = (faces1)/ Nfaces
	face1    = (faces1)% Nfaces
	element2 = (faces2)/ Nfaces
	face2    = (faces2)% Nfaces
	
	# Rearrange into Nelements x Nfaces sized arrays
	ind=numpy.ravel_multi_index((element1,face1), dims=(K,Nfaces), order='C')
	EToE      = (numpy.array([range(K)]).transpose()).dot(numpy.ones([1,Nfaces]))
	EToF      = numpy.ones([K,1]).dot(numpy.array([range(Nfaces)]))
	EToE.ravel()[ind] = element2 
	EToF.ravel()[ind] = face2
	return([EToE.astype(int),EToF.astype(int)])

def BuildMaps1D():

	"""function [vmapM, vmapP, vmapB, mapB] = BuildMaps1D
	Purpose: Connectivity and boundary tables for nodes given in the K # of elements,
		       each with N+1 degrees of freedom."""
	
	import globalVar
	
	# number volume nodes consecutively
	nodeids = numpy.array(range(globalVar.K*globalVar.Np)).reshape(globalVar.Np, globalVar.K).astype(int)
	vmapM   = numpy.zeros([globalVar.Nfaces, globalVar.K]).astype(int)
	vmapP   = numpy.zeros([globalVar.Nfaces, globalVar.K]).astype(int)

	for k1 in range(globalVar.K):
	  for f1 in range(globalVar.Nfaces):
	    # find index of face nodes with respect to volume node ordering
	    vmapM[f1,k1] = nodeids[globalVar.Fmask[f1], k1]
	for k1 in range(globalVar.K):
	  for f1 in range(globalVar.Nfaces):
	    # find neighbor
	    k2 = globalVar.EToE[k1,f1] 
	    f2 = globalVar.EToF[k1,f1]
	    
	    # find volume node numbers of left and right nodes 
	    vidM = vmapM[f1,k1]
	    vidP = vmapM[f2,k2]
	    
	    x1  = globalVar.x.flatten()[vidM]
	    x2  = globalVar.x.flatten()[vidP]

	    # Compute distance matrix
	    D = (x1 -x2)**2
	    if (D<globalVar.NODETOL):
			vmapP[f1,k1] = vidP
	
	vmapP = vmapP[:] 
	vmapM = vmapM[:]
	
	# Create list of boundary nodes
	mapB = numpy.nonzero(vmapP==vmapM) 
	vmapB = vmapM[mapB]
		
	# Create specific left (inflow) and right (outflow) maps
	globalVar.mapI = 0 
	globalVar.mapO = (globalVar.K*globalVar.Nfaces)-1
	globalVar.vmapI = 0
	globalVar.vmapO = (globalVar.K*globalVar.Np)-1
	return([vmapM, vmapP, vmapB, mapB])

def MeshGen1D(xmin,xmax,K):
	"""function [Nv, VX, K, EToV] = MeshGen1D(xmin,xmax,K)
	Purpose  : Generate simple equidistant grid with K elements"""
	
	Nv = K+1
	
	# Generate node coordinates
	VX = numpy.zeros(Nv)
	for i in range(Nv):
	  VX[i] = (xmax-xmin)*(float(i))/(Nv-1) + xmin
	
	# read element to node connectivity
	EToV = numpy.zeros([K, 2])
	for k in range(K):
	  EToV[k,0] = k
	  EToV[k,1] = k+1
	return([Nv,VX,K,EToV.astype(int)])

def advecRHS1D(u,time, a):
	"""function [rhsu] = AdvecRHS1D(u,time)
	Purpose  : Evaluate RHS flux in 1D advection"""
	
	import globalVar as glb
	
	# form field differences at faces
	alpha=1
	du = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K]) 
	du[:] = (u.flatten()[glb.vmapM]-u.flatten()[glb.vmapP])*(a*glb.nx[:]-(1-alpha)*abs(a*glb.nx[:]))/2
	# impose boundary condition at x=0
	
	#uin = -math.sin(a*time)
	#du.ravel()[glb.mapI] = (u.flatten()[glb.vmapI]- uin)*(a*glb.nx.flatten()[glb.mapI]-(1-alpha)*abs(a*glb.nx.flatten()[glb.mapI]))/2
	#du.ravel()[glb.mapO] = 0.0
	
	#Set periodic boundary condition
	uin=u.flatten()[glb.vmapO]
	uout=u.flatten()[glb.vmapI]
	du.ravel()[glb.mapI] = (u.flatten()[glb.vmapI]- uin)*(a*glb.nx.flatten()[glb.mapI]-(1-alpha)*abs(a*glb.nx.flatten()[glb.mapI]))/2
	du.ravel()[glb.mapO] = (u.flatten()[glb.vmapO]- uout)*(a*glb.nx.flatten()[glb.mapO]-(1-alpha)*abs(a*glb.nx.flatten()[glb.mapO]))/2

	# compute right hand sides of the semi-discrete PDE
	rhsu = -a*glb.rx*(glb.Dr.dot(u)) + glb.LIFT.dot(glb.Fscale*(du))
	return(rhsu)	

def advec1D(u, FinalTime):
	"""function [u] = Advec1D(u, FinalTime)
	Purpose  : Integrate 1D advection until FinalTime starting with
	           initial the condition, u"""
	
	import globalVar as glb
	time = 0
	# Runge-Kutta residual storage  
	resu = numpy.zeros([glb.Np,glb.K])

	# compute time step size
	xmin = min(abs(glb.x[0,:]-glb.x[1,:]))
	CFL=0.75
	dt   = CFL/(2*math.pi)*xmin
	dt = 0.5*dt
	Nsteps = int(math.ceil(FinalTime/dt))
	dt = FinalTime/Nsteps 
	
	# advection speed
	a = 2*math.pi
	
	# outer time step loop 
	for tstep in range(Nsteps):
		for INTRK in range(5):
			timelocal = time + glb.rk4c[INTRK]*dt
			rhsu = advecRHS1D(u, timelocal, a)
			resu = glb.rk4a[INTRK]*resu + dt*rhsu
			u = u+glb.rk4b[INTRK]*resu
	    # Increment time
		time = time+dt
	return(u)

def step(x,a=0.5):
	return(float(x>a))
