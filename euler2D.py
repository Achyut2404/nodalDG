# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Functions for solving 2-dimensional Euler equations

import math
import numpy
import functions2D
import mesh2D
import matplotlib.pyplot as plt

def isentropicVortexIC2D(x, y, time):
	 
	""" function Q = IsentropicVortexIC2D(x, y)
	 Purpose: compute flow configuration given by
	     Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159 """
	
	# based flow parameters
	x0 = 5.; y0 = 0.; beta = 5.; gamma = 1.4;
	rho = 1.; u = 1.; v = 0.; p = 1.;
	
	xmut = x-u*time; ymvt = y-v*time;
	r = ((xmut-x0)**2 + (ymvt-y0)**2)**0.5

	# perturbed density
	u   = u - beta*numpy.vectorize(math.exp)(1-r**2)*(ymvt-y0)/(2*math.pi)
	v   = v + beta*numpy.vectorize(math.exp)(1-r**2)*(xmut-x0)/(2*math.pi)
	rho1 = (1. - ((gamma-1.)*(beta**2.)*(numpy.vectorize(math.exp)(2.*(1.-r**2.)))/(16.*gamma*math.pi*math.pi)))**(1./(gamma-1))
	p1   = rho1**gamma
	
	#Calculate final Q
	Q=numpy.zeros([x.shape[0],x.shape[1],4])
	
	Q[:,:,0] = rho1
	Q[:,:,1] = rho1*u
	Q[:,:,2] = rho1*v
	Q[:,:,3] = p1/(gamma-1) + 0.5*rho1*(u**2 + v**2)
	return(Q)

def isentropicVortexBC2D(xin, yin, nxin, nyin, mapI, mapO, mapW, mapC, Q, time):
	"""function [Q] = IsentropicVortexBC2D(xin, yin, nxin, nyin, mapI, mapO, mapW, mapC, Q, time);
	Purpose: Impose boundary conditions on 2D Euler equations on weak form"""
	
	Qbc = isentropicVortexIC2D(xin, yin, time)
	mapB = numpy.concatenate([mapI,mapO,mapW])
	
	for n in range(4):
		Qn = Q[:,:,n].copy()
		Qbcn = Qbc[:,:,n]
		Qn.ravel()[mapB] = Qbcn.flatten()[mapB]
		Q[:,:,n] = Qn
	
	return(Q)

def testEuler(order=9):
	"""Driver script for solving the 2D Euler  Isentropic vortex equations"""
	import globalVar2D as glb
	glb.globalInit()
	import mesh2D

	# Order of polynomials used for approximation 
	glb.N = order
	
	# Read in Mesh
	filename = 'Grid/Euler2D/vortexA04.neu'
	InitialSolution = isentropicVortexIC2D
	ExactSolution   = isentropicVortexIC2D
	BCSolution      = isentropicVortexBC2D

	# read mesh from file
	[glb.Nv, glb.VX, glb.VY, glb.K, glb.EToV, glb.BCType] = mesh2D.createBC(filename)
	
	# set up nodes and basic operations
	execfile("initiate2D.py")
	
	# turn cylinders into walls
	ids = numpy.nonzero(glb.BCType==glb.Cyl) 
	glb.BCType[ids] = glb.Wall
	
	functions2D.BuildBCMaps2D()
	
	# compute initial condition
	Q = InitialSolution(glb.x, glb.y, 0.)
	
	# Solve Problem
	FinalTime = 1.0
	Q = eulerSolve(Q, FinalTime, BCSolution)
	
	# Calculate error
	err=Q-ExactSolution(glb.x,glb.y,FinalTime)
	L2Err=[(numpy.average((err[:,:,i]**2).flatten()))**0.5 for i in range(4)]
	
	return(Q,L2Err)

def eulerDT2D(Q, gamma):
	""" dt = eulerDT2D(Q, gamma)
	 purpose: compute the time step dt for the compressible Euler equations"""
	
	import globalVar2D as glb
	
	rho = Q[:,:,0]
	rhou = Q[:,:,1]
	rhov = Q[:,:,2]
	Ener = Q[:,:,3]
	rho1 = rho.flatten()[glb.vmapM]
	rhou1 = rhou.flatten()[glb.vmapM]
	rhov1 = rhov.flatten()[glb.vmapM]
	Ener1 = Ener.flatten()[glb.vmapM]
	
	u = rhou1/rho1
	v = rhov1/rho1
	p = (gamma-1.0)*(Ener1 - rho1*(u**2+v**2)/2)
	c = (abs(gamma*p/rho1))**0.5
	
	dt = 1./max(((glb.N+1)**2)*.5*glb.Fscale.flatten()*( ( u**2 + v**2 )**0.5 + c))
	rhoprange = [min(rho1),max(rho1),min(p),max(p)]
	return(dt)

def eulerSolve(Q,FinalTime,BCSolution):
	"""Q = eulerSolve(Q,FinalTime,BC)
	Purpose  : Integrate 2D Euler equations using a 3rd order SSP-RK"""
	
	import globalVar2D as glb
	
	# Initialize filter
	Filt = functions2D.cutOffFilter2D(glb.N,0.95)
	
	# compute initial timestep
	gamma = 1.4
	dt = eulerDT2D(Q, gamma)
	time = 0.
	tstep=1
	
	# storage for low storage RK time stepping
	rhsQ = 0.*Q
	resQ = 0.*Q
	
	global errors
	errors=[]
	
	# filter initial solution
	for n in range(4):
		Q[:,:,n] = Filt.dot(Q[:,:,n])
	# outer time step loop 
	while (time<FinalTime):
		print "time=%f"%time

		# check to see if we need to adjust for final time step
		if(time+dt>FinalTime):
			dt = FinalTime-time
		
		for INTRK in range(5):
			# compute right hand side of compressible Euler equations
			rhsQ  = EulerRHS2D(Q, time, BCSolution)
		
			#filter residual
			for n in range(4):
				rhsQ[:,:,n] = Filt.dot(rhsQ[:,:,n])
			
			# initiate and increment Runge-Kutta residuals
			resQ = glb.rk4a[INTRK]*resQ + dt*rhsQ 

			# update fields
			Q = Q+glb.rk4b[INTRK]*resQ  

		#Calculate errors
		err=Q-isentropicVortexIC2D(glb.x,glb.y,time+dt)
		L2Err=[(numpy.average((err[:,:,i]**2).flatten()))**0.5 for i in range(4)]
		errors.append(L2Err)
		# Increment time and compute new timestep
		time = time+dt
		dt = eulerDT2D(Q, gamma)
		tstep = tstep+1
	return(Q)

def EulerRHS2D(Q,time, ExactSolutionBC):
	
	"""rhsQ = EulerRHS2D(Q,time, ExactSolutionBC);
	Purpose: Evaluate RHS in 2D Euler equations, discretized on weak form
	            with a local Lax-Friedrich flux"""
	
	import globalVar2D as glb
	
	# 1. Compute volume contributions (NOW INDEPENDENT OF SURFACE TERMS)
	gamma = 1.4
	[F,G,rho,u,v,p] = EulerFluxes2D(Q, gamma)
	rhsQ=numpy.zeros(Q.shape)
	
	# Compute weak derivatives
	for n in range(4):
		dFdr = glb.Drw.dot(F[:,:,n])
		dFds = glb.Dsw.dot(F[:,:,n])
		dGdr = glb.Drw.dot(G[:,:,n])
		dGds = glb.Dsw.dot(G[:,:,n])
		rhsQ[:,:,n] = (glb.rx*dFdr + glb.sx*dFds) + (glb.ry*dGdr + glb.sy*dGds)
	    
	# 2. Compute surface contributions 
	# 2.1 evaluate '-' and '+' traces of conservative variables
	QM=numpy.zeros([glb.Nfaces*glb.Nfp,glb.K,4])
	QP=numpy.zeros([glb.Nfaces*glb.Nfp,glb.K,4])
	for n in range(4):
		Qn = Q[:,:,n]
		QM[:,:,n] = Qn.flatten()[glb.vmapM].reshape(QM[:,:,n].shape)
		QP[:,:,n] = Qn.flatten()[glb.vmapP].reshape(QP[:,:,n].shape)

	# 2.2 set boundary conditions by modifying positive traces 
	glb.Fx=glb.Fx.reshape([glb.Nfp*glb.Nfaces,glb.K])
	glb.Fy=glb.Fx.reshape([glb.Nfp*glb.Nfaces,glb.K])
	QP = ExactSolutionBC(glb.Fx, glb.Fy, glb.nx, glb.ny, glb.mapI, glb.mapO, glb.mapW, glb.mapC, QP, time)
	
	# 2.3 evaluate primitive variables & flux functions at '-' and '+' traces
	[fM,gM,rhoM,uM,vM,pM] = EulerFluxes2D(QM, gamma)
	[fP,gP,rhoP,uP,vP,pP] = EulerFluxes2D(QP, gamma)
	
	# 2.4 Compute local Lax-Friedrichs/Rusonov numerical fluxes
	lemda=numpy.zeros([glb.Nfaces*glb.Nfp,glb.K])
	m1=((uM**2+vM**2)**0.5 + (abs(gamma*pM/rhoM))**0.5)
	m2=((uP**2+vP**2)**0.5 + (abs(gamma*pP/rhoP))**0.5)
	for i in range(glb.Nfaces*glb.Nfp):
		for j in range(glb.K):
			lemda[i][j]=max(m1[i][j],m2[i][j])
	
	lemda = lemda.reshape([glb.Nfaces,glb.Nfp,glb.K])
	lemMax=lemda.max(1)
	for i in range(glb.Nfaces):
		for j in range(glb.K):
			lemda[i,:,j]=lemMax[i,j]
	lemda = lemda.reshape([glb.Nfaces*glb.Nfp,glb.K])
	
	# 2.5 Lift fluxes
	for n in range(4):
		nflux = glb.nx*(fP[:,:,n] + fM[:,:,n]) + glb.ny*(gP[:,:,n] + gM[:,:,n]) + lemda*(QM[:,:,n] - QP[:,:,n])
		rhsQ[:,:,n] = rhsQ[:,:,n] - glb.LIFT.dot(glb.Fscale*nflux/2)
	return(rhsQ)

def EulerFluxes2D(Q, gamma):
	     
	""" function [F,G,rho,u,v,p] = EulerFluxes2D(Q, gamma)
	Purpose: evaluate primitive variables and Euler flux functions"""
	
	# extract conserved variables
	rho = Q[:,:,0]; rhou = Q[:,:,1]; rhov = Q[:,:,2]; Ener = Q[:,:,3]
	
	# compute primitive variables
	u = rhou/rho; v = rhov/rho; p = (gamma-1)*(Ener - 0.5*(rhou*u + rhov*v))
	# compute flux functions
	F = numpy.zeros(Q.shape)
	F[:,:,0] = rhou; F[:,:,1] = rhou*u + p; F[:,:,2] = rhov*u; F[:,:,3] = u*(Ener+p)
	
	G = numpy.zeros(Q.shape)
	G[:,:,0] = rhov; G[:,:,1] = rhou*v; G[:,:,2] = rhov*v + p; G[:,:,3] = v*(Ener+p)
	return([F,G,rho,u,v,p])

def EulerLF2D(lnx, lny, QM, QP, gamma):
	
	"""EulerLF2D(nx, ny, QM, QP, gamma) = flux
	Purpose: compute Local Lax-Friedrichs/Rusonov fluxes for Euler equations"""
	
	import globalVar2D as glb
	
	# evaluate primitive variables & flux functions at '-' and '+' traces
	[fM,gM,rhoM,uM,vM,pM] = EulerFluxes2D(QM, gamma)
	[fP,gP,rhoP,uP,vP,pP] = EulerFluxes2D(QP, gamma)
	
	# Compute local Lax-Friedrichs/Rusonov numerical fluxes
	lemda=numpy.zeros(QM.shape[0:2])
	m1=((uM**2+vM**2)**0.5 + (abs(gamma*pM/rhoM))**0.5)
	m2=((uP**2+vP**2)**0.5 + (abs(gamma*pP/rhoP))**0.5)
	for i in range(lemda.shape[0]):
		for j in range(lemda.shape[1]):
			lemda[i][j]=max(m1[i][j],m2[i][j])

	lemda = lemda.reshape([glb.Nfaces,lemda.shape[0]/glb.Nfaces,lemda.shape[1]])
	lemMax=lemda.max(1)
	for i in range(glb.Nfaces):
		for j in range(glb.K):
			lemda[i,:,j]=lemMax[i,j]
	lemda = lemda.reshape(QM.shape[0:2])
	
	# Lift fluxes
	flux=numpy.zeros(QP.shape)
	for n in range(4):
		flux[:,:,n] = lnx*(fP[:,:,n] + fM[:,:,n]) + lny*(gP[:,:,n] + gM[:,:,n]) + lemda*(QM[:,:,n] - QP[:,:,n])
	return(flux)

def CurvedEulerRHS2D(Q, time, SolutionBC, fluxtype):
	
	"""function [rhsQ] = CurvedEulerRHS2D(Q, time, SolutionBC, fluxtype)
	Purpose: compute right hand side residual for the compressible Euler 
	gas dynamics equations"""
	
	import globalVar2D as glb
	import gaussFace2D as gss
	import cuba2D as cub
	
	cubState="on"
	gssState="off"
	if cubState=="on":
		# 1.1 Interpolate solution to cubature nodes 
		cQ = numpy.zeros([cub.Ncub, glb.K, 4])
		for n in range(4):
			cQ[:,:,n] = cub.V.dot(Q[:,:,n])
		
		# 1.2 Evaluate flux function at cubature nodes
		gamma = 1.4
		[F,G,rho,u,v,p] = EulerFluxes2D(cQ, gamma)
		
		# 1.3 Compute volume terms (dphidx, F) + (dphidy, G)
		rhsQ = numpy.zeros([glb.Np, glb.K, 4])
		for n in range(4):
			ddr = (cub.Dr.transpose()).dot(cub.W*(cub.rx*F[:,:,n] + cub.ry*G[:,:,n]))
			dds = (cub.Ds.transpose()).dot(cub.W*(cub.sx*F[:,:,n] + cub.sy*G[:,:,n]))
			rhsQ[:,:,n] = ddr + dds
	
	if cubState=="off":
		# 1. Compute volume contributions (NOW INDEPENDENT OF SURFACE TERMS)
		gamma = 1.4
		[F,G,rho,u,v,p] = EulerFluxes2D(Q, gamma)
		rhsQ=numpy.zeros(Q.shape)
		
		# Compute weak derivatives
		for n in range(4):
			dFdr = glb.Drw.dot(F[:,:,n])
			dFds = glb.Dsw.dot(F[:,:,n])
			dGdr = glb.Drw.dot(G[:,:,n])
			dGds = glb.Dsw.dot(G[:,:,n])
			rhsQ[:,:,n] = (glb.rx*dFdr + glb.sx*dFds) + (glb.ry*dGdr + glb.sy*dGds)
			rhsQ[:,:,n] = numpy.linalg.inv(glb.V.dot(glb.V.transpose())).dot(rhsQ[:,:,n])*glb.J
	
	if gssState=="on":
		# 2.1 SURFACE TERMS (using Gauss nodes on element faces)
		nx = gss.nx
		ny = gss.ny
		mapW = gss.mapW
		mapI = gss.mapI
		mapO = gss.mapO
		mapB = gss.mapB
		mapC = gss.mapC
		
		# 2.2 Interpolate solution to Gauss surface nodes
		gQM = numpy.zeros([gss.Ng*glb.Nfaces, glb.K, 4])
		gQP = numpy.zeros([gss.Ng*glb.Nfaces, glb.K, 4])
		for n in range(4):
			gQ = gss.interp.dot(Q[:,:,n])
			gQM[:,:,n] = gQ.flatten()[gss.mapM]
			gQP[:,:,n] = gQ.flatten()[gss.mapP]
		
		# 2.3 Apply boundary conditions to '+' traces
		if(SolutionBC!=[]):
			gQP = SolutionBC(gss.x, gss.y, gss.nx, gss.ny, gss.mapI, gss.mapO, gss.mapW, gss.mapC, gQP, time)
		
		# 2.4 Evaluate surface flux functions with stabilization
		if fluxtype=='LF':
			flux = EulerLF2D (gss.nx, gss.ny, gQM, gQP, gamma)
		if fluxtype=='Roe':
			flux = EulerRoe2D(gss.nx, gss.ny, gQM, gQP, gamma)
		if fluxtype=='HLL':
			flux = EulerHLL2D(gss.nx, gss.ny, gQM, gQP, gamma)
		if fluxtype=='HLLC':
			flux = EulerHLLC2D(gss.nx, gss.ny, gQM, gQP, gamma)	
		
		# 2.5 Compute surface integral terms
		for n in range(4):
			rhsQ[:,:,n] = rhsQ[:,:,n] - gss.interp.transpose().dot(gss.W*flux[:,:,n])
	
	if gssState=="off":
		# 2. Compute surface contributions 
		# 2.1 evaluate '-' and '+' traces of conservative variables
		QM=numpy.zeros([glb.Nfaces*glb.Nfp,glb.K,4])
		QP=numpy.zeros([glb.Nfaces*glb.Nfp,glb.K,4])
		for n in range(4):
			Qn = Q[:,:,n]
			QM[:,:,n] = Qn.flatten()[glb.vmapM].reshape(QM[:,:,n].shape)
			QP[:,:,n] = Qn.flatten()[glb.vmapP].reshape(QP[:,:,n].shape)

		# 2.2 set boundary conditions by modifying positive traces 
		glb.Fx=glb.Fx.reshape([glb.Nfp*glb.Nfaces,glb.K])
		glb.Fy=glb.Fx.reshape([glb.Nfp*glb.Nfaces,glb.K])
		QP = SolutionBC(glb.Fx, glb.Fy, glb.nx, glb.ny, glb.mapI, glb.mapO, glb.mapW, glb.mapC, QP, time)
	
		if fluxtype =='LF':	
			LFflux = EulerLF2D (glb.nx, glb.ny, QM, QP, gamma)	
		if fluxtype == 'Roe':
			LFflux = EulerRoe2D (glb.nx, glb.ny, QM, QP, gamma)
		
		# 2.5 Lift fluxes
		for n in range(4):
			nflux = LFflux[:,:,n] 
			fluxRHS  = glb.LIFT.dot(glb.Fscale*nflux/2)
			fluxRHS = numpy.linalg.inv(glb.V.dot(glb.V.transpose())).dot(fluxRHS)*glb.J
			rhsQ[:,:,n] = rhsQ[:,:,n] - fluxRHS 
	
	# 3.1 Multiply by inverse mass matrix
	for n in range(4):
		# 3.1.a Multiply straight sided elements by inverse mass matrix
		rhsQ[:,glb.straight, n] = glb.V.dot(glb.V.transpose().dot(rhsQ[:,glb.straight, n]/glb.J[:,glb.straight]))
	
		# 3.1.b Multiply curvilinear elements by custom inverse mass matrices
		Ncurved = len(glb.curved)
		for m in range(Ncurved):
			k = curved[m]
			mmCHOL = cub.mmCHOL[:,:,k]
			rhsQ[:,k,n] = numpy.linalg.inv(mmCHOL).dot(numpy.linalg.inv(mmCHOL.transpose()).dot(rhsQ[:,k,n]))
	return(rhsQ)

def EulerRoe2D(nx, ny, QM, QP, gamma):
	  
	"""Compute surface fluxes for Euler's equations using an 
	approximate Riemann solver based on Roe averages"""
	
	Nfields = 4
	
	# Rotate "-" trace momentum to face normal-tangent coordinates
	rhouM = QM[:,:,1]
	rhovM = QM[:,:,2]
	EnerM = QM[:,:,3]
	QM[:,:,1] =  nx*rhouM + ny*rhovM 
	QM[:,:,2] = -ny*rhouM + nx*rhovM;
	
	# Rotate "+" trace momentum to face normal-tangent coordinates
	rhouP = QP[:,:,1]
	rhovP = QP[:,:,2]
	EnerP = QP[:,:,3]
	QP[:,:,1] =  nx*rhouP + ny*rhovP 
	QP[:,:,2] = -ny*rhouP + nx*rhovP
	
	# Compute fluxes and primitive variables in rotated coordinates  
	[fxQM,fyQM,rhoM,uM,vM,pM] = EulerFluxes2D(QM, gamma)
	[fxQP,fyQP,rhoP,uP,vP,pP] = EulerFluxes2D(QP, gamma)
	
	# Compute enthalpy
	HM = (EnerM+pM)/rhoM
	HP = (EnerP+pP)/rhoP
	
	# Compute Roe average variables
	rhoMs = rhoM**0.5
	rhoPs = rhoP**0.5
	
	rho = rhoMs*rhoPs
	u   = (rhoMs*uM + rhoPs*uP)/(rhoMs + rhoPs)
	v   = (rhoMs*vM + rhoPs*vP)/(rhoMs + rhoPs)
	H   = (rhoMs*HM + rhoPs*HP)/(rhoMs + rhoPs)
	
	c2  = (gamma-1)*(H - 0.5*(u**2 + v**2))
	c = c2**0.5
	
	# Riemann fluxes
	dW1 = -0.5*rho*(uP-uM)/c + 0.5*(pP-pM)/c2
	dW2 = (rhoP-rhoM) - (pP-pM)/c2
	dW3 = rho*(vP-vM)
	dW4 = 0.5*rho *(uP-uM)/c + 0.5*(pP-pM)/c2
	
	dW1 = abs(u-c)*dW1
	dW2 = abs(u)*dW2
	dW3 = abs(u)*dW3
	dW4 = abs(u+c)*dW4
	
	# Form Roe fluxes
	fx = (fxQP+fxQM)/2.
	
	fx[:,:,0] =fx[:,:,0]-(dW1*1.       	+dW2*1.            	+dW3*0.	+dW4*1.     	)/2.
	fx[:,:,1] =fx[:,:,1]-(dW1*(u-c)   	+dW2*u            	+dW3*0.	+dW4*(u+c)   	)/2.
	fx[:,:,2] =fx[:,:,2]-(dW1*v       	+dW2*v            	+dW3*1.	+dW4*v       	)/2.
	fx[:,:,3] =fx[:,:,3]-(dW1*(H-u*c)	+dW2*(u**2+v**2)/2.	+dW3*v	+dW4*(H+u*c)	)/2.
	
	# rotate back to Cartesian
	flux = fx
	flux[:,:,1] = nx*fx[:,:,1] - ny*fx[:,:,2]
	flux[:,:,2] = ny*fx[:,:,1] + nx*fx[:,:,2]
	return(flux)

def CurvedEuler2D(Q, FinalTime, ExactSolution, ExactSolutionBC, fluxtype):	
	"""Integrate 2D Euler equations using a 4th order low storage RK"""
	
	import globalVar2D as glb
	import cuba2D as cub
	import gaussFace2D as gss
	# build cubature information
	CubatureOrder = (glb.N+1)*3
	cub.cubatureInitiate(CubatureOrder)
	
	# build Gauss node data
	NGauss = (glb.N+1)*2
	gss.gaussInit(NGauss)
	
	# compute initial timestep
	gamma = 1.4
	dt = eulerDT2D(Q, gamma)     # Adaptive time-stepping
	tstep = 1
	time = 0.
	rhsQ = 0.*Q
	resQ = 0.*Q;
	
	# outer time step loop 
	while (time<FinalTime):
		if(time+dt>FinalTime):
			dt = FinalTime-time
		
		print "time=%f"%time
		
		# 3rd order SSP Runge-Kutta
		rhsQ  = CurvedEulerRHS2D(Q, time, ExactSolutionBC, fluxtype)
		Q1 = Q + dt*rhsQ
		
		rhsQ  = CurvedEulerRHS2D(Q1, time, ExactSolutionBC, fluxtype)
		Q2 = (3.*Q + Q1 + dt*rhsQ)/4.
		
		rhsQ  = CurvedEulerRHS2D(Q2, time, ExactSolutionBC, fluxtype)
		Q = (Q + 2.*Q2 + 2.*dt*rhsQ)/3.
		
		# Increment time and compute new timestep
		time = time+dt
		dt = eulerDT2D(Q, gamma)
		
		tstep = tstep+1
	return(Q)

#### Test function for curved Euler codes  and Roe flux####
def testCurvedEuler(order=9):
	"""Driver script for solving the 2D Euler Isentropic vortex equations
	With higher order integrations"""
	import globalVar2D as glb
	glb.globalInit()
	import mesh2D

	# Order of polynomials used for approximation 
	glb.N = order
	
	# Define flux type
	fluxtype = 'LF'
	
	# Read in Mesh
	filename = 'Grid/Euler2D/vortexA04.neu'
	InitialSolution = isentropicVortexIC2D
	ExactSolution   = isentropicVortexIC2D
	BCSolution      = isentropicVortexBC2D

	# read mesh from file
	[glb.Nv, glb.VX, glb.VY, glb.K, glb.EToV, glb.BCType] = mesh2D.createBC(filename)
	
	# set up nodes and basic operations
	execfile("initiate2D.py")
	
	# turn cylinders into walls
	# There are no curved elements in isentropic vortex case
	ids = numpy.nonzero(glb.BCType==glb.Cyl) 
	glb.BCType[ids] = glb.Wall
	glb.straight=range(glb.K)
	
	functions2D.BuildBCMaps2D()
	
	# compute initial condition
	Q = InitialSolution(glb.x, glb.y, 0.)
	
	# Solve Problem
	FinalTime = 1.0
	Q = CurvedEuler2D(Q, FinalTime, ExactSolution, BCSolution, fluxtype)
	
	# Calculate error
	err=Q-ExactSolution(glb.x,glb.y,FinalTime)
	L2Err=[(numpy.average((err[:,:,i]**2).flatten()))**0.5 for i in range(4)]
	
	return(Q,L2Err)
