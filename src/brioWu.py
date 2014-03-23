# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

#Functions for solving 1D MHD shock tube problem

import numpy
import math

def mhdFlux(q,gamma,bx):
	"""Calculates flux for 1D Magneto-hydrodynamic problem
	here q is of shape [Np,Ne,7]"""
	
	#Initiate flux
	flux=numpy.zeros(q.shape)

	#Get fields
	ro=q[:,:,0]; 	roux=q[:,:,1]; 	rouy=q[:,:,2];	rouz=q[:,:,3]
	by=q[:,:,4]; 	bz=q[:,:,5]; 	Ener=q[:,:,6]
	
	#Calculate pressure and pressure*
	p=(gamma-1)*(Ener-(0.5*roux*roux/ro)-(0.5*rouy*rouy/ro)-(0.5*rouz*rouz/ro)-(0.5*bx*bx)-(0.5*by*by)-(0.5*bz*bz))
	pstar=p+(0.5*bx*bx)+(0.5*by*by)+(0.5*bz*bz)
	
	#Calculate fluxes (doesnt that sound like Xerxes!!)
	flux[:,:,0]=roux
	flux[:,:,1]=(roux*roux/ro)+pstar-(bx*bx)
	flux[:,:,2]=(roux*rouy/ro)-(bx*by)
	flux[:,:,3]=(roux*rouz/ro)-(bx*bz)
	flux[:,:,4]=(by*roux/ro)-(bx*rouy/ro)
	flux[:,:,5]=(bz*roux/ro)-(bx*rouz/ro)
	flux[:,:,6]=((Ener+pstar)*roux/ro)-(bx*((bx*roux/ro)+(by*rouy/ro)+(bz*rouz/ro)))
	return(flux)

def maxEigMHD(q,gamma,bx):
	"""Calculate maximum eigen value at each point for 1D MHD case.
	That will be |v|+cf. Cf is fast MHD Wave speed
	Note that q is of shape [Np,Ne,7]"""
	#Initiate flux
	maxEig=numpy.zeros(q.shape[0:2])
	
	#Get fields
	ro=q[:,:,0]; 	roux=q[:,:,1]; 	rouy=q[:,:,2];	rouz=q[:,:,3]
	by=q[:,:,4]; 	bz=q[:,:,5]; 	Ener=q[:,:,6]
	
	#Calculate pressure and pressure*
	p=(gamma-1)*(Ener-(0.5*roux*roux/ro)-(0.5*rouy*rouy/ro)-(0.5*rouz*rouz/ro)-(0.5*bx*bx)-(0.5*by*by)-(0.5*bz*bz))

	#Calculate Cf
	cf1=((gamma*p)+(bx**2+by**2+bz**2))/ro
	cf2=((cf1**2)-(4*gamma*p/ro*bx*bx/ro))**0.5
	cf=0.5*(cf1+cf2)
	
	#calculate maxEig
	c=((roux/ro)**2+(rouy/ro)**2+(rouz/ro)**2)**0.5
	maxEig=abs(c)+cf
	return(maxEig)
	
def RHS1D(q,gamma,bx):
	"""Purpose  : Evaluate RHS flux in 1D Magnetohydronymic"""
	
	import globalVar as glb
	
	# compute maximum velocity for LF flux
	lm = maxEigMHD(q,gamma,bx)
	
	# Compute fluxes
	qf = mhdFlux(q,gamma,bx)

	# Compute jumps at internal faces
	dq = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K,7])
	dqf = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K,7])
	
	for n in range(7):
		dq[:,:,n] = (q[:,:,n].flatten()[glb.vmapM]-q[:,:,n].flatten()[glb.vmapP]).reshape([glb.Nfp*glb.Nfaces,glb.K])

	for n in range(7):
		dqf[:,:,n] = (qf[:,:,n].flatten()[glb.vmapM]-qf[:,:,n].flatten()[glb.vmapP]).reshape([glb.Nfp*glb.Nfaces,glb.K])

	LFc = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	LFc[:] = numpy.maximum(lm.flatten()[glb.vmapP],lm.flatten()[glb.vmapM])
	
	# Compute fluxes at interfaces
	for n in range(7):
		dqf[:,:,n]=glb.nx*dqf[:,:,n]/2.0-LFc/2.0*dq[:,:,n]

	# Boundary conditions for Brio Wu Problem problem
	pin=1.0;		Enerin=(pin/(gamma-1))+(0.5*(1+bx**2))
	pout=0.0;		Enerout=(pout/(gamma-1))+(0.5*(1+bx**2))
	qin = numpy.array([1.,0.,0.,0.,1.,0.,Enerin])
	qout = numpy.array([0.125,0.,0.,0.,-1.,0.,Enerout])
	
	# Set fluxes at inflow/outflow
	qfin=mhdFlux(numpy.array([[qin]]),gamma,bx)[0][0]
	qfout=mhdFlux(numpy.array([[qout]]),gamma,bx)[0][0]
	lmI=lm.flatten()[glb.vmapI]/2.0
	nxI=glb.nx.flatten()[glb.mapI]
	lmO = lm.flatten()[glb.vmapO]/2.0
	nxO=glb.nx.flatten()[glb.mapO]

	for n in range(7):
		dqf[:,:,n].ravel()[glb.mapI]=nxI*(qf[:,:,n].flatten()[glb.vmapI]-qfin[n])/2.0-lmI*(q[:,:,n].flatten()[glb.vmapI] -qin[n]) 
		dqf[:,:,n].ravel() [glb.mapO]=nxO*(qf[:,:,n].flatten()[glb.vmapO] - qfout[n])/2.0-lmO*(q[:,:,n].flatten() [glb.vmapO]- qout[n]) 

	# compute right hand sides of the PDE's
	rhsq=numpy.zeros(q.shape)
	for n in range(7):
		rhsq[:,:,n] = -glb.rx*(glb.Dr.dot(qf[:,:,n]))  + glb.LIFT.dot(glb.Fscale*dqf[:,:,n])
	return(rhsq)

def mhd1D(q, FinalTime,gamma,bx):
	"""Purpose  : Integrate 1D MHD equations until FinalTime starting with
	           initial conditions q"""
	
	import globalVar as glb
	import slopeLimit as slp
	
	# Parameters
	CFL = 1.0; time = 0.0;
	
	# Prepare for adaptive time stepping
	mindx = min(glb.x[1,:]-glb.x[0,:])

	# Limit initial solution
	for n in range(7):
		q[:,:,n]=slp.SlopeLimitPoly(q[:,:,n])
	
	# outer time step loop 
	while(time<FinalTime):
		print time
		eigmax=maxEigMHD(q,gamma,bx)
		dt = CFL*min((mindx/eigmax).flatten())
		if(time+dt>FinalTime):
			dt = FinalTime-time
		
		# 3rd order SSP Runge-Kutta
		
		# SSP RK Stage 1.
		rhsq  = RHS1D(q,gamma,bx)
		q1  = q  + dt*rhsq
		
		# Limit fields 
		for n in range(7):
			q1[:,:,n]=slp.SlopeLimitPoly(q1[:,:,n])
		
		# SSP RK Stage 2.
		rhsq  = RHS1D(q1,gamma,bx)
		q2   = (3.*q  + q1  + dt*rhsq )/4.
		
		# Limit fields 
		for n in range(7):
			q2[:,:,n]=slp.SlopeLimitPoly(q2[:,:,n])
		
		# SSP RK Stage 3.
		rhsq  = RHS1D(q2,gamma,bx)
		q  = (q  + 2.*q2  + 2.*dt*rhsq )/3.
		
		# Limit fields 
		for n in range(7):
			q[:,:,n]=slp.SlopeLimitPoly(q[:,:,n])
		
		# Increment time and adapt timestep
		time = time+dt
	return(q)
def testBrioWu(Npoly=8,Npoint=250):
	"""Run 1D MHD test case BrioWu
		Driver script for solving the 1D Euler equations
	Refer to www.csnu.edu/~jb715473/examples/mhd1d.htm"""
	import globalVar as glb
	import numpy
	import functions
	import matplotlib.pyplot as plt
	
	# Polynomial order used for approximation 
	glb.N = Npoly
	
	# Generate simple mesh
	[glb.Nv, glb.VX, glb.K, EToV] = functions.MeshGen1D(0.0, 1.0, Npoint)
	
	# Initialize solver and construct grid and metric
	execfile("initiate.py")
	gamma = 2.0;		bx=0.75
	
	# Set up initial conditions -- Brio Wu's problem
	p1=1.0;		Ener1=(p1/(gamma-1))+(0.5*(1+bx**2))
	p2=0.1;		Ener2=(p2/(gamma-1))+(0.5*(1+bx**2))
	q1 = numpy.array([1.,0.,0.,0.,1.,0.,Ener1])
	q2 = numpy.array([0.125,0.,0.,0.,-1.,0.,Ener2])
	
	q=numpy.zeros([glb.Np,glb.K,7])
	for n in range(7):
		q[:,:,n] = q1[n]*(((-1)*numpy.vectorize(functions.step)(glb.x,0.5))+1)+(q2[n]*numpy.vectorize(functions.step)(glb.x,0.5))
	FinalTime = 0.12
	
	# Solve Problem
	q = mhd1D(q,FinalTime,gamma,bx)
	return(q)
