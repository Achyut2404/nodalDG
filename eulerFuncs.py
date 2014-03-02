# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Functions for solving 1-dimensional Euler problems

import numpy
import math
def EulerRHS1D(rho, rhou ,Ener):
	"""function [rhsrho, rhsrhou, rhsEner] = EulerRHS1D(rho, rhou ,Ener)
	Purpose  : Evaluate RHS flux in 1D Euler"""
	
	import globalVar as glb
	
	# compute maximum velocity for LF flux
	gamma = 1.4
	pres = (gamma-1.0)*(Ener - 0.5*(((rhou)**2.0)/rho))
	cvel = (gamma*pres/rho)**0.5
	lm = abs(rhou/rho)+cvel
	
	# Compute fluxes
	rhof = rhou
	rhouf=(((rhou)**2)/rho)+pres
	Enerf=(Ener+pres)*(rhou/rho)
	
	# Compute jumps at internal faces
	drho  = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K]) 
	drho[:] =  rho.flatten()[glb.vmapM]-  rho.flatten()[glb.vmapP]
	
	drhou = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	drhou[:]  = rhou.flatten()[glb.vmapM]- rhou.flatten()[glb.vmapP]
	
	dEner = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	dEner[:]  = Ener.flatten()[glb.vmapM]- Ener.flatten()[glb.vmapP]
	
	drhof = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	drhof[:]  = rhof.flatten()[glb.vmapM]- rhof.flatten()[glb.vmapP]
	
	drhouf = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	drhouf[:]  = rhouf.flatten()[glb.vmapM]- rhouf.flatten()[glb.vmapP]
	
	dEnerf = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	dEnerf[:]  = Enerf.flatten()[glb.vmapM]- Enerf.flatten()[glb.vmapP]
	
	LFc = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K])
	LFc[:] = numpy.maximum(lm.flatten()[glb.vmapP],lm.flatten()[glb.vmapM])
	
	# Compute fluxes at interfaces
	drhof[:] = glb.nx[:]*drhof[:]/2.0-LFc[:]/2.0*drho[:]
	drhouf[:] = glb.nx[:]*drhouf[:]/2.0-LFc[:]/2.0*drhou[:] 
	dEnerf[:]=glb.nx[:]*dEnerf[:]/2.0-LFc[:]/2.0*dEner[:] 

	# Boundary conditions for Sod's problem
	rhoin    = 1.000;   rhouin   = 0.0;
	pin      = 1.000;   Enerin   = pin/(gamma-1.0);
	rhoout   = 0.125;   rhouout  = 0.0;
	pout     = 0.100;   Enerout  = pout/(gamma-1.0);
	
	# Set fluxes at inflow/outflow
	rhofin = rhouin
	rhoufin = (rhouin**2/rhoin)+pin
	Enerfin = (pin/(gamma-1.0)+(0.5*(rhouin**2)/(rhoin+pin)))*rhouin/rhoin
	lmI=lm.flatten()[glb.vmapI]/2.0
	nxI=glb.nx.flatten()[glb.mapI]
	drhof.ravel()[glb.mapI]=nxI*(rhof.flatten()[glb.vmapI]-rhofin)/2.0-lmI*(rho.flatten()[glb.vmapI] -rhoin) 
	drhouf.ravel()[glb.mapI]=nxI*(rhouf.flatten()[glb.vmapI]-rhoufin)/2.0-lmI*(rhou.flatten()[glb.vmapI]-rhouin)
	dEnerf.ravel()[glb.mapI]=nxI*(Enerf.flatten()[glb.vmapI]-Enerfin)/2.0-lmI*(Ener.flatten()[glb.vmapI]-Enerin)
	
	rhofout = rhouout
	rhoufout = (rhouout**2/rhoout)+pout
	Enerfout = (pout/(gamma-1.0)+(0.5*(rhouout**2)/(rhoout+pout)))*rhouout/rhoout
	lmO = lm.flatten()[glb.vmapO]/2.0
	nxO=glb.nx.flatten()[glb.mapO]
	drhof.ravel() [glb.mapO]=nxO*(rhof.flatten()[glb.vmapO] - rhofout)/2.0-lmO*(rho.flatten() [glb.vmapO]- rhoout) 
	drhouf.ravel()[glb.mapO]=nxO*(rhouf.flatten()[glb.vmapO]-rhoufout)/2.0-lmO*(rhou.flatten()[glb.vmapO]-rhouout)
	dEnerf.ravel()[glb.mapO]=nxO*(Enerf.flatten()[glb.vmapO]-Enerfout)/2.0-lmO*(Ener.flatten()[glb.vmapO]-Enerout)
	
	# compute right hand sides of the PDE's
	rhsrho  = -glb.rx*(glb.Dr.dot(rhof))  + glb.LIFT.dot(glb.Fscale*drhof)
	rhsrhou = -glb.rx*(glb.Dr.dot(rhouf)) + glb.LIFT.dot(glb.Fscale*drhouf)
	rhsEner = -glb.rx*(glb.Dr.dot(Enerf)) + glb.LIFT.dot(glb.Fscale*dEnerf)

	return([rhsrho, rhsrhou, rhsEner])

def Euler1D(rho, rhou, Ener, FinalTime):
	"""function [rho, rhou, Ener] = Euler1D(rho, rhou, Ener, FinalTime)
	Purpose  : Integrate 1D Euler equations until FinalTime starting with
	           initial conditions [rho, rhou, Ener]"""
	
	import globalVar as glb
	import slopeLimit as slp
	
	# Parameters
	gamma = 1.4; CFL = 1.0; time = 0.0;
	
	# Prepare for adaptive time stepping
	mindx = min(glb.x[1,:]-glb.x[0,:])

	# Limit initial solution
	rho =slp.SlopeLimitPoly(rho); rhou=slp.SlopeLimitPoly(rhou); Ener=slp.SlopeLimitPoly(Ener);
	
	# outer time step loop 
	while(time<FinalTime):
		print time
		Temp = (Ener - ((0.5*(rhou)**2)/rho))/rho
		cvel = (gamma*(gamma-1)*Temp)**0.5
		dt = CFL*min((mindx/(abs(rhou/rho)+cvel)).flatten())/1.0
		if(time+dt>FinalTime):
			dt = FinalTime-time
		
		# 3rd order SSP Runge-Kutta
		
		# SSP RK Stage 1.
		[rhsrho,rhsrhou,rhsEner]  = EulerRHS1D(rho, rhou, Ener)
		rho1  = rho  + dt*rhsrho
		rhou1 = rhou + dt*rhsrhou
		Ener1 = Ener + dt*rhsEner

		# Limit fields 
		rho1  = slp.SlopeLimitPoly(rho1); rhou1 = slp.SlopeLimitPoly(rhou1); Ener1 = slp.SlopeLimitPoly(Ener1);
		
		# SSP RK Stage 2.
		[rhsrho,rhsrhou,rhsEner]  = EulerRHS1D(rho1, rhou1, Ener1);
		rho2   = (3.*rho  + rho1  + dt*rhsrho )/4.
		rhou2  = (3.*rhou + rhou1 + dt*rhsrhou)/4.
		Ener2  = (3.*Ener + Ener1 + dt*rhsEner)/4.
		
		# Limit fields
		rho2  = slp.SlopeLimitPoly(rho2); rhou2 = slp.SlopeLimitPoly(rhou2); Ener2 = slp.SlopeLimitPoly(Ener2);
		
		# SSP RK Stage 3.
		[rhsrho,rhsrhou,rhsEner]  = EulerRHS1D(rho2, rhou2, Ener2);
		rho  = (rho  + 2.*rho2  + 2.*dt*rhsrho )/3.;
		rhou = (rhou + 2.*rhou2 + 2.*dt*rhsrhou)/3.;
		Ener = (Ener + 2.*Ener2 + 2.*dt*rhsEner)/3.;
		
		# Limit solution
		rho = slp.SlopeLimitPoly(rho); rhou=slp.SlopeLimitPoly(rhou); Ener=slp.SlopeLimitPoly(Ener);
		
		# Increment time and adapt timestep
		time = time+dt
	return([rho,rhou,Ener])

def exactSod(x,t=0.2,x0=0.5,ql=[1.,1.,0.],qr=[0.125,0.1,0.],gamma=1.4):
	""" Gives exact solution to Sod shock tube problem in order to check for accuracies and compare with computational solution.
	Exact solution is given at x points.
	Algorith taken from http://www.phys.lsu.edu/~tohline/PHYS7412/sod.html
	Ref. Sod, G. A. 1978, Journal of Computational Physics, 27, 1-31. """
	
	#Import stuff
	from sympy.solvers import nsolve
	from sympy import Symbol
	import matplotlib.pyplot as plt
	
	#Initiate stuff
	
	shape=x.shape
	x=x.flatten()
	p1 = Symbol('p1')
	[rol,pl,vl]=ql
	[ror,pr,vr]=qr
	
	#Calculate wave velocities and values
	
	cleft=(gamma*pl/rol)**(0.5)
	cright=(gamma*pr/ror)**(0.5)
	m=((gamma-1)/(gamma+1))**0.5
	eq=((2*(gamma**0.5))/(gamma-1)*(1-(p1**((gamma-1)/2/gamma))))-((p1-pr)*(((1-(m**2))**2)*((ror*(p1+(m*m*pr)))**(-1)))**(0.5))
	ppost=float(nsolve(eq,p1,0.))
	rpostByrright=((ppost/pr)+(m*m))/(1+((ppost/pr)*(m*m)))
	vpost=(2*(gamma**0.5))/(gamma-1)*(1-(ppost**((gamma-1)/2/gamma)))
	romid=((ppost/pl)**(1/gamma))*rol
	vshock=vpost*(rpostByrright)/(rpostByrright-1)
	ropost=rpostByrright*ror
	pmid=ppost
	vmid=vpost

	#Calculate locations
	x1=x0-(cleft*t)
	x3=x0+(vpost*t)
	x4=x0+(vshock*t)
	ro=[]
	p=[]
	v=[]
	for i in x:
		csound=((m*m)*(x0-i)/t)+((1-(m*m))*cleft)
		vinst=(1-(m*m))*(((i-x0)/t)+cleft)
		roinst=rol*((csound/cleft)**(2/(gamma-1)))
		pinst=pl*((roinst/rol)**(gamma))
		if i<x1:
			ro.append(rol)
			p.append(pl)
			v.append(vl)
		elif (i>=x4):
			ro.append(ror)
			p.append(pr)
			v.append(vr)
		elif (i<x4) and (i>=x3):
			ro.append(ropost)
			p.append(ppost)
			v.append(vpost)
		elif (i<x3) and (((roinst>rol) and (roinst<romid)) or ((roinst<rol) and (roinst>romid))):
			ro.append(roinst)
			p.append(pinst)
			v.append(vinst)
		else:
			ro.append(romid)
			p.append(pmid)
			v.append(vmid)
			
	#Reshape solutions
	ro=numpy.array(ro).reshape(shape)
	v=numpy.array(v).reshape(shape)
	p=numpy.array(p).reshape(shape)
	
	#calculate conserved variables
	rou = ro*v
	ener=p/(gamma-1)+(0.5*ro*v*v)

	return([ro,v,p,rou,ener])
