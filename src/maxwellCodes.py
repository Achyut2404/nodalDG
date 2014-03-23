# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Functions for solving 2 - dimensional maxwell equations (Linear Problem) using nodal DG

import numpy
import math
import globalVar2D as glb
import functions2D
import functions
import matplotlib.pyplot as plt

def MaxwellRHS2D(Hx,Hy,Ez):	
	""" function [rhsHx, rhsHy, rhsEz] = MaxwellRHS2D(Hx,Hy,Ez)
	 Purpose  : Evaluate RHS flux in 2D Maxwell TM form """
	
	import globalVar2D as glb
	
	# Define field differences at faces
	dHx = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K]); dHx.ravel()[:] = Hx.flatten()[glb.vmapM]-Hx.flatten()[glb.vmapP];
	dHy = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K]); dHy.ravel()[:] = Hy.flatten()[glb.vmapM]-Hy.flatten()[glb.vmapP]; 
	dEz = numpy.zeros([glb.Nfp*glb.Nfaces,glb.K]); dEz.ravel()[:] = Ez.flatten()[glb.vmapM]-Ez.flatten()[glb.vmapP];
	
	# Impose reflective boundary conditions (Ez+ = -Ez-)
	dHx.ravel()[glb.mapB] = 0; dHy.ravel()[glb.mapB] = 0; dEz.ravel()[glb.mapB] = 2*Ez.ravel()[glb.vmapB];
	
	# evaluate upwind fluxes
	alpha = 1.0; 
	ndotdH =  glb.nx*dHx+glb.ny*dHy;
	fluxHx =  glb.ny*dEz + alpha*(ndotdH*glb.nx-dHx);
	fluxHy = -glb.nx*dEz + alpha*(ndotdH*glb.ny-dHy);
	fluxEz = -glb.nx*dHy + glb.ny*dHx - alpha*dEz;
	
	# local derivatives of fields
	[Ezx,Ezy] = functions2D.Grad2D(Ez); [CuHx,CuHy,CuHz] = functions2D.Curl2D(Hx,Hy);
	
	# compute right hand sides of the PDE's
	rhsHx = -Ezy  + glb.LIFT.dot(glb.Fscale*fluxHx)/2.0;
	rhsHy =  Ezx  + glb.LIFT.dot(glb.Fscale*fluxHy)/2.0;
	rhsEz =  CuHz + glb.LIFT.dot(glb.Fscale*fluxEz)/2.0;
	return([rhsHx, rhsHy, rhsEz])

def Maxwell2D(Hx, Hy, Ez, FinalTime):
	""" function [Hx,Hy,Ez] = Maxwell2D(Hx, Hy, Ez, FinalTime)
	 Purpose :Integrate TM-mode Maxwell's until FinalTime starting with initial conditions Hx,Hy,Ez"""
	
	import globalVar2D as glb
	time = 0.0
	errors=[]
	
	# Runge-Kutta residual storage  
	resHx = numpy.zeros([glb.Np,glb.K]); resHy = numpy.zeros([glb.Np,glb.K]); resEz = numpy.zeros([glb.Np,glb.K]); 
	
	# compute time step size
	rLGL = functions.JacobiGQ(0,0,glb.N); rmin = abs(rLGL[0]-rLGL[1]).min();
	dtscale = functions2D.dtscale2D(); dt = (dtscale.min())*rmin*2/3
	# outer time step loop 
	while (time<FinalTime):
		print "time=%f"%time
		if(time+dt>FinalTime):
		  dt = FinalTime-time
		
		for INTRK in range(5):
		  # compute right hand side of TM-mode Maxwell's equations
		  [rhsHx, rhsHy, rhsEz] = MaxwellRHS2D(Hx,Hy,Ez);
		
		  # initiate and increment Runge-Kutta residuals
		  resHx = glb.rk4a[INTRK]*resHx + dt*rhsHx;  
		  resHy = glb.rk4a[INTRK]*resHy + dt*rhsHy; 
		  resEz = glb.rk4a[INTRK]*resEz + dt*rhsEz; 
			
		  # update fields
		  Hx = Hx+glb.rk4b[INTRK]*resHx; Hy = Hy+glb.rk4b[INTRK]*resHy; Ez = Ez+glb.rk4b[INTRK]*resEz;        
		# Increment time
		time = time+dt
		
		mmode=1.0
		nmode=1.0
		omega=math.pi*(2)**0.5
		EzExac = numpy.vectorize(math.sin)(mmode*math.pi*glb.x)*numpy.vectorize(math.sin)(nmode*math.pi*glb.y)*math.cos(omega*time)
		HxExac=(-math.pi*nmode/omega)*numpy.vectorize(math.sin)(mmode*math.pi*glb.x)*numpy.vectorize(math.cos)(nmode*math.pi*glb.y)*math.sin(omega*time)
		HyExac=(math.pi*mmode/omega)*numpy.vectorize(math.cos)(mmode*math.pi*glb.x)*numpy.vectorize(math.sin)(nmode*math.pi*glb.y)*math.sin(omega*time)
		errEz=(numpy.average((Ez-EzExac).flatten()**2))**0.5
		errHx=(numpy.average((Hx-HxExac).flatten()**2))**0.5
		errHy=(numpy.average((Hy-HyExac).flatten()**2))**0.5
		errors.append(errEz)
		
	return([Hx,Hy,Ez,time,errors])
