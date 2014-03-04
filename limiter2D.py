# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# 2-dimensional limiter function for euler equation, has to be modified for a more general purpose
 
import numpy
import globalVar2d as glb
def limit2D(Q,time,solutionBC,gamma):
	"""Applies 2D limiter on euler equation framework"""
	# Calculate average matrix
	ave = numpy.array([sum(glb.massMatrix[:,i]) for i in range(glb.Np)])

	# Calculate displacements from center
	dropav = numpy.eye(glb.Np)-numpy.ones([glb.Np,1].dot(numpy.array([ave))
	dx = dropav.dot(glb.x)
	dy = dropav.dot(glb.y)

	# Find neighbours
	e1 = glb.EToE[:,0]
	e2 = glb.EToE[:,1]
	e3 = glb.EToE[:,2]

	# Extract vortices and centers
	v1 = glb.EToV[:,0]
	vx1 = glb.VX[v1]
	vy1 = glb.VY[v1]

	v2 = glb.EToV[:,1]
	vx2 = glb.VX[v2]
	vy2 =glb.VY[v2]

	v3 = glb.EToV[:,2]
	vx3 = glb.VX[v3]
	vy3 = glb.VY[v3]

	# Find normals and side lengths
	fnx = numpy.array([vy2-vy1, vy3-vy2, vy1-vy3])
	fny = -numpy.array([vx2-vx1, vx3-vx2, vx1-vx3])
	fL = (fnx**2 + fny**2)**0.5
	fnx = fnx/fL
	fny = fny/fL

	# Find element centers
	xc0 = ave.dot(glb.x)
	xc1 = xc0[e1]
	xc2 = xc0[e2]
	xc3 = xc0[e3]

	yc0 = ave.dot(glb.y)
	yc1 = yc0[e1]
	yc2 = yc0[e2]
	yc3 = yc0[e3]

	# Find triangle related areas
	A0 = ave.dot(glb.J)*2./3
	A1 = A0 + A0[e1]
	A2 = A0 + A0[e2]
	A3 = A0 + A0[e3]

	# Find ids where boundary exists, and there is a need for ghost cells
	id1 = numpy.nonzero(glb.BCType[:,0]!=0)[0]
	id2 = numpy.nonzero(glb.BCType[:,1]!=0)[0]
	id3 = numpy.nonzero(glb.BCType[:,2]!=0)[0]

	# Find ghost element centers
	h1 = 2.*A0[id1]/fL[0,id1]
	xc1[id1] = xc1[id1] + 2*fnx[0:id1]*h1
	yc1[id1] = yc1[id1] + 2*fny[0:id1]*h1

	h2 = 2.*A0[id2]/fL[1,id2]
	xc1[id2] = xc2[id2] + 2*fnx[1:id2]*h2
	yc1[id2] = yc2[id2] + 2*fny[1:id2]*h2

	h3 = 2.*A0[id3]/fL[2,id3]
	xc3[id3] = xc3[id3] + 2*fnx[2:id3]*h3
	yc3[id3] = yc3[id3] + 2*fny[2:id3]*h3

	# Find averages of conserved variables and convert them back to primitive variables
	rho = Q[:,:,0]
	rhou = Q[:,:,1]
	rhov = Q[:,:,2]
	Ener = Q[:,:,3]

	# Find average of conserved variables
	rhoc = ave.dot(rho)
	rhouc = ave.dot(rhou)
	rhovc = ave.dot(rhov)
	Enerc = ave.dot(Ener)

	avrho = numpy.ones([glb.Np,1]).dot(numpy.array([rhoc]))
	avrhou = numpy.ones([glb.Np,1]).dot(numpy.array([rhouc]))
	avrhov = numpy.ones([glb.Np,1]).dot(numpy.array([rhovc]))
	avEner = numpy.ones([glb.Np,1]).dot(numpy.array([Enerc]))

	# Find primitive variables
	pc0 = numpy.zeros([1,glb.K,4])
	pc0[0,:,0] = rhoc
	pc0[0,:,1] = rhouc/rhoc
	pc0[0,:,2] = rhovc/rhoc
	pc0[0,:,3] = (gamma-1)*(Enerc-0.5*(rhouc**2+rhovc**2)/rhoc)

	# Find neighbour conserved  values
	pc = numpy.zeros([glb.Nfaces,glb.K,4])
	pc[:,:,0] = rhoc[glb.EToE.transpose()]
	pc[:,:,1] = rhouc[glb.EToE.transpose()]
	pc[:,:,2] = rhovc[glb.EToE.transpose()]
	pc[:,:,3] = Enerc[glb.EToE.transpose()]

	# Find boundary faces
	idW = numpy.nonzero(glb.BCType.transpose()==glb.Wall)
	idI = numpy.nonzero(glb.BCType.transpose()==glb.In)
	idO = numpy.nonzero(glb.BCType.transpose()==glb.Out)
	idC = numpy.nonzero(glb.BCType.transpose()==glb.Cyl)

	# Creates maps from this
	mapW = idW[0]+idW[1]
	mapI = idI[0]+idI[1]
	mapO = idO[0]+idO[1]
	mapC = idC[0]+idC[1]
	xB = numpy.array([xc1,xc2,xc3])
	yB = numpy.array([yc1,yc2,yc3])

	# Apply boundary conditions
	pc = solutionBC(xB,yB,fnx,fny, mapW,mapI,mapO,mapC, pc,time)
	
	# Find primitive variables
	pc = numpy.zeros([1,glb.K,4])
	pc[:,:,1] = pc[:,:,1]/pc[:,:,0]
	pc[:,:,2] = pc[:,:,2]/pc[:,:,0]
	pc[:,:,3] = (gamma-1)*(pc[:,:,3]-0.5*pc[:,:,0]*(pc[:,:,1]**2+pc[:,:,2]**2))

	# Calculate avg on interelement boundaries
	idBV = [0,Nfp-1,Nfp,2*Nfp-1,3*Nfp-1,2*Nfp]     # Check the order of points
	mapM = glb.mapM.reshape([glb.Nfp*glb.Nfaces,glb.K])[idBV,:]
	mapP = glb.mapP.reshape([glb.Nfp*glb.Nfaces,glb.K])[idBV,:]
	
	rhoA = (rho.flatten()[mapM]+rho.flatten()[mapP])/2.0
	rhouA = (rhou.flatten()[mapM]+rhou.flatten()[mapP])/2.0
	rhovA = (rhov.flatten()[mapM]+rhov.flatten()[mapP])/2.0
	EnerA = (Ener.flatten()[mapM]+Ener.flatten()[mapP])/2.0

	uA = rhouA/rhoA
	vA = rhovA/rhoA
	pA = (gamma-1)*(enerA - 0.5*rhoA*(uA**2 + vA**2))

	pva = numpy.zeros([glb.Nfaces*2,glb.K,4])
	pva[:,:,0] = rhoA
	pva[:,:,1] = uA
	pva[:,:,2] = vA
	pva[:,:,3] = pA
	
	# Find gradient values (Do that for all primitive variables)
	for n in range(4):
		vc0 = pc0[0,:,n]
		vc1 = pc[0,:,n]
		vc2 = pc[1,;,n]
		vc3 = pc[2,:,n]
		va = pva[:,:,n]

		# Compute gradients at three faces
		dvdxe1 = 0.5*((vc1-vc0)*(vy2-vy1) + (va[0,:]-va[1,:])*(yc1-yc0))/A1
		dvdye1 = -0.5*((vc1-vc0)*(vx2-vx1) + (va[0,:]-va[1,:])*(xc1-xc0))/A1
		
		dvdxe2 = 0.5*((vc2-vc0)*(vy3-vy2) + (va[2,:]-va[3,:])*(yc2-yc0))/A2
		dvdye2 = -0.5*((vc2-vc0)*(vx3-vx2) + (va[2,:]-va[3,:])*(xc2-xc0))/A2
		
		dvdxe3 = 0.5*((vc3-vc0)*(vy1-vy3) + (va[4,:]-va[5,:])*(yc3-yc0))/A3
		dvdye3 = -0.5*((vc3-vc0)*(vx1-vx3) + (va[4,:]-va[5,:])*(xc3-xc0))/A3

		# Area averaging
		dvdxc0 = A1*dvdxe1 + A2*dvdxe2 + A3*dvdxe3/(A1+A2+A3)
		dvdyc0 = A1*dvdye1 + A2*dvdye2 + A3*dvdye3/(A1+A2+A3)
		
		dvdxc1 = dvdxc0[e1]
		dvdyc1 = dvdyc0[e1]
		dvdxc2 = dvdxc0[e2]
		dvdyc2 = dvdyc0[e2]
		dvdxc3 = dvdxc0[e3]
		dvdyc3 = dvdyc0[e3]

		# Assign face gradients as center gradients for ghost elements
		dvdxc1[id1]  = dvdxe1[id1]
		dvdyc1[id1]  = dvdye1[id1]
		dvdxc2[id2]  = dvdxe2[id2]
		dvdyc2[id2]  = dvdye2[id2]
		dvdxc3[id3]  = dvdxe3[id3]
		dvdyc3[id3]  = dvdye3[id3]

		# Calculate weights from gradients
		g1 = dvdxc1**2 + dvdyc1**2
		g2 = dvdxc2**2 + dvdyc2**2
		g3 = dvdxc3**2 + dvdyc3**2
		
		tol = 1e-10   # factor to avoid division by zero
		w1  = ((g2*g3)+tol)/(g1**2+g2**2+g3**2+3*tol)
		w2  = ((g1*g3)+tol)/(g1**2+g2**2+g3**2+3*tol)
		w3  = ((g2*g1)+tol)/(g1**2+g2**2+g3**2+3*tol)

		# Apply limiting to the centered gradient
		lvdxc0  = w1*dvdxc1 + w2*dvdxc2 + w3*dvdxc3
		lvdyc0  = w1*dvdyc1 + w2*dvdyc2 + w3*dvdyc3
		
		# calculate cell averages
		dV[:,;,n] = dx*numpy.ones([glb.Np,1]).dot(numpy.array([lvdxc0])) + dy*numpy.ones([glb.Np,1]).dot(numpy.array([lvdyc0]))
		aV[:,:,n] = numpy.ones([glb.Np,1]).dot(numpy.array([vc0]))

	# Get back variables from average and gradient values
	avrho = aV[:,:,0]; avu = aV[:,:,1];  avv = aV[:,;,2];  avp = aV[:,:,3];	
	drho = dV[:,:,0]; du = dV[:,:,1];  dv = dV[:,;,2];  dp = dV[:,:,3];

	# Check whether limiter needs to be applied again
	tol = 1e-02	
	
	limrho = avrho + drho
	ids = numpy.nonzero(limrho.flatten().min()<tol)[0]
	while(len(ids)!=0):
		print "Negative density again! Correcting ..."
		drho.ravel()[ids] = drho.flatten()[ids]/2.0
		drho = drho.reshape(avrho.shape)
		limrho = avrho + drho
		ids = numpy.nonzero(limrho.flatten().min()<tol)[0]
	
	# Calculate final values
	limrhou = avrhou + avrho*du + avu*drho
	limrhov = avrrhov + avrho*dv + avv*drho
	dEner = (1./(gamma-1))*dp + (0.5*drho)*(avu**2 + avv**2) + avrho*(avu*du + avv*dv)
	limEner = avEner + dEner

	# Check if there is negative pressure anywhere
	# If found put energy gradient zero at that point
	limp = (gamma-1)*(limEner -0.5*(limrhou**2 + limrhov**2)/limrho)
	ids = numpy.nonzero(limp.flatten().min()<tol)[0]
	if(len(ids)!=0):
		print "Negative pressure! Correcting ..."
		limEner.ravel[ids] = avEner.flatten()[ids] 
	
	limQ[:,:,0] = limrho; limQ[;,:,1] = limrhou; limQ[:,;,2] = limrhov; limQ[:,:,3] = limEner
	
	return(limQ)

