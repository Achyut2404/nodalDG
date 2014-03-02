# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Slope Limiter functions for 1-Dimensional case

import sys
import globalVar as glb
import numpy

def sign(x):
	if x==0:
		return(0.0)
	else:
		return(x/abs(x))
def minmod(v):
	v=numpy.array(v)
	s=(sum(numpy.vectorize(sign)(v)))/len(v)
	if abs(s)==1:
		return(s*min(abs(v)))
	else:
		return(0)

def SlopeLimitLin(f):
	"""Linear slope Limiter
	(f is function to be slope Limited)
	(x is grid points)
	Np and Ne being number of points in an element and number of elements
	R is the -(differentiation matrix)"""
	
	# Compute modal coefficients
	fModal = glb.invV.dot(f)
	
	# Extract linear polynomial
	fLinear = numpy.copy(fModal)
	fLinear[2:glb.Np,:] = 0
	fLinear = glb.V.dot(fLinear)
	
	# Extract cell averages
	fModal[1:glb.Np,:]=0
	uavg = glb.V.dot(fModal)
	vavg = uavg[0]

	xavg=[numpy.average(glb.x.transpose()[i]) for i in range(len(glb.x.transpose()))]

	vdiff1=glb.rx*(glb.Dr.dot(fLinear))
	vdiff1=vdiff1[0]
	
	vdiff2=[(vavg[i+1]-vavg[i])*2.0/(glb.x.transpose()[i][-1]-glb.x.transpose()[i][0]) for i in range(len(vavg)-1)]
	vdiff2=numpy.concatenate([vdiff2,numpy.array([numpy.average(vdiff1.transpose()[-1])])])	
	vdiff3=[(vavg[i]-vavg[i-1])*2.0/(glb.x.transpose()[i][-1]-glb.x.transpose()[i][0]) for i in range(1,len(vavg))]
	vdiff3=numpy.concatenate([numpy.array([numpy.average(vdiff1.transpose()[0])]),vdiff3])
	
	minmodv=numpy.copy(vdiff1)

	for i in range(len(minmodv)):
		minmodv[i]=minmod([vdiff1[i],vdiff2[i],vdiff3[i]])

	minmodv=numpy.ones([glb.Np,1]).dot(numpy.array([minmodv]))
	vavg=numpy.ones([glb.Np,1]).dot(numpy.array([vavg]))
	xavg=numpy.ones([glb.Np,1]).dot(numpy.array([xavg]))

	limitf=vavg+((glb.x-xavg)*minmodv)
	return(limitf)

def SlopeLimitPoly(f):
	"""Polynomial slope limiter"""
	
	eps=1e-8
	
	# Compute linear polynomial
	fModal = glb.invV.dot(f)
	fLinear=numpy.copy(fModal)
	fLinear[2:glb.Np,:]=0.0
	fLinear = glb.V.dot(fLinear)
	
	# Compute cell averages
	fModal[1:glb.Np,:]=0.0
	uavg = glb.V.dot(fModal)
	vavg = uavg[0]
	
	# Apply slope limiter as needed.
	limitf=f.copy()
	
	# find end values of each element
	ue1 = f[0]
	ue2 = f[-1]
	
	# find cell averages
	vk = vavg.copy()
	vkm1 = numpy.concatenate([numpy.array([vavg[0]]),vavg[0:glb.K-1]])
	vkp1 = numpy.concatenate([vavg[1:glb.K],numpy.array([vavg[glb.K-1]])]) 

	# Apply reconstruction to find elements in need of limiting
	ve1=numpy.copy(vk)
	ve2=numpy.copy(vk)
	for i in range(len(vk)):
		ve1[i] = vk[i] - minmod([vk[i]-ue1[i],vk[i]-vkm1[i],vkp1[i]-vk[i]])
		ve2[i] = vk[i] - minmod([ue2[i]-vk[i],vk[i]-vkm1[i],vkp1[i]-vk[i]])

	# Find elements that needs limiting
	error=abs(ve1-ue1)*abs(ve2-ue2)
	toLimit=numpy.nonzero(error>eps)[0]

	xavg=[numpy.average(glb.x.transpose()[i]) for i in range(len(glb.x.transpose()))]
	
	vdiff1=glb.rx*(glb.Dr.dot(fLinear))
	vdiff1=vdiff1[0]
	
	# Distance between elements
	h=glb.x[-1]-glb.x[0]
	
	vdiff2=(vk-vkm1)/h*2.0
	vdiff3=(vkp1-vk)/h*2.0
	
	minmodv=numpy.zeros(len(vdiff1))

	for i in toLimit:
		minmodv=minmod([vdiff1[i],vdiff2[i],vdiff3[i]])
		limitf.transpose()[i]=vavg[i]*numpy.ones(glb.Np)+(((glb.x.transpose()[i])-(xavg[i]*numpy.ones(glb.Np)))*minmodv)
	
	return(limitf)

def minmodB(v,h,M=20):
	vdash=[]
	for i in range(len(v)):
		if i==0:
			vdash.append(v[i])
		else:
			vdash.append(v[i]+(M*h*h*v[i]/abs(v[i])))
	return(minmod(array(vdash)))
