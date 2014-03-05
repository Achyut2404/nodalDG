# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Functions for higher order surface intergration using gaussian qudrature
# There are some bugs in this script, need to work on them

import numpy
import functions as func1
import functions2D as func2
import matplotlib.pyplot as plt
import globalVar2D as glb

def gaussInit(NGauss):

	""" Purpose:  Compute Gauss nodes for face term integration, and interpolation matrices 
	This will have its own Fscale, LIFT,nx,ny,Fx,Fy,mapM,mapP,boundary maps,interpolation matrice ..."""
	
	global Ng
	Ng=NGauss
	
	global z,w
	global face1r,face1s,face2r,face2s,face3r,face3s
	[z, w] = func1.JacobiGQ(0., 0., NGauss-1)
	face1r =  numpy.array([z]).transpose()
	face2r = -numpy.array([z]).transpose()
	face3r = -numpy.ones([NGauss, 1])
	face1s = -numpy.ones([NGauss, 1])
	face2s =  numpy.array([z]).transpose()
	face3s = -numpy.array([z]).transpose()

	#Build face interpolation matrix        (Interpolate field to NGauss order points)
	global finterp
	finterp = numpy.zeros([NGauss,glb.Np,glb.Nfaces])
	V1 = func2.Vandermonde2D(glb.N, face1r[:,0], face1s[:,0]);   finterp[:,:,0] = V1.dot(glb.invV)
	V2 = func2.Vandermonde2D(glb.N, face2r[:,0], face2s[:,0]);   finterp[:,:,1] = V2.dot(glb.invV)
	V3 = func2.Vandermonde2D(glb.N, face3r[:,0], face3s[:,0]);   finterp[:,:,2] = V3.dot(glb.invV)

	#Initiation for building maps for gaussian points
	global interp,mapM,mapP
	global mapI,mapO,mapW,mapB,mapS,mapD,mapN,mapC
	interp = numpy.array([finterp[:,:,0], finterp[:,:,1], finterp[:,:,2]])
	
	global x, y
	x = interp.dot(glb.x)
	y = interp.dot(glb.y)
	
	mapM = numpy.arange(NGauss*glb.Nfaces*glb.K).reshape([NGauss*glb.Nfaces, glb.K]).astype(int)	    
	mapP = numpy.arange(NGauss*glb.Nfaces*glb.K).reshape([NGauss*glb.Nfaces, glb.K]).astype(int) 
	
	mapI = numpy.array([]).astype(int)
	mapO = numpy.array([]).astype(int)
	mapW = numpy.array([]).astype(int)
	mapB = numpy.array([]).astype(int)
	mapS = numpy.array([]).astype(int)
	mapD = numpy.array([]).astype(int)
	mapN = numpy.array([]).astype(int)
	mapC = numpy.array([]).astype(int)

	zer = numpy.zeros([NGauss*glb.Nfaces,glb.K])
	one = numpy.ones([1, NGauss])

	#Initiation for computation of normals and jacobians element and face wise
	global nx,ny,rx,ry,sx,sy,J,sJ
	nx = zer.copy()
	ny = zer.copy()
	rx = zer.copy()
	ry = zer.copy()
	sx = zer.copy()
	sy = zer.copy()
	J = zer.copy()
	sJ = zer.copy()
	for f1 in range(glb.Nfaces):
		VM = finterp[:,:,f1]
		dVMdr = VM.dot(glb.Dr)
		dVMds = VM.dot(glb.Ds)
		ids1 = range((f1)*NGauss,(f1+1)*NGauss)
		for k1 in range(glb.K):
			# calculate geometric factors at Gauss points
			[grx,gsx,gry,gsy,gJ] = func2.GeometricFactors2D(glb.x[:,k1],glb.y[:,k1],dVMdr,dVMds)
			# compute normals at Gauss points  (Take reference to functions2D.Normals2D())
			if f1==0:
				gnx = -gsx
				gny = -gsy
			if f1==1:
				gnx =  grx+gsx
				gny =  gry+gsy
			if f1==2:
				gnx = -grx
				gny = -gry
			
			gsJ = (gnx*gnx+gny*gny)**0.5
			gnx = gnx/gsJ
			gny = gny/gsJ
			gsJ = gsJ*gJ
			
			nx[ids1,k1] = gnx; 		ny[ids1,k1] = gny; 		sJ[ids1,k1] = gsJ
			rx[ids1,k1] = grx; 		ry[ids1,k1] = gry; 		J [ids1,k1] = gJ
			sx[ids1,k1] = gsx;		sy[ids1,k1] = gsy
			
			#Build maps (Take reference to functions2D.buildMaps2D())
			k2 = glb.EToE[k1,f1]
			f2 = glb.EToF[k1,f1]
			
			# reference length of edge
			v1 = glb.EToV[k1,f1]
			v2 = glb.EToV[k1, (f1+1)%3]
			refd = ( (glb.VX[v1]-glb.VX[v2])**2 + (glb.VY[v1]-glb.VY[v2])**2 )**0.5
			
			# find find volume node numbers of left and right nodes 
			ids2 = range((f2)*NGauss,(f2+1)*NGauss)
			
			x1 = x.flatten()[mapM[ids1,k1]]; y1 = y.flatten()[mapM[ids1,k1]]
			x2 = x.flatten()[mapP[ids2,k2]]; y2 = y.flatten()[mapP[ids2,k2]]
			x1 = numpy.array([x1]).transpose().dot(one)
			y1 = numpy.array([y1]).transpose().dot(one)
			x2 = numpy.array([x2]).transpose().dot(one)
			y2 = numpy.array([y2]).transpose().dot(one)
			
			# Compute distance matrix
			D = (x1 -x2.transpose())**2 + (y1-y2.transpose())**2
			[idM, idP] = numpy.nonzero((abs(D)**0.5)<glb.NODETOL*refd)
			
			ids1 = (f1)*NGauss+idM
			ids2 = (f2)*NGauss+idP
			
			if k1!=k2:
				mapP[ids1, k1] = mapM[ids2,k2]
			
			else:
				mapP[ids1, k1] = mapM[ids1,k1]	
				mapB = numpy.concatenate([mapB,mapM[ids1,k1]]) 
			  
				if glb.BCType[k1,f1]==glb.Wall:
				  mapW=numpy.concatenate([mapW,mapM[ids1,k1]])
				if glb.BCType[k1,f1]==glb.In:
				  mapI=numpy.concatenate([mapI,mapM[ids1,k1]])
				if glb.BCType[k1,f1]==glb.Out:
				  mapO=numpy.concatenate([mapO,mapM[ids1,k1]])
				if glb.BCType[k1,f1]==glb.Slip:
				  mapS=numpy.concatenate([mapS,mapM[ids1,k1]])
				if glb.BCType[k1,f1]==glb.Drichlet:
				  mapD=numpy.concatenate([mapD,mapM[ids1,k1]])
				if glb.BCType[k1,f1]==glb.Neuman:
				  mapN=numpy.concatenate([mapN,mapM[ids1,k1]])
				if glb.BCType[k1,f1]==glb.Cyl:
				  mapC=numpy.concatenate([mapC,mapM[ids1,k1]])
	
	global W
	W = numpy.array([numpy.concatenate([w,w,w])]).transpose().dot(numpy.ones([1,glb.K]))
	W = W*sJ
	
	#Flatten x and y and interp
	x=x.reshape([NGauss*glb.Nfaces,glb.K])
	y=y.reshape([NGauss*glb.Nfaces,glb.K])
	interp=interp.reshape([NGauss*glb.Nfaces,glb.Np])
	return()

def testGauss1(Ngauss=30):
	"""Test normals and connectivity maps with gaussian face interpolation"""
	
	#Normals
	gaussInit(Ngauss)
	plt.figure(1)
	plt.quiver(x.flatten(),y.flatten(),nx.flatten(),ny.flatten())
	plt.title("Gaussian Normals at the boundaries 1")

	#Check Connectivity maps
	plt.figure(2)
	plt.plot(x.flatten()[mapM],y.flatten()[mapM],'o')
	plt.title('Boundary elements mapM Gaussian' )
	
	plt.figure(3)
	plt.plot(x.flatten()[mapM],y.flatten()[mapM],'o')
	plt.title('Boundary elements mapP Gaussian')
	
	plt.figure(4)
	plt.plot(x.flatten()[mapB],y.flatten()[mapB],'o')
	plt.title('Physical boundary elements mapB Gaussian')
	
	plt.figure(5)
	plt.quiver(x.flatten()[mapM.flatten()],y.flatten()[mapM.flatten()],nx.flatten(),ny.flatten())
	plt.title("Gaussian Normals at the boundaries 2")

	if abs(x.flatten()[mapM]-x.flatten()[mapP]).max()>0.000000001:
		print "Maps aint correct!"
	else:
		print "Maps seem correct"

	if abs(x.flatten()[mapM.flatten()]-x.flatten()).max()>0.000000001:
		print "MapM aint correct!"
	else:
		print "MapM seem correct"

	if abs(nx.flatten()[mapM.flatten()]-nx.flatten()).max()>0.000000001:
		print "MapM aint correct!"
	else:
		print "MapM seem correct"

	if abs(nx.flatten()[numpy.delete(mapM,mapB)]+nx.flatten()[numpy.delete(mapP,mapB)]).max()>0.000000001:
		print "MapP aint correct!"
	else:
		print "MapP seem correct"

	return()

def testGauss2():
	import globalVar2D as glb
	import check2DInitiatation
	 # Check difference in nx,ny,rx,ry,sx,sy,J,sJ,x,y
	
	gaussInit(glb.Nfp)
	# Printing errors
	print (abs(nx-glb.nx)).max()
	print (abs(ny-glb.ny)).max()
	print (abs(rx-glb.rx.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	print (abs(ry-glb.ry.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	print (abs(sx-glb.sx.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	print (abs(sy-glb.sy.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	print (abs(J-glb.J.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	print (abs(sJ-glb.sJ)).max()
	print (abs(x-glb.x.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	print (abs(y-glb.y.flatten()[glb.vmapM].reshape([glb.Nfaces*glb.Nfp,glb.K]))).max()
	return()

def testGauss3():
	""" Check integration for arbitary fields"""
	
	# Initiate
	import globalVar2D as glb
	import check2DInitiatation
	
	gaussInit(30)
	
	# Check weight values
	print "Errors in weights"
	print abs(numpy.array(  [sum(W[0:Ng,el]) - (2*numpy.average(glb.sJ[0:glb.Nfp,el])) for el in range(glb.K)]  )).max()
	print abs(numpy.array(  [sum(W[Ng:2*Ng,el]) - (2*numpy.average(glb.sJ[glb.Nfp:2*glb.Nfp,el])) for el in range(glb.K)]  )).max()
	print abs(numpy.array(  [sum(W[2*Ng:3*Ng,el]) - (2*numpy.average(glb.sJ[2*glb.Nfp:3*glb.Nfp,el])) for el in range(glb.K)]  )).max()
	
	# find integration with old methodn
	nflux = numpy.ones([glb.Nfp*glb.Nfaces,glb.K])
	fluxRHS  = glb.LIFT.dot(glb.Fscale*nflux/2)
	fluxRHS = numpy.linalg.inv(glb.V.dot(glb.V.transpose())).dot(fluxRHS)*glb.J
	Iold = fluxRHS 
	
	# find integration with new method
	fl = numpy.ones([Ng*glb.Nfaces, glb.K])	
	If = interp.transpose().dot(W/2.*fl)
	Inew = If

	# Match values
	print "Errors"
	print abs(Iold-Inew).max()
