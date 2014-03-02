# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Reads fluent mesh files, and creates appropriate data for Nodal DG in 2 dimensions

import numpy

def createBC(Filename):
	"""Reads 2D .neu files created by gambit and converts them to required mesh format"""
	f=open(Filename, 'r')
		
	lines=f.readlines()
	#First 6 lines are intro
	# Find number of nodes and number of elements
	dims=readFloat(lines[6][0:-2])
	Nv = int(dims[0])
	K = int(dims[1])
	BCType = numpy.zeros([K,3])
	
	# No info in 7th and 8th line
	# read node coordinates
	VX = numpy.array(range(Nv)).astype(float)
	VY = numpy.array(range(Nv)).astype(float)
	for i in range(Nv):
	  tmpx=readFloat(lines[9+i][0:-2])
	  VX[i] = tmpx[1]
	  VY[i] = tmpx[2]

	#no info in lines 9+Nv and 9+Nv+1
	
	# read element to node connectivity
	EToV = numpy.zeros([K, 3]).astype(int)
	for k in range(K):
	  tmpcon=readFloat(lines[9+Nv+2+k][0:-2])
	  EToV[k,0] = int(tmpcon[3])
	  EToV[k,1] = int(tmpcon[4])
	  EToV[k,2] = int(tmpcon[5])
	
	#Skip through the material selection section
	lastLine=9+Nv+2+k
	lastLine=lastLine+2
	while(lines[lastLine]!=lines[-1]):
		#Check for ENDOFSECTION
		lastLine=lastLine+1
	
	#Boundary starts from lastLine+1
	BCstart=lastLine+2

	# boundary codes (defined in Globals2D)
	import globalVar2D as glb
	
	# Read all the boundary conditions at the nodes
	i=BCstart
	while(lines[i]!=lines[-1]):
		bcflag=0
		if lines[i].find('In')!=-1:
			bcflag = glb.In
			tag=lines[i].find('In')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Out')!=-1:
			bcflag = glb.Out
			tag=lines[i].find('Out')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Wall')!=-1:
			bcflag = glb.Wall
			tag=lines[i].find('Wall')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Far')!=-1:
			bcflag = glb.Far
			tag=lines[i].find('Far')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Cyl')!=-1:
			bcflag = glb.Cyl
			tag=lines[i].find('Cyl')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Dirichlet')!=-1:
			bcflag = glb.Dirichlet
			tag=lines[i].find('Dirichlet')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Neuman')!=-1:
			bcflag = glb.Neuman
			tag=lines[i].find('Neuman')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		if lines[i].find('Slip')!=-1:
			bcflag = glb.Slip
			tag=lines[i].find('Slip')
			nPoints=int(readFloat(lines[i][tag+5:-2])[1])
		
		i=i+1
		if bcflag!=0:
			for j in range(nPoints):
				tmpcon=readFloat(lines[i][0:-2])
				BCType[int(tmpcon[0])-1,int(tmpcon[2])-1] = bcflag
				i=i+1
				
	# Close file
	f.close()
	return([Nv, VX, VY, K, EToV-1,BCType])
	
def create(Filename):
	"""Reads 2D .neu files created by gambit and converts them to required mesh format"""
	f=open(Filename, 'r')
		
	lines=f.readlines()
	#First 6 lines are intro
	# Find number of nodes and number of elements
	dims=readFloat(lines[6][0:-2])
	Nv = int(dims[0])
	K = int(dims[1])
	
	# No info in 7th and 8th line
	# read node coordinates
	VX = numpy.array(range(Nv)).astype(float)
	VY = numpy.array(range(Nv)).astype(float)
	for i in range(Nv):
	  tmpx=readFloat(lines[9+i][0:-2])
	  VX[i] = tmpx[1]
	  VY[i] = tmpx[2]

	#no info in lines 9+Nv and 9+Nv+1
	
	# read element to node connectivity
	EToV = numpy.zeros([K, 3]).astype(int)
	for k in range(K):
	  tmpcon=readFloat(lines[9+Nv+2+k][0:-2])
	  EToV[k,0] = int(tmpcon[3])
	  EToV[k,1] = int(tmpcon[4])
	  EToV[k,2] = int(tmpcon[5])
	
	# Close file
	f.close()
	return([Nv, VX, VY, K, EToV-1])

def readFloat(line):
	line=line.split(' ') 
	newarray=[]
	for i in line:
		if i=='':
			doNone=1
		else:
			newarray.append(float(i))
	return(newarray)

def tempCreate():
	"""Create a temporary mesh, just by putting random numbers"""
	Nv=9
	K=8
	tempXY=numpy.array([[-1.00000000000e+00,-1.00000000000e+00],
	[1.00000000000e+00,-1.00000000000e+00],
	[0.00000000000e+00,-1.00000000000e+00],
	[1.00000000000e+00, 1.00000000000e+00],
	[1.00000000000e+00, 0.00000000000e+00],
	[-1.00000000000e+00, 1.00000000000e+00],
	[0.00000000000e+00, 1.00000000000e+00],
	[-1.00000000000e+00, 0.00000000000e+00],
	[-1.63975270435e-01, -1.63975270435e-01]])
	
	[VX,VY]=tempXY.transpose()
	
	EToV=numpy.array([[  8    ,   1  ,     9],
	[9,       1     ,  3],
	[7 ,      6    ,   8],
	[4  ,     7   ,    5],
	[2   ,    5       ,3],
	[7    ,   8      , 9],
	[5     ,  7     ,  9],
	[3      , 5    ,   9]])
	return([Nv, VX, VY, K, EToV-1])
