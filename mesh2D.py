# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Reads fluent mesh files, and creates appropriate data for Nodal DG in 2 dimensions

import numpy
def readGmsh(Filename):
	"""Reads 2D gmsh filed and generates appropriate mesh data"""

	meshFile = open(Filename, 'r')
	import globalVar2D as glb
	lines = meshFile.readlines()
	# First 3 lines are details of gmsh version
	print lines[3][0:-1]
	
	# Check that physical names are defined for boundary conditions
	if (lines[3][0:-1]=="$PhysicalNames"):
		print "Reading physical names"
		noPhy = int(readFloat(lines[4][0:-1])[0])
		bcMap = numpy.zeros(noPhy) 
		# Read only boundary physical names (Hard coded for 2D only)
		cLine = 4  # Current line
		cLine += 1
		for n in range(noPhy):
			if(int(lines[cLine][0])==1):
				bcName = lines[cLine][5:-2]
				bcNum = int(lines[cLine][2])
				if bcName == "Inlet":
					bcMap[bcNum-1] = glb.In	
				if bcName == "Outlet":
					bcMap[bcNum-1] = glb.Out
				if bcName == "Wall":
					bcMap[bcNum-1] = glb.Wall
				if bcName == "Cyl":
					bcMap[bcNum-1] = glb.Cyl
			cLine = cLine + 1
		cLine = cLine + 1 # Ditch the EndPhysicalName line
		
	# Read Nodes
	print lines[cLine][0:-1]
	if (lines[cLine][0:-1]=="$Nodes"):
		print "Reading Nodes"
		cLine = cLine + 1
		Nv = int(readFloat(lines[cLine][0:-1])[0])
		print "Number of nodes are %i"%Nv
		cLine += 1
		# read node coordinates
		VX = numpy.array(range(Nv)).astype(float)
		VY = numpy.array(range(Nv)).astype(float)
		for i in range(Nv):
			tmpx=readFloat(lines[cLine][0:-1])
			VX[i] = tmpx[1]
			VY[i] = tmpx[2]
			cLine += 1
		cLine += 1

	# Read Elements
	print lines[cLine][0:-1]
	if (lines[cLine][0:-1]=="$Elements"):
		print "Reading Elements"
		cLine = cLine + 1
		KAll = int(readFloat(lines[cLine][0:-1])[0])
		print "Number of Boundary and surface Elements are %i"%KAll
		cLine += 1
	
		lineStart = cLine
		# Initiate bcFaces (stores line physical region values)
		bcFaces = []
		for i in range(noPhy):
			bcFaces.append([])
		
		for n in range(KAll):
			# Read a line/Physical map
			tmpx = readFloat(lines[cLine][0:-1])
		
			if tmpx[1] == 1:
				# if physical map is a line relate vortices to boundaries 
				bcCode = int(tmpx[3])
				v1 = int(tmpx[5])
				v2 = int(tmpx[6])
				bcFaces[bcCode-1].append([v1,v2])
				KBoundary = n+1

			cLine += 1	
		
		K = KAll - KBoundary
		cLine = lineStart	
		ele = 0
		# Initiate EToV (With wrong size, size will be changed later on)
		EToV = numpy.zeros([K,3]).astype(int).astype(int)

		for n in range(KAll):
			# Read a line/Physical map
			tmpx = readFloat(lines[cLine][0:-1])
		
			if tmpx[1] ==2:
				# If physical map is triangle, add element to EToV matrix
				v1 = int(tmpx[5])
				v2 = int(tmpx[6])
				v3 = int(tmpx[7])
				EToV[ele] = numpy.array([v1,v2,v3])
				ele = ele+1
				
			cLine += 1	
	
		cLine += 1
	
	# Based on these two maps, create BCType
	BCType = numpy.zeros([K,3]).astype(int)
	bcId = 0
	for eachBC in bcFaces:
		for eachFace in eachBC:
			v1 = eachFace[0]; v2 = eachFace[1]
			# Find element and face for vortices
			[k,f] = findFace(v1,v2,EToV)
			# Attach value in BCType
			BCType[k,f] = bcMap[bcId]
		bcId += 1
	
	# Close file
	meshFile.close()
	return([Nv, VX, VY, K, EToV-1,BCType])
	
def findFace(v1,v2,EToV):
	""" Find the face and element number of (v1,v2) from EToV"""
	K = EToV.shape[0]
	V = EToV.shape[1]
	for k in range(K):
		for v in range(V):
			if EToV[k,v] == v1:
				if v != V-1 and v != 0:
					if EToV[k,v+1] == v2:
						return(k,v)
					if EToV[k,v-1] == v2:
						return(k,v-1)
				if v == 0:
					if EToV[k,1] == v2:
						return(k,0)
					if EToV[k,-1] == v2:
						return(k,V-1)
				if v == V-1:
					if EToV[k,0] == v2:
						return(k,V-1)
					if EToV[k,V-2] == v2:
						return(k,V-2)
	return(0)

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
	BCType=numpy.zeros([K,Nv])
	return([Nv, VX, VY, K, EToV-1,BCType])

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
