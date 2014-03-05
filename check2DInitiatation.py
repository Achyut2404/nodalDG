# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Testing script for 2-dimensional initialization

#Initialize everything
import globalVar2D as glb
import math
import numpy
import matplotlib.pyplot as plt

glb.globalInit()
# Polynomial order used for approximation 
glb.N = 9

# Read in Mesh
import mesh2D
[glb.Nv, glb.VX, glb.VY, glb.K, glb.EToV,glb.BCType] = mesh2D.createBC('Grid/Euler2D/vortexA04.neu')

### Initialize solver and construct grid and metric
execfile("initiate2D.py")

###Check overall Mesh
print "Checking overall Mesh"
#Check size of x and y
if glb.x.shape[0]!=glb.Np or glb.x.shape[1]!=glb.K or glb.y.shape[0]!=glb.Np or glb.y.shape[1]!=glb.K:
	print "Error in shapes of x and y"
else:
	print "x and y shapes checked!"
if __name__ == "__main__":
	#Plot all points
	plt.figure(1)
	plt.title('All points')
	plt.plot(glb.x,glb.y,'o')
print "x and y checked!"

###Check differentiation matrices
print "Checking differentiation matrices"
#Check their shape
if glb.Dr.shape[0]!=glb.Np or glb.Dr.shape[1]!=glb.Np or glb.Ds.shape[0]!=glb.Np or glb.Ds.shape[1]!=glb.Np:
	print "Error in shapes of Dr and Ds"
else:
	print "Shapes of Dr and Ds checked"

###Check LIFT
print "Dont know how to check LIFT!"
#Dont know how to do that!!

###Check Div,curl grad

#Div
#Define a vector field
u=glb.x*glb.y
v=-(glb.y**2)/2.0
divu=functions2D.Div2D(u,v)
if abs(divu).max()<0.0000001:
	print "Divergence seems correct"
else:
	print "Error in divergence"

#Grad
#Define a scalar field
u=glb.x*glb.y
[gradu,gradv]=functions2D.Grad2D(u)
err=max(abs(gradu-glb.y).max(),abs(gradv-glb.x).max())
if err<0.0000001:
	print "Grad seems cool"
else:
	print "Problem in grad"

#Curl
u=glb.y*glb.y/2.0
v=glb.x*glb.y
[curlu,curlv,curlw]=functions2D.Curl2D(u,v)
if abs(curlw).max()<0.0000001:
	print "Curl is cool!"
else:
	print "Error in curl"

###Check Normals
if __name__=="__main__":
	plt.figure(2)
	plt.quiver(glb.x.flatten()[glb.vmapM],glb.y.flatten()[glb.vmapM],glb.nx.flatten(),glb.ny.flatten())
	plt.title("Normals at the boundaries")

###Check Connectivity maps
# Check 1:
if abs(glb.vmapM[glb.mapM]-glb.vmapM).max()>0.000001:
	print "Error in maps vmapM"
if abs(glb.vmapM[glb.mapP]-glb.vmapP).max()>0.000001:
	print "Error in maps vmapP"
if abs(glb.nx.flatten()[numpy.delete(glb.mapM,glb.mapB)]+glb.nx.flatten()[numpy.delete(glb.mapP,glb.mapB)]).max()>0.000001:
	print "Error in mapM or mapP"
if abs(glb.ny.flatten()[numpy.delete(glb.mapM,glb.mapB)]+glb.ny.flatten()[numpy.delete(glb.mapP,glb.mapB)]).max()>0.000001:
	print "Error in mapM or mapP"
if __name__=="__main__":
	plt.figure(3)
	plt.plot(glb.x.flatten()[glb.vmapM],glb.y.flatten()[glb.vmapM],'o')
	plt.title('Boundary elements mapM')

	plt.figure(4)
	plt.plot(glb.x.flatten()[glb.vmapM],glb.y.flatten()[glb.vmapM],'o')
	plt.title('Boundary elements mapP')

	plt.figure(5)
	plt.plot(glb.x.flatten()[glb.vmapB],glb.y.flatten()[glb.vmapB],'o')
	plt.title('Physical boundary elements mapB')

if abs(glb.x.flatten()[glb.vmapM]-glb.x.flatten()[glb.vmapP]).max()>0.000000001:
	print "Some error in maps"
else:
	print "Maps seem correct"

#Check filter matrix
print "Checking filter matrix"
filt=functions2D.cutOffFilter2D(glb.N,0.95)
