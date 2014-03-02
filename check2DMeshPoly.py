# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Testing script for 2 dimensional mesh import

import functions2D
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#Order of points
NPoints=40
#Order of polynomials
Ni=1
Nj=5
#Calculation
#Create Mesh
[x,y]=functions2D.Nodes2D(NPoints)
[r,s]=functions2D.xytors(x,y)
#Create Polynomials
[a,b]=functions2D.rstoab(r,s)
P=functions2D.Simplex2DP(a,b,Ni,Nj)
#Check Vandermonde
NVander=10
Vander=functions2D.Vandermonde2D(NVander, r, s)
gradVander=functions2D.GradVandermonde2D(NVander,r,s)
print Vander.shape
print gradVander[0].shape
print gradVander[1].shape
#Plot

plt.figure(1)
plt.scatter(x,y,c=P,cmap=cm.jet)

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(x,y,P,c=P,cmap=cm.jet)

plt.show()
