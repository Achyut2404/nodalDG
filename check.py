# Codes for Nodal Discontinuous Galerkin Methods
#Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay
# The codes are inspired , and examples are followed from reference
#Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# Testing script for 1 dimensional initialization

##Check
import numpy
import functions
import math
#Order of polynomial
N=128
#Sample points
x=functions.JacobiGL(0,0,N)
#Function Defination
u=numpy.array([math.exp(math.sin(math.pi*i)) for i in x])
#u=numpy.array([math.sin(i) for i in x])
#Exact Derivative
uDerExac=numpy.array([(math.pi)*(math.cos(math.pi*i))*(math.exp(math.sin(math.pi*i))) for i in x])
#uDerExac=numpy.array([math.cos(i) for i in x])
#Calculate D matrix
Dr=functions.Dmatrix1D(N,x,functions.Vandermonde1D(N,x))
#Numerical Derivative
uDerNum=Dr.dot(u)
#Calculate L2 Norm
L2=[((uDerNum[i]-uDerExac[i])**2) for i in range(len(x))]
print sum(L2)
