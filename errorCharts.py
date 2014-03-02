# Codes for Nodal Discontinuous Galerkin Methods
# Written by: Achyut Panchal 
# Aerospace Engineering, Indian Institute of Technology Bombay 
# The codes are inspired , and examples are followed from reference
# Jan S. Hesthaven. Nodal Discontinuous Galerkin Methods. Springer, 2008

# This plots L2 error norms of various problems using DG
# Values in this are entered manually, by running the codes each time
# Manually created by Achyut on 20 Feb 2014

import matplotlib.pyplot as plt

# Advection continuous wave
#Variation with grid points
consPoly=5
NPoints=[5,10,20,50,100]
L2Norm=[5.8e-06,1.38e-07,5.47e-09,1.68e-11,8.30e-13]
time=[0.04,0.1,0.19,0.5,1.09]
#Plot Stuff
plt.figure(1)
plt.semilogx()
plt.semilogy()
plt.xlim([2,110])
plt.ylabel('L2 Norm')
plt.xlabel('Number of elements')
plt.rc('font', family='serif')
plt.plot(NPoints,L2Norm,'o') 
plt.title("L2 Error: Varying Grid Points,  Polynomial Order=%i  \n Linear 1D Advection Equation:Continuous Solution"%consPoly)

#Variation with polynomial order
consPoints = 50
NPoly = [1,2,4,6,8,12]
L2Norm = [1.51e-03,9.47e-06,4.8e-10,3.8e-12,4.65e-13,8.23e-15]
time = [0.06,0.11, 0.34, 0.7, 1.22, 2.73]
# Plot stuff
plt.figure(2)
plt.xlim([0.9,15])
plt.ylabel('L2 Norm')
plt.xlabel('Polynomial Order')
plt.rc('font', family='serif')
plt.plot(NPoly,L2Norm,'o')
plt.semilogx()
plt.semilogy()
plt.title("L2 Error: Varying Order of Polynomials,  Grid Points=%i  \n Linear 1D Advection Equation: Continuous Solution"%consPoints)

# Advection Discontinuous wave
#Variation with grid points
consPoly=5
NPoints=[5,10,20,50,100,200,400]
L2Norm=[0.23,0.12,0.1,0.07,0.05,0.03,0.026]
time=[0.04,0.1,0.19,0.5,1.09,2.51,6.32]
#Plot Stuff
plt.figure(3)
plt.ylabel('L2 Norm')
plt.xlabel('Number of elements')
plt.xlim([2,410])
plt.semilogx()
plt.semilogy()
plt.rc('font', family='serif')
plt.plot(NPoints,L2Norm,'o') 
plt.title("L2 Error: Varying Grid Points,  Polynomial Order=%i  \n Linear 1D Advection Equation:Discontinuous Solution"%consPoly)

#Variation with polynomial order
consPoints = 200
NPoly = [1,2,4,6,8,12]
L2Norm = [0.1,0.059,0.045,0.035, 0.031,0.024]
time = [0.3,0.55, 1.68, 3.59, 6.46, 15.69]
# Plot stuff
plt.figure(4)
plt.xlim([0.5,13])
plt.ylim([0.01,0.15])
plt.rc('font', family='serif')
plt.plot(NPoly,L2Norm,'o')
plt.ylabel('L2 Norm')
plt.xlabel('Polynomial Order')
plt.semilogx()
plt.semilogy()
plt.title("L2 Error: Varying Order of Polynomials,  Grid Points=%i  \n Linear 1D Advection Equation: Discontinuous Solution"%consPoints)

# Euler Sod shock tube problem
#Variation with grid points
consPoly=8
NPoints=[10,50,100,200,400]
L2Norm=[0.064,0.032,0.026,0.0234, 0.02]
time=[1.23,23.43,88.27,339.5,1331.41]
#Plot Stuff
plt.figure(7)
plt.xlim([9,410])
plt.ylabel('L2 Norm')
plt.xlabel('Number of elements')
plt.semilogx()
plt.semilogy()
plt.rc('font', family='serif')
plt.plot(NPoints,L2Norm,'o') 
plt.title("L2 Error: Varying Grid Points,  Polynomial Order=%i  \n Sod Shock Tube"%consPoly)

#Variation with polynomial order
consPoints = 100
NPoly = [1,2,4,6]
L2Norm = [0.03, 0.0268, 0.0266,0.0264]
time = [4.42,21.70,31.81,64.02]
# Plot stuff
plt.figure(8)
plt.xlim([0.9,7])
plt.ylim([0.02,0.04])
plt.ylabel('L2 Norm')
plt.xlabel('Polynomial Order')
plt.rc('font', family='serif')
plt.plot(NPoly,L2Norm,'o')
plt.title("L2 Error: Varying Order of Polynomials,  Grid Points=%i  \n Sod Shock Tube"%consPoints)

# 2D Linear maxwell equation
#Variation with polynomial order
consPoints = 146
NPoly = [1,2,4,6,8,12]
L2Norm = [0.031,0.0037,1.57e-05, 3.49e-08, 4.93e-11,2.31e-13]
time = [0.27,0.55, 3.03, 12.41, 41.05, 454.88]
# Plot stuff
plt.figure(5)
plt.xlim([0.9,13])
plt.rc('font', family='serif')
plt.ylabel('L2 Norm')
plt.xlabel('Polynomial Order')
plt.plot(NPoly,L2Norm,'o')
plt.semilogx()
plt.semilogy()
plt.title("L2 Error: Varying Order of Polynomials,  Grid Points=%i  \n 2D Linear Equation: Continuous Solution"%consPoints)

# 2D Non-linear Euler smooth wave
#Variation with polynomial order
consPoints = 256
NPoly = [1,2,4,6,8]
L2Norm = [0.030,0.010,0.0044,0.0016, 0.0004]
time = [4.61,18.46, 92.35, 266.89, 589.43]
# Plot stuff
plt.figure(6)
plt.xlim([0.9,9])
plt.rc('font', family='serif')
plt.plot(NPoly,L2Norm,'o')
plt.ylabel('L2 Norm')
plt.xlabel('Polynomial Order')
plt.semilogx()
plt.semilogy()
plt.title("L2 Error: Varying Order of Polynomials,  Grid Points=%i  \n Linear 2D Euler Equation: Continuous Solution"%consPoints)

plt.show()
