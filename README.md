AUTHOR: Achyut Panchal, Aerospace Engg., IIT Bombay

The code aims to solve ideal Magneto-hydrodynamic equations using nodal discontinuous Galerkin approach. The codes are inspired from a book by J. Hesthaven: www.nudg.org

Any hyperbolic equation can be solved using nodal DG with this code.

Currently, the code works for 1D and 2D linear and non-linear hyperbolic equations. Gradient limiting slope limiters are used to avoid negative densities. Hydrodynamic shocks can be properly captured using that.
However higher order accuracies cannot be achieved due to the nature of slope limiters, I am working on seperating smooth and non-smooth regions , and selectively applying slope limiters, to achieve higher accuracies.:w

python -c "import euler2D as eul; eul.testCurvedEuler()"   will solve a isentropic vortex problem.

Hydrodynamic Bow Shock at Mach 3
![alt tag](https://raw.github.com/Achyut2404/nodalDG/master/results/bowShock.png)

Hydrodynamic Forward Step Shock Structure at Mach 3
![alt tag](https://raw.github.com/Achyut2404/nodalDG/master/results/fwdStepM3_HD.png)

