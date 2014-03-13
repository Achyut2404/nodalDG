AUTHOR: Achyut Panchal, Aerospace Engg., IIT Bombay

The code aims to solve ideal Magneto-hydrodynamic equations using nodal discontinuous Galerkin approach. The codes are inspired from a book by J. Hesthaven: www.nudg.org

Any hyperbolic equation can be solved using nodal DG with this code.

Currently, the code works for 1D and 2D linear and non-linear hyperbolic equations. I am writing slope-limiters in order to remove negative densities, and simulate shock waves.

python -c "import euler2D as eul; eul.testCurvedEuler()"   will solve a isentropic vortex problem.

![alt tag](https://raw.github.com/Achyut2404/nodalDG/master/results/bowShock.png)
![alt tag](https://raw.github.com/Achyut2404/nodalDG/master/results/fwdStepM3_HD.png)

