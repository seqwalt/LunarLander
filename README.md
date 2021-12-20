# LunarLander
Direct collocation is used to obtain an optimal trajectory for a lunar lander simulation. Trapeziodal quadrature is used to approximate the various integrals used in the optimal control problem. Runge-Kutta 4 is used to simulate the dynamics of the system. This is the final project for MATH 514 - Numerical Analysis.

## Usage
To run the code, the three main dependencies `numpy`, `ipopt`, and `pyomo` need to be installed. The file `lander_sim.py` can be run in order to solve the optimal control problem, simulate the dynamics and create a movie of the simulation.

## Examples
- Example of optimal trajectory with path shown.
- Low gravity example with g = 0.1 m/s^2.
- A flip can be achieved by setting the initial angle to 2\*pi, and the final angle to 0.

![](movies/gifs/show_traj.gif) ![](movies/gifs/flip.gif)

![](movies/gifs/low_grav.gif)


