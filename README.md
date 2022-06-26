# Lunar Lander Optimal Control
Inspired by the [Lunar Lander Atari](https://en.wikipedia.org/wiki/Lunar_Lander_(1979_video_game)) game, this project aims to perform optimal control on a lunar lander system. Direct collocation is used to obtain an optimal trajectory for simulating the dynamics. Trapeziodal quadrature is used to approximate the various integrals used in the optimal control problem. Runge-Kutta 4 is used to simulate the dynamics of the system. This is the final project for MATH 514 - Numerical Analysis.

## Usage
To run the code, the three main dependencies [numpy](https://numpy.org/), [ipopt](https://coin-or.github.io/Ipopt/), and [pyomo](http://www.pyomo.org/) need to be installed. Note `lander_sim.py` is the main file, and can be run in order to solve the optimal control problem, simulate the dynamics and create a movie of the simulation. For comparison, `lander_PD.py` simulates the dynamics using a hand-tuned state-feedback controller. The other python files store functions used by `lander_sim.py` and `lander_PD.py`.

## Some Examples
- Example of optimal trajectory with path shown.
- Low gravity example with g = 0.1 m/s^2.
- A flip can be achieved by setting the initial angle to 2\*pi, and the final angle to 0.

![](movies/gifs/show_traj.gif)
![](movies/gifs/low_grav.gif)
![](movies/gifs/flip.gif)
