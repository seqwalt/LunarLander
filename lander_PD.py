import os
import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
from lander_vis import VisualizeLander
# ----------
# States: in SI units (Newtons; m/s; kg; etc.)
# x = horizontal position; y = vertical position; ang = tilt angle
# vx = x velocity; vy = y velocity; omega = angular velocity
# --
# Constants: in SI units (Newotons; m/s; kg; etc.)
# b = drag coeff; g = absolute value of gravity
# m = mass; rotI = moment of inertia
# --
# Control inputs: thrust and torque
# ----------
def F(STATE,CONTROL,CONSTANTS):
    x,y,ang,vx,vy,omega = STATE
    thrust,torque = CONTROL
    bv,bo,m,g,rotI = CONSTANTS

    f_x  = vx; f_y = vy; f_ang = omega  # velocities
    f_vx = -bv*vx + thrust*sin(-ang)/m    # accelerations
    f_vy = -(g + bv*vy) + thrust*cos(ang)/m
    f_omega  = -bo*omega + torque/rotI

    vector = np.array(([f_x, f_y, f_ang, f_vx, f_vy, f_omega]))
    return vector.reshape(6,1)

# meta data
op_sys = "linux" # options are "linux" or "mac"
open_mov = 1     # open movie on completion? 1=yes, 0=no
file_name = "PD_controller002"
fps = 15 # frames per second of movie
meta_data = (op_sys, open_mov, file_name, fps)

# Constants
bv = 0.5; bo = 1;
g = 9.8; m = 10
rotI = (13/12)*m
Const = np.array(([bv,bo,g,m,rotI]))

# Initial State
x0 = 4; y0 = 2; ang0 = 0;
vx0 = 0; vy0 = 0; omega0 = 0;
X0 = np.array(([x0],[y0],[ang0],[vx0],[vy0],[omega0])) # initial state
X = X0

# Initial Control
max_thrust = 100*m*g
thrust0 = m*g
torque0 = 0
U0 = np.array(([thrust0],[torque0]))
U = U0

# Reference tracking state
xref = 1; yref = 0; angref = 0;
vxref = 0; vyref = 0; omegaref = 0;
Xref = np.array(([xref],[yref],[angref],\
    [vxref],[vyref],[omegaref])) # reference state

# Total time and time steps sizes
T = 10
h = 0.0025
step_sizes = np.array(([h]))

print()
print('Running simulation...')
print()
# Run simulation
for i in range(len(step_sizes)):
    X = X0
    U = U0
    h = step_sizes[i]
    x_arr = np.zeros((int(T/h)+1))
    y_arr = np.zeros((int(T/h)+1))
    ang_arr = np.zeros((int(T/h)+1))
    u0_arr = np.zeros((int(T/h)+1))  # thrust array
    u1_arr = np.zeros((int(T/h)+1))  # torque array
    t_arr = np.zeros((int(T/h)+1))

    for j in range(int(T/h)+1):
        X = X + h*F(X,U,Const)
        x_arr[j] = X[0]
        y_arr[j] = X[1]
        ang_arr[j] = X[2]
        u0_arr[j] = U[0]
        u1_arr[j] = U[1]
        t_arr[j] = h*j

        K = np.array(([0, -60*m*g*(2+cos(X[2][0]-pi)), 0, 0, -80*m*g*(2+cos(X[2][0]-pi)), 0],\
                      [8, 0, -75, 10, 0, -40]))
        U = K@(X-Xref)
        U[0] = U[0] + m*g
        U[0] = np.clip(U[0],0,max_thrust) # constraint thrust
        #U[1] = np.clip(U[1],-70,70) # constrain torque

# Suppress GTK warning outputs
fd = os.open('/dev/null',os.O_WRONLY)
os.dup2(fd,2)

# Create a movie of the simulation:
#VisualizeLander(x_arr,y_arr,ang_arr,u0_arr,u1_arr,t_arr,Xref,meta_data)
#'''
plt.plot(x_arr,y_arr,'k',label="trajectory")
#plt.plot(t_arr,ang_arr,'g.')
plt.legend()
plt.xlabel("x position")
plt.ylabel("y position")
plt.axis("equal")
plt.show()
#'''
