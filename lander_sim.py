import os
from scipy.interpolate import interp1d
import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
from lander_vis import VisualizeLander
from traj_solver import SolveTrajectory
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
file_name = "OPT_controller011"
fps = 20 # frames per second of movie
meta_data = (op_sys, open_mov, file_name, fps)

# Constants
bv = 0.5; bo = 1
g = 9.8; m = 10
rotI = (13/12)*m
Const = np.array(([bv,bo,m,g,rotI])) # order matters with these consts

# Initial State
x0 = -2; y0 = 0; ang0 = 2*pi;
vx0 = 0; vy0 = 0; omega0 = 0;
X0 = np.array(([x0],[y0],[ang0],[vx0],[vy0],[omega0])) # initial state
X = X0

# Initial Control
max_thrust = 5000
max_torque = 200
thrust0 = m*g
torque0 = 0
U0 = np.array(([thrust0],[torque0]))
U = U0

# Bound on Control
Ubound = np.array(([max_thrust], [max_torque]))

# Reference tracking state
xref = 2; yref = 0; angref = 0;
vxref = 0; vyref = 0; omegaref = 0;
Xref = np.array(([xref],[yref],[angref],\
    [vxref],[vyref],[omegaref])) # reference state

# Total time and time steps size
T = 10
h = 0.0025
step_sizes = np.array(([h]))

# Solve optimal trajectory
numColl = 100 # number of collocation points
OPT_TRAJ = SolveTrajectory(X0, Xref, Ubound, Const, T, numColl, op_sys)
opt_x      = np.array(OPT_TRAJ[:,0])
opt_y      = np.array(OPT_TRAJ[:,1])
opt_ang    = np.array(OPT_TRAJ[:,2])
opt_vx     = np.array(OPT_TRAJ[:,3])
opt_vy     = np.array(OPT_TRAJ[:,4])
opt_omega  = np.array(OPT_TRAJ[:,5])
opt_thrust = np.array(OPT_TRAJ[:,6])
opt_torque = np.array(OPT_TRAJ[:,7])
opt_times  = np.array(OPT_TRAJ[:,8])

# Interpolate optimal control trajectory
interp_order = 'quadratic'
t_steps  = np.linspace(0,T-1e-10,int(T/h),endpoint=True).reshape(-1,1)
thrust_interp = interp1d(opt_times, opt_thrust, kind=interp_order)
torque_interp = interp1d(opt_times, opt_torque, kind=interp_order)
U_interp = np.hstack((thrust_interp(t_steps),torque_interp(t_steps)))

# Interpolate optimal state trajectory
x_interp     = interp1d(opt_times, opt_x, kind=interp_order)
y_interp     = interp1d(opt_times, opt_y, kind=interp_order)
ang_interp   = interp1d(opt_times, opt_ang, kind=interp_order)
vx_interp    = interp1d(opt_times, opt_vx, kind=interp_order)
vy_interp    = interp1d(opt_times, opt_vy, kind=interp_order)
omega_interp = interp1d(opt_times, opt_omega, kind=interp_order)
X_interp = np.hstack((x_interp(t_steps),y_interp(t_steps),\
           ang_interp(t_steps),vx_interp(t_steps),vy_interp(t_steps),\
           omega_interp(t_steps)))

# Initialize data arrays
x_arr = np.zeros((int(T/h)+1))
y_arr = np.zeros((int(T/h)+1))
ang_arr = np.zeros((int(T/h)+1))
u0_arr = np.zeros((int(T/h)+1))  # thrust array
u1_arr = np.zeros((int(T/h)+1))  # torque array
t_arr = np.zeros((int(T/h)+1))

# Run simulation
print('Running simulation...')
print()

for j in range(int(T/h)):
    # gain matrix for closed-loop control
    # only works for small angles
    if abs(X[2]) < pi/2.5:
        a = 1
    else:
        a = 0
    K = a*np.array(([0, -60*m*g*(2+cos(X[2][0]-pi)), 0, 0, -80*m*g*(2+cos(X[2][0]-pi)), 0],\
                  [8, 0, -75, 10, 0, -40]))
    U_opt = U_interp[j,:].reshape(-1,1)
    Xref_new = X_interp[j,:].reshape(-1,1)
    U = K@(X-Xref_new) + U_opt
    #U = U_opt
    U[0] = np.clip(U[0],0,max_thrust) # constraint thrust
    U[1] = np.clip(U[1],-max_torque,max_torque) # constrain torque

    x_arr[j] = X[0]
    y_arr[j] = X[1]
    ang_arr[j] = X[2]
    u0_arr[j] = U[0]
    u1_arr[j] = U[1]
    t_arr[j] = h*j

    # RK-4
    k1 = F(X,U,Const)
    k2 = F(X+h/2*k1,U,Const)
    k3 = F(X+h/2*k2,U,Const)
    k4 = F(X+h*k3,U,Const)
    k = (k1+2*k2+2*k3+k4)/6
    X = X + h*k

x_arr[j+1] = X[0]
y_arr[j+1] = X[1]
ang_arr[j+1] = X[2]
u0_arr[j+1] = U[0]
u1_arr[j+1] = U[1]
t_arr[j+1] = h*j

# Visualizations
    # Suppress GTK warning outputs
fd = os.open('/dev/null',os.O_WRONLY)
os.dup2(fd,2)

# Make plot of trajectory

print('Creating plot...')
print()
csfont = {'fontname':'Times New Roman'}
plt.plot(x_arr,y_arr + 0.75,'k',label="sim trajectory")
plt.plot(OPT_TRAJ[:,0],OPT_TRAJ[:,1] + 0.75,'g',label="opt trajectory")
plt.legend()
plt.xlabel("x position")
plt.ylabel("y position")
plt.axis("equal")

plt.figure()
plt.plot(t_arr,ang_arr,'k',label="sim trajectory")
plt.plot(OPT_TRAJ[:,8],OPT_TRAJ[:,2],'g',label="opt trajectory")
plt.legend()
plt.xlabel("time")
plt.ylabel("angle")
plt.show()

# Create a movie of the simulation:
VisualizeLander(x_arr,y_arr,ang_arr,u0_arr,u1_arr,t_arr,Xref,meta_data)
