import os
import platform
from scipy.interpolate import interp1d
import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
from traj_solver import SolveTrajectory
from create_files import TrajFiles
from lander_vis import LanderMovie
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
    f_vx = -bv*vx/m + thrust*sin(-ang)/m    # accelerations
    f_vy = -(g + bv*vy/m) + thrust*cos(ang)/m
    f_omega  = -bo*omega/rotI + torque/rotI

    vector = np.array(([f_x, f_y, f_ang, f_vx, f_vy, f_omega]))
    return vector.reshape(6,1)

if platform.system() == "Linux":
    OP_SYS = "linux"
elif platform.system() == "Darwin":
    OP_SYS = "mac"
# meta data
op_sys = OP_SYS
gen_mov  = 1     # generate movie on completion? 1=yes, 0=no
open_mov = 1     # open movie on completion? 1=yes, 0=no
traj_data_generate = 0 # generate trajectory csv and readme? 1=yes, 0=no
make_plots = 0   # generate trajectory and control plots? 1=yes, 0=no
rand_BC = 0      # Random boundary conditions? 1=yes, 0=no
file_name = "OPT_controller011" # movie file name
fps = 15 # frames per second of movie
dpi = 100 # resolution of movie (dots per inch)
meta_data = (op_sys, open_mov, file_name, fps, dpi)

# Constants
bv = 5; bo = 11 # b_v and b_{\omega}
g = 1.62 # moon
# g = 9.81 # earth
# g = 0.2
m = 10 # mass
rotI = (13/12)*m
Const = np.array(([bv,bo,m,g,rotI])) # order matters with these consts

# Set boundary conditions and final time
if rand_BC != 1:
    # Not random boundary conditions
    # Total time in seconds
    T = 3

    # Initial State
    x0 = -3; y0 = 0; ang0 = 0*2*pi;
    vx0 = 0; vy0 = 0; omega0 = 0;
    X0 = np.array(([x0],[y0],[ang0],[vx0],[vy0],[omega0])) # initial state
    X = X0

    # Reference tracking state
    xref = 3; yref = 0; angref = 0;
    vxref = 0; vyref = 0; omegaref = 0;
    Xref = np.array(([xref],[yref],[angref],\
        [vxref],[vyref],[omegaref])) # reference state
else:
    # Random boundary conditions
    rand_init = lambda : 2*(np.random.rand(6,1) - 0.5) # vals from -1 to 1
    range_vals = np.array(([-5,5],[0,10],[-2*np.pi,2*np.pi],[-3,3],[-3,3],[-2*np.pi,2*np.pi])) # row 1: x range, row 2: y range etc.
    centers = np.mean(range_vals,1).reshape(-1,1)
    radii = 0.5*(range_vals[:,1] - range_vals[:,0]).reshape(-1,1)

    X0 = radii*rand_init() + centers # random initial conditions
    X = X0
    Xref = radii*rand_init() + centers # random final conditions

    T_range = np.array(([5,10]))
    T = np.random.rand()*(T_range[1] - T_range[0]) + T_range[0] # random final time

# Time step size
h = 0.0025
step_sizes = np.array(([h]))

# Initial Control
max_thrust = 5000
max_torque = 1000
thrust0 = m*g
torque0 = 0
U0 = np.array(([thrust0],[torque0]))
U = U0

# Bound on Control
Ubound = np.array(([max_thrust], [max_torque]))

# Solve optimal trajectory
numColl = 100 # number of collocation points
opt_dict = SolveTrajectory(X0, Xref, Ubound, Const, T, numColl, op_sys)
if opt_dict['feasible'] == False:
    # abort
    print('Aborting without creating files.')
    exit()

OPT_TRAJ = opt_dict['traj']
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
vx_arr = np.zeros((int(T/h)+1))
vy_arr = np.zeros((int(T/h)+1))
omega_arr = np.zeros((int(T/h)+1))
u0_arr = np.zeros((int(T/h)+1))  # thrust array
u1_arr = np.zeros((int(T/h)+1))  # torque array
t_arr = np.zeros((int(T/h)+1))

# Run simulation
print('Running simulation...')
print()

# Runge-Kutta 4 simulation
for j in range(int(T/h)):
    # closed-loop control
    temp = 20*cos(X[2][0])
    K = np.array(([0, -100*temp, 0, 0, -20*temp, 0], [10*temp, 0, -262.5, 10*temp, 0, -140]))
    U_opt = U_interp[j,:].reshape(-1,1)
    Xref_new = X_interp[j,:].reshape(-1,1)
    U = K@(X-Xref_new) + U_opt
    # U = U_opt
    U[0] = np.clip(U[0],0,max_thrust) # constrain thrust
    U[1] = np.clip(U[1],-max_torque,max_torque) # constrain torque

    # Store data
    x_arr[j] = X[0]
    y_arr[j] = X[1]
    ang_arr[j] = X[2]
    vx_arr[j] = X[3]
    vy_arr[j] = X[4]
    omega_arr[j] = X[5]
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

# Store final time-step data
x_arr[j+1] = X[0]
y_arr[j+1] = X[1]
ang_arr[j+1] = X[2]
vx_arr[j+1] = X[3]
vy_arr[j+1] = X[4]
omega_arr[j+1] = X[5]
u0_arr[j+1] = U[0]  # Thrust
u1_arr[j+1] = U[1]  # Torque
t_arr[j+1] = h*(j+1)

# Generate trajectory data files (csv and readme)
if traj_data_generate == 1:
    TrajFiles(t_arr,x_arr,y_arr,ang_arr,vx_arr,vy_arr,omega_arr,\
    u0_arr,u1_arr,Const,X0,Xref,U0,Ubound,T,h)

# Visualizations
    # Suppress GTK warning outputs
# fd = os.open('/dev/null',os.O_WRONLY)
# os.dup2(fd,2)

# Make plot of trajectory
if make_plots == 1:
    print('Creating plot...')
    print()

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.plot(x_arr,y_arr + 0.75,'k:',label="sim trajectory")
    plt.plot(OPT_TRAJ[:,0],OPT_TRAJ[:,1] + 0.75,'g',label="opt trajectory")
    plt.plot(x_arr[0],y_arr[0]+0.75,'bo',label="start point")
    plt.legend()
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")

    plt.figure()
    plt.plot(t_arr,ang_arr,'k:',label="sim trajectory")
    plt.plot(OPT_TRAJ[:,8],OPT_TRAJ[:,2],'g',label="opt trajectory")
    plt.plot(t_arr[0],ang_arr[0],'bo',label="start point")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.show()

# Create a movie of the simulation:
if gen_mov == 1:
    LanderMovie(x_arr,y_arr,ang_arr,u0_arr,u1_arr,t_arr,Xref,meta_data)
