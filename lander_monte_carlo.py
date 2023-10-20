import os
import time
import platform
from scipy.interpolate import interp1d
import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
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
    x = STATE[0,:]
    y = STATE[1,:]
    ang = STATE[2,:]
    vx = STATE[3,:]
    vy = STATE[4,:]
    omega = STATE[5,:]
    thrust = CONTROL[0,:]
    torque = CONTROL[1,:]
    bv = CONSTANTS[0,:]
    bo = CONSTANTS[1,:]
    m = CONSTANTS[2,:]
    g = CONSTANTS[3,:]
    rotI = CONSTANTS[4,:]

    f_x  = vx; f_y = vy; f_ang = omega  # velocities
    f_vx = -bv*vx/m + thrust*sin(-ang)/m    # accelerations
    f_vy = -(g + bv*vy/m) + thrust*cos(ang)/m
    f_omega  = -bo*omega/rotI + torque/rotI

    f_x  = f_x.reshape(1,-1); f_y = f_y.reshape(1,-1); f_ang = f_ang.reshape(1,-1)
    f_vx = f_vx.reshape(1,-1); f_vy = f_vy.reshape(1,-1); f_omega  = f_omega.reshape(1,-1)

    vectors = np.vstack((f_x, f_y, f_ang, f_vx, f_vy, f_omega))
    return vectors

if platform.system() == "Linux":
    OP_SYS = "linux"
elif platform.system() == "Darwin":
    OP_SYS = "mac"
# meta data
op_sys = OP_SYS
monte_carlo_for_loop = 0 # do for-loop monte carlo sim to compare speed to vectorized? 1=yes, 0=no
closed_loop = 1  # apply heuristic closed loop controller in addition to open-loop optimal control values 1=yes, 0=no
num_mc_sims = 1000 # number of Monte Carlo simulations
mc_uncert_scl = 1.0 # scale the monte carlo uncertainties (1 is nominal)
traj_data_generate = 0 # generate trajectory csv and readme? 1=yes, 0=no
make_plots = 1   # generate trajectory and control plots? 1=yes, 0=no
rand_BC = 0      # Random boundary conditions? 1=yes, 0=no

# Constants
bv = 5; bo = 11 # b_v and b_{\omega}
g = 2.0; m = 10.0 # gravity and mass
rotI = (13/12)*m
Const = np.array((bv,bo,m,g,rotI)).reshape(-1,1) # order matters with these consts

# Set boundary conditions and final time
if rand_BC != 1:
    # Not random boundary conditions
    # Total time in seconds
    T = 5

    # Initial State
    x0 = -1; y0 = 0; ang0 = 0;
    vx0 = 0; vy0 = 0; omega0 = 0;
    X0 = np.array(([x0],[y0],[ang0],[vx0],[vy0],[omega0])) # initial state
    X = X0

    # Reference tracking state
    xref = 1; yref = 0; angref = 0;
    vxref = 0; vyref = 0; omegaref = 0;
    Xref = np.array(([xref],[yref],[angref],\
        [vxref],[vyref],[omegaref])) # reference state
else:
    # Random boundary conditions
    rand_init = lambda : 2*(np.random.rand(6,1) - 0.5) # vals from -1 to 1
    range_vals = np.array(([-5,5],[0,10],[-2*np.pi,2*np.pi],[-3,3],[-3,3],[-2*np.pi,2*np.pi])) # row 1: x range, row 2: y range etc.
    # range_vals = np.array(([-5,5],[0,10],[-2*np.pi,2*np.pi],[-3,3],[-3,3],[-2*np.pi,2*np.pi])) # row 1: x range, row 2: y range etc.
    centers = np.mean(range_vals,1).reshape(-1,1)
    radii = 0.5*(range_vals[:,1] - range_vals[:,0]).reshape(-1,1)

    X0 = radii*rand_init () + centers # random initial conditions
    X = X0
    Xref = radii*rand_init () + centers # random final conditions

    T_range = np.array(([5,10]))
    T = np.random.rand()*(T_range[1] - T_range[0]) + T_range[0] # random final time

# Time step size
h = 0.0025
step_sizes = np.array(([h]))

# Initial Control
max_thrust = 5000
max_torque = 200
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

# Monte Carlo Runge-Kutta 4 simulation function
# Using numpy vectorization, each step of the RK4 advances all Monte Carlo simulations
def MC_RK4_sim(X_init, Const_true, num_sims):
    # Initialize data arrays
    x_arr = np.zeros((int(T/h)+1, num_sims))
    y_arr = np.zeros((int(T/h)+1, num_sims))
    ang_arr = np.zeros((int(T/h)+1, num_sims))
    vx_arr = np.zeros((int(T/h)+1, num_sims))
    vy_arr = np.zeros((int(T/h)+1, num_sims))
    omega_arr = np.zeros((int(T/h)+1, num_sims))
    u0_arr = np.zeros((int(T/h)+1, num_sims))  # thrust array
    u1_arr = np.zeros((int(T/h)+1, num_sims))  # torque array
    t_arr = np.zeros((int(T/h)+1))

    K_init = np.zeros((num_sims, 2, 6))
    K_init[:,1,2] = -262.5*np.ones(num_sims)
    K_init[:,1,5] = -140*np.ones(num_sims)

    # Lambdas/Functions
    tensor_ones = lambda A : np.ones((num_sims, A.shape[0], A.shape[1]))
    def tensor2matrix(A_tensor):
        # input shape: (m x n x 1), output shape: (n x m)
        assert(A_tensor.shape[2] == 1)
        return A_tensor.reshape(A_tensor.shape[0],A_tensor.shape[1]).T
    #tensor2matrix = lambda A_tensor : A_tensor.reshape(A_tensor.shape[0],A_tensor.shape[1]).T   # input shape: (m x n x 1), output shape: (n x m)
    matrix2tensor = lambda A_matrix : A_matrix.T.reshape(A_matrix.shape[1],A_matrix.shape[0],1) # input shape: (n x m), output shape: (m x n x 1)

    # Init tensor state matrix
    X_init = X_init.reshape(-1,1)
    X_tensor = X_init * tensor_ones(X_init) # X_tensor.shape = (num_sims, 6, 1)

    # Inject noise
    rand_init = lambda : 2*(np.random.rand(5,num_sims) - 0.5) # vals from -1 to 1
    noise_range = mc_uncert_scl*np.array((bv*0.05, bo*0.05, m*0.02, g*0.02, rotI*0.02)).reshape(-1,1)
    noise = noise_range*rand_init()
    Const_ = Const_true * np.ones((5,num_sims)) + noise

    for j in range(int(T/h)):
        U_opt = U_interp[j,:].reshape(-1,1)
        U_opt = U_opt * tensor_ones(U_opt) # U_opt.shape = (num_sims, 2, 1)
        K = K_init
        if closed_loop == 1:
            # closed-loop controller
            temp = 20*cos(X_tensor[:,2,0])
            K[:,0,1] = -100*temp
            K[:,0,4] = -20*temp
            K[:,1,0] = 10*temp
            K[:,1,3] = 10*temp
            Xref_new = X_interp[j,:].reshape(-1,1)
            U_tensor = K@(X_tensor-Xref_new) + U_opt # U_tensor.shape = (num_sims, 2, 1)
        else:
            U_tensor = U_opt

        X_ = tensor2matrix(X_tensor)
        U = tensor2matrix(U_tensor)

        U[0,:] = np.clip(U[0,:],0,max_thrust) # constrain thrust
        U[1,:] = np.clip(U[1,:],-max_torque,max_torque) # constrain torque

        # Store data
        x_arr[j,:] = X_[0,:]
        y_arr[j,:] = X_[1,:]
        ang_arr[j,:] = X_[2,:]
        vx_arr[j,:] = X_[3,:]
        vy_arr[j,:] = X_[4,:]
        omega_arr[j,:] = X_[5,:]
        u0_arr[j,:] = U[0,:]
        u1_arr[j,:] = U[1,:]
        t_arr[j] = h*j

        # RK-4
        k1 = F(X_,U,Const_)
        k2 = F(X_+h/2*k1,U,Const_)
        k3 = F(X_+h/2*k2,U,Const_)
        k4 = F(X_+h*k3,U,Const_)
        k = (k1+2*k2+2*k3+k4)/6
        X_ = X_ + h*k
        X_tensor = matrix2tensor(X_)

    # Store final time-step data
    x_arr[j+1,:] = X_[0,:]
    y_arr[j+1,:] = X_[1,:]
    ang_arr[j+1,:] = X_[2,:]
    vx_arr[j+1,:] = X_[3,:]
    vy_arr[j+1,:] = X_[4,:]
    omega_arr[j+1,:] = X_[5,:]
    u0_arr[j+1,:] = U[0,:]  # Thrust
    u1_arr[j+1,:] = U[1,:]  # Torque
    t_arr[j+1] = h*(j+1)

    return x_arr, y_arr, ang_arr, vx_arr, vy_arr, omega_arr, u0_arr, u1_arr, t_arr


# Run Monte Carlo simulation (vectorized)
print('Running simulations with vectorization...')
start = time.time()
x_arr, y_arr, ang_arr, vx_arr, vy_arr, omega_arr, u0_arr, u1_arr, t_arr = MC_RK4_sim(X, Const, num_mc_sims)
end = time.time()
print('Finished '+ str(num_mc_sims) +' simulations in ' + "{:.3f}".format(end - start) + ' seconds')
print()
# Processing
# Note: x_arr.shape = y_arr.shape = (num times, num_mc_sims)
# TODO compute standard deviation at each time step and plot over a x_err/y_err.
x_err = x_interp(t_arr[0:-1]).reshape(-1,1) - x_arr[0:-1,:]
# x_sdv =
y_err = y_interp(t_arr[0:-1]).reshape(-1,1) - y_arr[0:-1,:]

# Run Monte Carlo simulation (for loop)
# for time-comparison only (no stored data)
if monte_carlo_for_loop == 1:
    # Injected noise
    rand_init = lambda : 2*(np.random.rand(5,1) - 0.5) # vals from -1 to 1
    noise_range = np.array((bv*0.05, bo*0.02, m*0.02, g*0.02, rotI*0.02)).reshape(-1,1)
    print('Running simulations with for loop...')
    start = time.time()
    for i in range(0, num_mc_sims):
        noise = noise_range*rand_init()
        Const_n = Const + noise
        # RK4
        X_ = X
        for j in range(int(T/h)):
            U_opt = U_interp[j,:].reshape(-1,1)
            if closed_loop == 1:
                temp = 20*cos(X_[2][0])
                K = np.array(([0, -100*temp, 0, 0, -20*temp, 0], [10*temp, 0, -262.5, 10*temp, 0, -140]))
                Xref_new = X_interp[j,:].reshape(-1,1)
                U = K@(X_-Xref_new) + U_opt
            else:
                U = U_opt
            U[0] = np.clip(U[0],0,max_thrust) # constrain thrust
            U[1] = np.clip(U[1],-max_torque,max_torque) # constrain torque
            # RK-4
            k1 = F(X_,U,Const_n)
            k2 = F(X_+h/2*k1,U,Const_n)
            k3 = F(X_+h/2*k2,U,Const_n)
            k4 = F(X_+h*k3,U,Const_n)
            k = (k1+2*k2+2*k3+k4)/6
            X_ = X_ + h*k
    end = time.time()
    print('Finished '+ str(num_mc_sims) +' simulations in ' + "{:.3f}".format(end - start) + ' seconds')
    print()

# Plot using data from vectorized Monte Carlo sims
if make_plots == 1:
    # Suppress GTK warning outputs
    # CAUTION: all error messages suppressed
    #'''
    # Comment out for debugging
    fd = os.open('/dev/null',os.O_WRONLY)
    os.dup2(fd,2)
    #'''

    print('Creating plots...')
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

    sim_alpha = 0.05
    opt_line = 'y:'
    sim_line = 'k'
    strt_mrkr = 'bo'
    opt_wid = 3

    # Position
    plt.plot(x_arr, y_arr,sim_line,alpha=sim_alpha)
    plt.plot(x_arr[:,0], y_arr[:,0],sim_line,label="sim trajectories",alpha=sim_alpha)
    plt.plot(OPT_TRAJ[:,0],OPT_TRAJ[:,1],opt_line,label="opt trajectory",linewidth=opt_wid)
    plt.plot(x_arr[0,0],y_arr[0,0],strt_mrkr,label="start point")
    plt.legend(loc="lower left")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")

    # Angle
    plt.figure()
    plt.plot(t_arr,ang_arr,sim_line,alpha=sim_alpha)
    plt.plot(t_arr,ang_arr[:,0],sim_line,label="sim trajectory",alpha=sim_alpha)
    plt.plot(OPT_TRAJ[:,8],OPT_TRAJ[:,2],opt_line,label="opt trajectory",linewidth=opt_wid)
    plt.plot(t_arr[0],ang_arr[0,0],strt_mrkr,label="start point")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")

    # Error
    plt.figure()
    plt.plot(t_arr[0:-1],x_err,sim_line,alpha=sim_alpha)
    plt.xlabel("time (s)")
    plt.ylabel("x distance (m)")
    plt.title("x distance from nominal trajectory")
    plt.figure()
    plt.plot(t_arr[0:-1],y_err,sim_line,alpha=sim_alpha)
    plt.xlabel("time (s)")
    plt.ylabel("y distance (m)")
    plt.title("y distance from nominal trajectory")
    plt.show()
