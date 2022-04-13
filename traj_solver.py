from __future__ import division
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
from create_files import GenerateDAT

def SolveTrajectory(INIT_STATE, REF_STATE, U_BOUND, CONSTANTS, T, numColl, op_sys):
    print()
    print('Solving optimal trajectory...')
    print()
    x0,y0,ang0,vx0,vy0,omega0 = INIT_STATE
    x0=float(x0); y0=float(y0); ang0=float(ang0)
    vx0=float(vx0); vy0=float(vy0); omega0=float(omega0)

    xN,yN,angN,vxN,vyN,omegaN = REF_STATE
    xN=float(xN); yN=float(yN); angN=float(angN)
    vxN=float(vxN); vyN=float(vyN); omegaN=float(omegaN)

    max_thrust, max_torque = U_BOUND
    max_thrust=float(max_thrust); max_torque=float(max_torque)

    bv,bo,m,g,rotI = CONSTANTS
    bv=float(bv); bo=float(bo); m=float(m)
    g=float(g); rotI=float(rotI);

    # create an abstact model instance
    model = pyo.AbstractModel()

    # create index parameters
    model.N = pyo.Param(within=pyo.NonNegativeIntegers)
    # index value range (sets of possible indices)
    model.I = pyo.RangeSet(0,model.N)
    model.J = pyo.RangeSet(1,model.N)
    # # Create h_k param
    model.h = pyo.Param(model.J,domain=pyo.NonNegativeReals)
    # # Create state decision variables
    model.x = pyo.Var(model.I,domain=pyo.Reals)
    model.y = pyo.Var(model.I,domain=pyo.NonNegativeReals)
    model.ang = pyo.Var(model.I,domain=pyo.Reals)
    model.vx = pyo.Var(model.I,domain=pyo.Reals)
    model.vy = pyo.Var(model.I,domain=pyo.Reals)
    model.omega = pyo.Var(model.I,domain=pyo.Reals)
    # Create control decision variables
    model.thrust = pyo.Var(model.I,bounds=(0,max_thrust))
    model.torque = pyo.Var(model.I,bounds=(-max_torque,max_torque))

    # Create objective
    def obj_expression(M):
        return sum(M.thrust[i]**2 + M.thrust[i+1]**2 + \
                   M.torque[i]**2 + M.torque[i+1]**2 \
                   for i in range(len(M.I)-1))
    model.OBJ = pyo.Objective(rule=obj_expression)

    ## ----- Create Dynamics Constraints ----- ##

    def x_constraint(M, i):
        return M.x[i] - M.x[i-1] == .5*M.h[i]*(M.vx[i] + M.vx[i-1])
    model.xConstraint = pyo.Constraint(model.J, rule=x_constraint)

    def y_constraint(M, i):
        return M.y[i] - M.y[i-1] == .5*M.h[i]*(M.vy[i] + M.vy[i-1])
    model.yConstraint = pyo.Constraint(model.J, rule=y_constraint)

    def ang_constraint(M, i):
        return M.ang[i] - M.ang[i-1] == .5*M.h[i]*(M.omega[i] + M.omega[i-1])
    model.angConstraint = pyo.Constraint(model.J, rule=ang_constraint)

    def vx_constraint(M, i):
        return M.vx[i] - M.vx[i-1] == \
            .5*M.h[i]*( (1/m)*(M.thrust[i]*pyo.sin(-M.ang[i]) + \
            M.thrust[i-1]*pyo.sin(-M.ang[i-1])) - (bv/m)*(M.vx[i] + M.vx[i-1]) )
    model.vxConstraint = pyo.Constraint(model.J, rule=vx_constraint)

    def vy_constraint(M, i):
        return M.vy[i] - M.vy[i-1] == \
            .5*M.h[i]*( -2*g + (1/m)*(M.thrust[i]*pyo.cos(M.ang[i]) + \
            M.thrust[i-1]*pyo.cos(M.ang[i-1])) - (bv/m)*(M.vy[i] + M.vy[i-1]) )
    model.vyConstraint = pyo.Constraint(model.J, rule=vy_constraint)

    def omega_constraint(M, i):
        return M.omega[i] - M.omega[i-1] == \
            .5*M.h[i]*( (1/rotI)*(M.torque[i] + M.torque[i-1]) - \
            (bo/rotI)*(M.omega[i] + M.omega[i-1]) )
    model.omegaConstraint = pyo.Constraint(model.J, rule=omega_constraint)

    ## ------------------------------------ ##

    ## ----- Create State Constraints ----- ##

    def legs_constraint(M, i): # legs always above ground
        return M.y[i] >= abs(.78*pyo.sin(M.ang[i]))
    model.legsConstraint = pyo.Constraint(model.I, rule=legs_constraint)

    ## ------------------------------------ ##

    # create dat file
    data_file_name = 'traj_info.dat'
    h_list = (float(T)/float(numColl))*np.ones((numColl))
    GenerateDAT(h_list, data_file_name)
    # create instance from dat file
    instance = model.create_instance('param_files/'+data_file_name)

    ## ----- Create Inital Condtion Constraints ----- ##

    instance.x[0].fix(x0)
    instance.y[0].fix(y0)
    instance.ang[0].fix(ang0)
    instance.vx[0].fix(vx0)
    instance.vy[0].fix(vy0)
    instance.omega[0].fix(omega0)

    #instance.ang[int(instance.N/2)].fix(np.pi)

    ## --------------------------------- ##

    ## ----- Create Final Condtion Constraints ----- ##

    instance.x[instance.N].fix(xN)
    instance.y[instance.N].fix(yN)
    instance.ang[instance.N].fix(angN)
    instance.vx[instance.N].fix(vxN)
    instance.vy[instance.N].fix(vyN)
    instance.omega[instance.N].fix(omegaN)
    instance.thrust[instance.N].fix(m*g)  # hovering
    instance.torque[instance.N].fix(0)    # no torque

    ## -------------------------------------------- ##

    # Compute a solution using ipopt for nonlinear optimization
    if op_sys == 'linux':
        solver_exe='~/Software/ipopt_binary/ipopt'
        results = pyo.SolverFactory('ipopt',executable=solver_exe).solve(instance)
    elif op_sys == 'mac':
        results = pyo.SolverFactory('ipopt').solve(instance)

    # Check for infeasability
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('Feasible solution found')
        feasible = True
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('No feasable solution found')
        feasible = False
    else:
        # Something else is wrong
        print('Solver Status: ',  result.solver.status)

    #instance.pprint()
    STATES = np.zeros((len(instance.x),6))
    CONTROLS = np.zeros((len(instance.x),2))
    TIMES = np.zeros((len(instance.x),1))
    time_prev = 0
    for i in range(len(instance.x)):
        STATES[i,0] = instance.x[i].value
        STATES[i,1] = instance.y[i].value
        STATES[i,2] = instance.ang[i].value
        STATES[i,3] = instance.vx[i].value
        STATES[i,4] = instance.vy[i].value
        STATES[i,5] = instance.omega[i].value
        CONTROLS[i,0] = instance.thrust[i].value
        CONTROLS[i,1] = instance.torque[i].value
        if i>0:
            TIMES[i] = time_prev + instance.h[i]
        else:
            TIMES[i] = 0
        time_prev = TIMES[i]
    #print(np.hstack((STATES, CONTROLS, TIMES)))
    OPT_TRAJ = np.hstack((STATES, CONTROLS, TIMES))
    return {'feasible':feasible,'traj':OPT_TRAJ}
