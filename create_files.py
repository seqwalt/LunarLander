import os
import numpy as np

# Generate .dat file for nonlinear optimization
# parameters use by pyomo
def GenerateDAT(h_list, filename):
    N = len(h_list)

    if os.path.exists('param_files/'+filename):
        os.remove('param_files/'+filename)

    f = open('param_files/'+filename, 'w')
    f.write('param N := '+str(N)+' ;\n\n')
    f.write('param h :=\n')
    for i in range(1,N+1):
        f.write(str(i)+' '+str(h_list[i-1])+'\n')
    f.write(';')

# Generate trajectory data csv and readme
def TrajFiles(t_arr,x_arr,y_arr,ang_arr,vx_arr,vy_arr,omega_arr,\
              u0_arr,u1_arr,Const,X0,Xref,U0,Ubound,T,h):
    print('Generating files...')

    # Find next file number to use
    max_filenum = -1
    files_list = os.listdir('traj_data')

    if len(files_list)==0:
        str_num='00'
    else:
        for filestr in files_list:
            num = int(filestr[-6:-4])
            if num > max_filenum:
                max_filenum = num
        if max_filenum < 9:
            str_num = '0'+str(max_filenum + 1) # leading zero
        else:
            str_num = str(max_filenum + 1)

    # Generate trajectory csv files. Rows correspons to time instances,
    # and columns are of the forms:
    # {times, x, y, ang, thrust (i.e. u0), torque (i.e. u1)}
    csvname = 'trajdata_'+str_num+'.csv'
    print('   '+csvname)
    if os.path.exists('traj_data/'+csvname):
        os.remove('traj_data/'+csvname)
    f = open('traj_data/'+csvname, 'w')
    f.write('t,x,y,ang,vx,vy,omega,thrust,torque\n')

    N = len(t_arr)
    for i in range(N):
        f.write(str(t_arr[i])+','+str(x_arr[i])+','+str(y_arr[i])+','+str(ang_arr[i])+','\
        +str(vx_arr[i])+','+str(vy_arr[i])+','+str(omega_arr[i])+','+str(u0_arr[i])+','+str(u1_arr[i])+'\n')
    f.close()

    # Generate trajectory readme with meta data
    txtname = 'readme_'+str_num+'.txt'
    print('     '+txtname)
    print('')
    f = open('traj_data/'+txtname, 'w')
    # Const,X0,Xref,U0,Ubound,T,h
    f.write('Readme for information about the simulation corresponding to '+csvname+'. \n')
    f.write('The trajectory is '+ str(T)+' seconds long with a '+str(h)+' second sampling \n')
    f.write('period, resulting in '+str(N)+' data points.\n\n')
    f.write('Initial x, y, angle, dx, dy, omega:\n')
    f.write(str(X0.T)+'\n\n')
    f.write('Final x, y, angle, dx, dy, omega:\n')
    f.write(str(Xref.T)+'\n\n')
    f.write('Inital thrust, torque:\n')
    f.write(str(U0.T)+'\n\n')
    f.write('Bound on thrust, torque:\n')
    f.write('[['+str(Ubound[0][0])+'  +/-'+str(Ubound[1][0])+']]'+'\n\n')
    f.write('Constants b_v, b_omega, mass, gravity, rotational inetia:\n')
    f.write(str(Const)+'\n')
    f.close()
