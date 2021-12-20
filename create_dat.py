import os
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
