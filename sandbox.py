import numpy as np
from scipy.special import roots_legendre

def f(x):
    return np.exp(-x**2)*np.cos(x)
def g(x,a,b):
    val = 0.5*(b-a)*f( 0.5*(b-a)*x + 0.5*(a+b) )
    return val

# integrate f from -1 to 1
n = 2
x,w = roots_legendre(n)
x = x.reshape(-1,1)
w = w.reshape(-1,1)
print(x)
integral_approx = w.T@f(x)
print(float(integral_approx))

# integrate f from -1 to 1 in chunks
integral_approx = 0
list = np.linspace(-1,1,10)
for i in range(len(list)-1):
    a = list[i]; b = list[i+1]
    integral_approx += w.T@g(x,a,b)
print(float(integral_approx))
