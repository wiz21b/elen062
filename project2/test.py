import random
from scipy.integrate import dblquad
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Evaluating the integral for epexctation of misclassification

rho = 0.75
det = 1 - rho*rho
k = 0.5 * (1/(2*pi)) * (1 / sqrt(det))

def integrand(x0, x1):
    return k*np.exp(-0.5 * (1/(rho*rho - 1)) * (-x0**2 - x1**2 + 2*x0*x1*(-rho)))

# Times four because we just compute it over one quadrant
i, err_on_i = 4 * dblquad(
    integrand,
    0, float('inf'),
    0, float('inf'))


print("integral's value = {}".format(i))


# -----------------------------------------------------------------------------
# Empiracally counting misclassification

NP = 10000

s1 = []
s2 = []
for y in np.random.randint(0, 2, size=NP):
    if y == 0:
        cov = np.array([ [1, +rho], [+rho, 1]  ])
        k = np.random.multivariate_normal([0,0], cov, 1)[0]
        s1.append( (k[0],k[1]) )
    else:
        cov = np.array([ [1, -rho], [-rho, 1]  ])
        k = np.random.multivariate_normal([0,0], cov, 1)[0]
        s2.append( (k[0],k[1]) )


s1 = np.array(s1)
s2 = np.array(s2)

# cov = np.array([ [1, +rho], [+rho, 1]  ])
# s1 = np.random.multivariate_normal([0,0], cov, NP)
# cov = np.array([ [1, -rho], [-rho, 1]  ])
# s2 = np.random.multivariate_normal([0,0], cov, NP)


s1_in_q1 = 0
for x,y in s1:
    if x >= 0 and y >= 0:
        s1_in_q1 += 1

s2_in_q2 = 0
for x,y in s2:
    if x >= 0 and y >= 0:
        s2_in_q2 += 1


misclassification_rate = s2_in_q2 / (s2_in_q2 + s1_in_q1)

print("proportion of misclassified as class -1 in quadrant : {}".format(misclassification_rate))
print( 4*i/misclassification_rate)

# Drawing

x, y = s1.T
plt.scatter(x, y, marker='.')
x, y = s2.T
plt.scatter(x, y, marker='.', label="s2")
plt.axvline(x=0)
plt.axhline(y=0)
plt.legend()
plt.show()
