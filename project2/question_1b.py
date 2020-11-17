import math
import random
from scipy.integrate import dblquad
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1B
# -----------------------------------------------------------------------------
# Evaluating the integral for expectation of misclassification
# This is done numerically first.

rho = 0.75
det = 1 - rho**2
k = 0.5 * (1/(2*pi)) * (1 / sqrt(det))

def integrand(x0, x1):
    return k*np.exp(-0.5 * (1/(rho*rho - 1)) * (-x0**2 - x1**2 + 2*x0*x1*(-rho)))

# Times four because we just compute it over one quadrant
i, err_on_i = dblquad(
    integrand,
    0, float('inf'),
    0, float('inf'))

# we just computed the integral over *one* single quadrant.
# Don't forget to multiply by 4 to have the actual expecation !

expectation_numerical = i * 4

# -----------------------------------------------------------------------------
# Empirically counting misclassification

NP = 40000

# We just construct the points according to the problem statement
s1 = []
s2 = []
for y in np.random.randint(0, 2, size=NP):
    if y == 0:
        cov = np.array([[1, +rho], [+rho, 1]])
        k = np.random.multivariate_normal([0, 0], cov, 1)[0]
        s1.append((k[0], k[1]))
    else:
        cov = np.array([[1, -rho], [-rho, 1]])
        k = np.random.multivariate_normal([0, 0], cov, 1)[0]
        s2.append((k[0], k[1]))


# we compute how much our classifier will misclassify.
# we actually compute the proportion of misclassification to the total
# of classifications.
# we compute that for one quadrant because we can more easily
# simulate our classifier's behaviour.
# computing over a quadrant doesn't change the result since
# ou proportion will be the same in any quadrant (by symmetry)
# and so in the whole plane.

s1 = np.array(s1)
s2 = np.array(s2)

# Count the good classifications
s1_in_q1 = 0
for x, y in s1:
    if x >= 0 and y >= 0:
        s1_in_q1 += 1

# Count the bad classifications
s2_in_q2 = 0
for x, y in s2:
    if x >= 0 and y >= 0:
        s2_in_q2 += 1


misclassification_rate = s2_in_q2 / (s2_in_q2 + s1_in_q1)

print("After {} trials : proportion of misclassified as class -1 in quadrant : {}".format(NP, misclassification_rate))
print("Value of expectation for misclassifications computed numerically from analytical formula = {}".format(expectation_numerical))
print("Likeliness of both these values : {}".format(misclassification_rate/expectation_numerical))

# Drawing the points, just to check it looks like what we expect

# x, y = s1.T
# plt.scatter(x, y, marker='.')
# x, y = s2.T
# plt.scatter(x, y, marker='.', label="s2")
# plt.axvline(x=0)
# plt.axhline(y=0)
# plt.legend()
# plt.show()
