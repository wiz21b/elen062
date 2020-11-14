import math
import random
from scipy.integrate import dblquad
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 2.c Bayes model

def truth(x, sigma_squared):
    # In numpy, scale is the standard deviation,
    # we're given the square of that.
    return x + np.random.normal(0, math.sqrt(sigma_squared))


x0 = 10
N = 1000
sigma_squared = 0.1

samples = np.array([truth(x0, sigma_squared) for i in range(N)])
bayes = np.ones((N,)) * x0

experimental_residual_error = np.sum((samples - bayes)**2) / N

print("experimental residual error = {:.3f}, expected = {:.3f}".format(
    experimental_residual_error, sigma_squared))


## 2.d

def bayes(x):
    return -x**3 + 3*x**2 - 2*x + 1

def f(x):
    # From the statement : Let us now assume that
    # f(x) =−x3+3x2−2x+1 and σ2= 0.1.

    # In numpy, scale is the standard deviation,
    # we're given the square of that.
    return bayes(x) + np.random.normal(0, math.sqrt(0.1))


def make_n_learning_set( n, s):
    # Make n learning sets of s elements each

    # Statement : Estimate and plot the following quantities
    # for x ∈ [0,2] and using learning samples of size N= 30

    ls = []
    for i in range(n):
        xs = np.random.uniform(0, 2, s)
        ys = np.array([f(x) for x in xs])
        ls.append((ys, xs))

    return ls


def train_models(learning_sets, learning_algorithm):
    # Use the algorithm number learning_algorithm to train models over
    # the learning set.
    # We create as many models as learning sets.

    # The learning algorithm is a number in [0-5], that's
    # the m number in the problem statement.

    assert 0 <= learning_algorithm <= 5

    models = []
    for ys, xs in learning_sets:

        # A model is a0 + a1 * x + a2 * x² + ...
        # range will go from 0 to m inclusive !

        # For the training, we use a0 + a1 * x_i + a2 * x_i² + ...
        powers_of_x = np.stack([xs**i for i in range(0, learning_algorithm+1)])
        model = LinearRegression()
        reg = model.fit(powers_of_x.T, ys)

        models.append(reg)

    return models


def squared_bias_of_models_at_x(learning_algorithm, models, x, bayes_value):
    powers_of_x = np.array([ [x**i for i in range(0, learning_algorithm+1)] ])

    # Ask each model to predict its value of y
    predictions = []
    for model in models:
        predictions.append(model.predict(powers_of_x))

    return (bayes_value - np.average(np.array(predictions)))**2

def variance_of_models_at_x(learning_algorithm, models, x, bayes_value):
    powers_of_x = np.array([ [x**i for i in range(0, learning_algorithm+1)] ])

    predictions = []
    for model in models:
        predictions.append(model.predict(powers_of_x))
    avg_pred = np.average(predictions)

    diff = []
    for model in models:
        diff = (model.predict(powers_of_x) - avg_pred)**2

    return np.average(np.array(diff))


biases = []
for learning_algorithm in range(0, 5+1):
    learning_sets = make_n_learning_set(n=20, s=30)
    models = train_models(learning_sets, learning_algorithm)
    bias = []
    for x0 in np.arange(0, 2, 0.01):
        #bias.append(squared_bias_of_models_at_x(learning_algorithm, models, x0, bayes(x0)))
        bias.append(variance_of_models_at_x(learning_algorithm, models, x0, bayes(x0)))

    biases.append(bias)

for algo, bias in enumerate(biases):
    plt.scatter(np.arange(0, 2, 0.01), bias, marker='.', linewidths=0, label=f"m={algo}")

for h in [0, 0.5, 1, 1.75]:
    plt.axvline(x=h,c='black')
# plt.axvline(x=0.5)
# plt.axvline(x=1)
# plt.axvline(x=1.75)
plt.title("Bias")
plt.legend()
plt.show()
exit()


xs = np.random.uniform(0,2,N)
ys = np.array([f(x) for x in xs] )
plt.scatter(xs, ys, marker='.')

N = 30
for M in range(0,5+1):

    powers_of_x = np.stack( [xs**i for i in range(0,M+1)] )
    model = LinearRegression()
    reg = model.fit(powers_of_x.T, ys)

    prediction = reg.predict(powers_of_x.T)

    plt.scatter(xs, prediction, marker='.')

plt.show()



exit()


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
