import math
import random
from scipy.integrate import dblquad
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge


def bayes(x):
    return -x**3 + 3*x**2 - 2*x + 1


def truth(x, sigma_squared):
    # In numpy, scale is the standard deviation,
    # we're given the square of that.
    return bayes(x) + np.random.normal(0, math.sqrt(sigma_squared))


x0 = 10
N = 1000
sigma_squared = 0.1

samples = np.array([truth(x0, sigma_squared) for i in range(N)])
bayes_values = np.ones((N,)) * bayes(x0)

experimental_residual_error = np.sum((samples - bayes_values)**2) / N

print("experimental residual error = {:.3f}, expected = {:.3f}".format(
    experimental_residual_error, sigma_squared))


# 2.c Bayes model


# Plotting Bayes model

plt.figure(10)
rng = np.arange(-2,+4,0.1)
plt.plot(rng, [bayes(x) for x in rng])
#plt.plot(rng, [truth(x, sigma_squared) for x in rng])
plt.title("Bayes model")
plt.xlabel("x")
plt.savefig("q2c_bayes.pdf")
#plt.show()


plt.figure(11)
rng = np.arange(-2,+4,0.1)
errors = []
for x0 in rng:
    samples = np.array([truth(x0, sigma_squared) for i in range(N)])
    bayes_values = np.ones((N,)) * bayes(x0)
    experimental_residual_error = np.sum((samples - bayes_values)**2) / N
    errors.append(experimental_residual_error)

plt.plot(rng, errors)
plt.ylim(0, sigma_squared * 1.5)
plt.title("Experimental error")
plt.xlabel("x")
plt.ylabel("Experimental error")
plt.axhline(y=sigma_squared, c='black', linestyle='--', label="σ²")
plt.legend()
plt.savefig("q2c_error.pdf")
#plt.show()


## 2.d


def f(x):
    # From the statement : Let us now assume that
    # f(x) =−x3+3x2−2x+1 and σ2= 0.1.

    # In numpy, scale is the standard deviation,
    # we're given the square of that.
    return bayes(x) + np.random.normal(0, math.sqrt(0.1))


def make_learning_sets( nb_learning_sets):
    # Make s learning sets of n elements each

    # Statement : Estimate and plot the following quantities
    # for x ∈ [0,2] and using learning samples of size N=30

    ls = []
    for i in range(nb_learning_sets):
        xs = np.random.uniform(0, 2, 30)
        ys = np.array([f(x) for x in xs])
        ls.append((ys, xs))

    return ls


def train_models(learning_sets, learning_algorithm):
    # Use the algorithm number learning_algorithm to train models over
    # the learning sets.
    # We create as many models as learning sets.

    # The learning algorithm is a number in [0-5], that's
    # the m number in the problem statement.

    assert 0 <= learning_algorithm <= 5

    models = []
    for ys, xs in learning_sets:

        # A model is a0 + a1 * x + a2 * x² + ...
        # range will go from 0 to m inclusive !

        # For the training, we use a0*x_i⁰ + a1 * x_i¹ + a2 * x_i² + ...
        powers_of_x = np.stack([xs**i for i in range(0, learning_algorithm+1)])

        model = LinearRegression(fit_intercept=False)
        reg = model.fit(powers_of_x.T, ys)

        models.append(reg)


    return models


def squared_bias_of_models_at_x(learning_algorithm, models, x, bayes_value):
    # powers of x are needed to use the linear regression results
    powers_of_x = np.array([[x**i for i in range(0, learning_algorithm+1)] ])

    # Ask each model to predict its value of y
    predictions = []
    for model in models:
        predictions.append(model.predict(powers_of_x))

    return (bayes_value - np.average(np.array(predictions)))**2


def variance_of_models_at_x(learning_algorithm, models, x, bayes_value):
    powers_of_x = np.array([ [x**i for i in range(0, learning_algorithm+1)] ])

    # Ask each model to predict its value of y at point x
    predictions = []
    for model in models:
        predictions.append(model.predict(powers_of_x))
    predictions = np.array(predictions)

    # variance is squared standard dev.
    return np.average((predictions - np.average(predictions))**2)


biases = []
variances = []
expected_errors = []

NB_LEARNING_SETS = 400 # seem stable 200
X_RANGE = np.linspace(0,2,100)

for learning_algorithm in range(0, 5+1):
    print(f"training for algo m={learning_algorithm}")
    learning_sets = make_learning_sets(NB_LEARNING_SETS)
    models = train_models(learning_sets, learning_algorithm)
    bias = []
    variance = []
    for x0 in X_RANGE:
        bias.append(squared_bias_of_models_at_x(
            learning_algorithm, models, x0, bayes(x0)))
        variance.append(variance_of_models_at_x(
            learning_algorithm, models, x0, bayes(x0)))

    biases.append(bias)
    variances.append(variance)
    ee = np.array(bias) + np.array(variance) + experimental_residual_error
    expected_errors.append(ee)


# Draw biases plot

plt.figure(20)
for algo, bias in enumerate(biases):
    plt.scatter(X_RANGE, bias, marker='.', linewidths=0, label=f"m={algo}")

for h in [0, 0.5, 1, 1.75]:
    plt.axvline(x=h,c='black')
plt.title("Bias")
plt.xlabel("x")
plt.legend()
plt.savefig("bias_d.pdf")
#plt.show()

# Draw variances plot

plt.figure(21)
for algo, variance in enumerate(variances):
    plt.scatter(X_RANGE, variance, marker='.', linewidths=0, label=f"m={algo}")

for h in [0, 0.5, 1, 1.75]:
    plt.axvline(x=h,c='black')
plt.title("Variance")
plt.ylim(0, sigma_squared * 1.1)
plt.xlabel("x")
plt.legend()
plt.savefig("var_d.pdf")

# Draw expected error plot

plt.figure(22)
for algo, expected_error in enumerate(expected_errors):
    plt.scatter(X_RANGE, expected_error, marker='.', linewidths=0, label=f"m={algo}")

for h in [0, 0.5, 1, 1.75]:
    plt.axvline(x=h,c='black')
plt.title("Expected error")
plt.ylim(0, 0.4)
plt.xlabel("x")
plt.legend()
plt.savefig("exp_err_d.pdf")

# Scatter plot bias versus variance

plt.figure(25)
for algo in range(0,5+1):
    v = np.mean(variances[algo])
    b = np.mean(biases[algo])

    plt.scatter(v,b,label=f"m={algo}")
plt.xlabel("variance")
plt.ylabel("bias")
plt.title("Bias/Variance averaged on all x")
plt.legend()
plt.savefig("q2d_bias_variance.pdf")


# Mean of expected error

plt.figure(27)
for algo in range(0,5+1):
    # Draw one at a time to have the same colors as in other graphics
    plt.scatter([algo],
                [np.mean(expected_errors[algo])],
                label=f"m={algo}")
plt.xlabel("m")
plt.ylabel("Expected error average")
plt.title("Expected error averaged on all x")
plt.legend()
plt.savefig("q2e_exp_error_avg.pdf")

######################################################################
# # Start over with ridge regression

def train_models_ridge(learning_sets, learning_algorithm, lambda_):
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

        model = Ridge(alpha=lambda_,fit_intercept=False)  # scipy has other naming convention
        reg = model.fit(powers_of_x.T, ys)

        models.append(reg)

    return models


biases = []
variances = []
expected_errors = []

lambdas = [x for x in np.linspace(0, 2, 5)]
for lambda_ in lambdas:

    print(f"Training on lambda = {lambda_}")
    learning_algorithm = 5 # per problem statement
    learning_sets = make_learning_sets(NB_LEARNING_SETS)
    models = train_models_ridge(learning_sets, learning_algorithm, lambda_)
    bias = []
    variance = []
    for x0 in X_RANGE:
        bias.append(squared_bias_of_models_at_x(
            learning_algorithm, models, x0, bayes(x0)))
        variance.append(variance_of_models_at_x(
            learning_algorithm, models, x0, bayes(x0)))

    biases.append(bias)
    variances.append(variance)
    ee = np.array(bias) + np.array(variance) + experimental_residual_error
    expected_errors.append(ee)


plt.figure(123)
for algo, variance in enumerate(variances):
    plt.scatter(X_RANGE, variance, marker='.', linewidths=0, label=f"m=5, lambda={lambdas[algo]}")
plt.ylim(0, 0.1)
plt.title("Variance - Ridge regression")
plt.xlabel("x")
plt.legend()
plt.savefig("q2f_variance.pdf")


plt.figure(124)
for algo, bias in enumerate(biases):
    plt.scatter(X_RANGE, bias, marker='.', linewidths=0, label=f"m=5, lambda={lambdas[algo]:.1f}")
plt.ylim(0, 0.1)
plt.title("Bias - Ridge regression")
plt.xlabel("x")
plt.ylim(0, 0.02)
plt.legend()
plt.savefig("q2f_bias.pdf")


# Draw expected error plot

plt.figure(125)
for algo, expected_error in enumerate(expected_errors):
    plt.scatter(X_RANGE, expected_error, marker='.', linewidths=0, label=f"m=5, lambda={lambdas[algo]:.1f}")

for h in [0, 0.5, 1, 1.75]:
    plt.axvline(x=h,c='black')
plt.title("Expected error - Ridge regression")
plt.ylim(0, 0.4)
plt.xlabel("x")
plt.legend()
plt.savefig("q2f_exp_err.pdf")


# Scatter plot bias versus variance

plt.figure(130)
for algo in range(0,len(lambdas)):
    v = np.mean(variances[algo])
    b = np.mean(biases[algo])

    plt.scatter(v,b,label=f"m=5, lambda={lambdas[algo]:.1f}")
plt.xlabel("variance")
plt.ylabel("bias")
plt.title("Bias/Variance averaged on all x - Ridge regression")
plt.legend()
plt.savefig("q2f_bias_variance.pdf")


plt.show()
