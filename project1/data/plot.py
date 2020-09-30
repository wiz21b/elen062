"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def make_cmaps():
    """
    Return
    ------
    bg_map, sc_map: tuple (colormap, colormap)
        bg_map: The colormap for the background
        sc_map: Binary colormap for scatter points
    """
    top = mpl.cm.get_cmap('Oranges_r')
    bottom = mpl.cm.get_cmap('Blues')

    newcolors = np.vstack((top(np.linspace(.25, 1., 128)),
                           bottom(np.linspace(0., .75, 128))))
    bg_map = ListedColormap(newcolors, name='OrangeBlue')

    sc_map = ListedColormap(["#ff8000", "DodgerBlue"])

    return bg_map, sc_map


def plot_boundary(fname, fitted_estimator, X, y, mesh_step_size=0.1, title="", inline_plotting=False):
    """Plot estimator decision boundary and scatter points

    Parameters
    ----------
    fname : str
        File name where the figures is saved.

    fitted_estimator : a fitted estimator

    X : array, shape (n_samples, 2)
        Input matrix

    y : array, shape (n_samples, )
        Binary classification target

    mesh_step_size : float, optional (default=0.2)
        Mesh size of the decision boundary

    title : str, optional (default="")
        Title of the graph

    inline_plotting : bool, optional (default=False)
        Show the plot in addition to save it

    """
    bg_map, sc_map = make_cmaps()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    if hasattr(fitted_estimator, "decision_function"):
        Z = fitted_estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    try:
        plt.title(title)
        plt.xlabel("X_0")
        plt.ylabel("X_1")

        # Put the result into a color plot
        plt.contourf(xx, yy, Z, cmap=bg_map, alpha=.8)

        # Plot testing point
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=sc_map, edgecolor='black')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.savefig("{}.pdf".format(fname))
        plt.savefig("{}.svg".format(fname))

    finally:
        if inline_plotting:
            plt.plot()
        else:
            plt.close()

def plot_boundary_extended(fname, fitted_estimator, X, y, mesh_step_size=0.1, title="", inline_plotting=False):
    """Plot estimator decision boundary and scatter points

    Parameters
    ----------
    fname : str
        File name where the figures is saved.

    fitted_estimator : a fitted estimator

    X : array, shape (n_samples, 2)
        Input matrix

    y : array, shape (n_samples, )
        Binary classification target

    mesh_step_size : float, optional (default=0.2)
        Mesh size of the decision boundary

    title : str, optional (default="")
        Title of the graph

    inline_plotting : bool, optional (default=False)
        Show the plot in addition to save it

    """
    bg_map, sc_map = make_cmaps()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))


    if hasattr(fitted_estimator, "decision_function"):
        Z = fitted_estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*xx.ravel(), yy.ravel()*yy.ravel(), xx.ravel()*yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    try:
        plt.title(title)
        plt.xlabel("X_0")
        plt.ylabel("X_1")
        # Put the result into a color plot
        plt.contourf(xx, yy, Z, cmap=bg_map, alpha=.8)
        #plt.scatter(xx.ravel(),yy.ravel(),c=Z.ravel()>0.5,alpha=0.1)

        # Plot testing point
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=sc_map, edgecolor='black')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.savefig("{}.pdf".format(fname))

    finally:
        if inline_plotting:
            plt.plot()
        else:
            plt.close()
