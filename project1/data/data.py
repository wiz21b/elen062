"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
import numpy as np
from sklearn.utils import check_random_state

def generate_disk(n_points, radius=1., noise=.1, random_state=None):
    """Generate a noisy disk of points

    Parameters
    ----------
    n_points : int >0
        The number of points to generate
    radius : float, optional (default=1.)
        The expected radius of the circle
    noise : float, optional (default=.1)
        The variance of the gaussian radius perturbation
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    xs : array of shape [n_points]
        The x coordinate of the points. The nth points has coordinates
        (xs[n], ys[n])
    ys : array of shape [n_points]
        The y coordinate of the points. The nth points has coordinates
        (xs[n], ys[n])
    """
    # Build the random generator
    drawer = check_random_state(random_state)
    # Draw the angles
    thetas = drawer.uniform(0, 2*np.pi, n_points)
    # Draw the radius variations
    rhos = drawer.normal(0, noise, n_points)+radius
    # Transform to cartesian coordinates
    xs = rhos*np.cos(thetas)
    ys = rhos*np.sin(thetas)
    return xs, ys

def make_circles(n_points, random_state=None):
    """Generate a dataset of two perpendicular disks

    Parameters
    ----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    # A few preliminary computations
    drawer = check_random_state(random_state)

    # First disk
    n_points1 = n_points//2
    xs1, ys1 = generate_disk(n_points1, radius=0., random_state=drawer)
    scale_matrix = np.matrix([[1, 0], [0, 1]])
    rotation_matrix = np.matrix([[1, 0], [0, 1]])
    ellipse_map = rotation_matrix.dot(scale_matrix)
    ls_x1 = ellipse_map.dot(np.matrix([xs1, ys1]))
    ls_y1 = np.zeros(n_points1, dtype=int)

    # Second disk
    n_points2 = n_points - n_points1
    xs2, ys2 = generate_disk(n_points2, radius=1., random_state=drawer)
    scale_matrix = np.matrix([[1, 0], [0, 1]])
    ellipse_map = rotation_matrix.dot(scale_matrix)
    ls_x2 = ellipse_map.dot(np.matrix([xs2, ys2]))
    ls_y2 = np.ones(n_points2, dtype=int)

    # Combine the disks
    X = np.vstack([np.array(ls_x1).T, np.array(ls_x2).T])
    y = np.hstack([ls_y1, ls_y2])
    shuffler = check_random_state(random_state)
    permutation = np.arange(n_points)
    shuffler.shuffle(permutation)

    return X[permutation], y[permutation]

def make_ellipses(n_points, flattening=2., rotation=0, random_state=None):
    """Generate a dataset of two perpendicular ellipses

    Parameters
    ----------
    n_points : int >0
        The number of points to generate
    flattening : float, optional (default=2.)
        The flattening coefficient of the ellipses
    rotation : float, optional (default=0)
        The rotation angle (in rad) of the ellipses
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    # A few preliminary computations
    drawer = check_random_state(random_state)
    sin = np.sin(rotation)
    cos = np.cos(rotation)
    dr = (flattening-1.)/(flattening+1.)

    # First ellipse
    n_points1 = n_points//2
    xs1, ys1 = generate_disk(n_points1, random_state=drawer)
    scale_matrix = np.matrix([[1+dr, 0], [0, 1-dr]])
    rotation_matrix = np.matrix([[cos, -sin], [sin, cos]])
    ellipse_map = rotation_matrix.dot(scale_matrix)
    ls_x1 = ellipse_map.dot(np.matrix([xs1, ys1]))
    ls_y1 = np.zeros(n_points1, dtype=int)

    # Second ellipse
    n_points2 = n_points - n_points1
    xs2, ys2 = generate_disk(n_points2, random_state=drawer)
    scale_matrix = np.matrix([[1-dr, 0], [0, 1+dr]])
    ellipse_map = rotation_matrix.dot(scale_matrix)
    ls_x2 = ellipse_map.dot(np.matrix([xs2, ys2]))
    ls_y2 = np.ones(n_points2, dtype=int)

    # Combine the ellipses
    X = np.vstack([np.array(ls_x1).T, np.array(ls_x2).T])
    y = np.hstack([ls_y1, ls_y2])
    shuffler = check_random_state(random_state)
    permutation = np.arange(n_points)
    shuffler.shuffle(permutation)

    return X[permutation], y[permutation]

def make_data1_(n_points, random_state=None):
    """Generate a dataset of two perpendicular and axis-aligned ellipses

    Parameters
    ----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    return make_circles(n_points, random_state=random_state)

def make_data2_(n_points, random_state=None):
    """Generate a dataset of two perpendicular ellipses rotated by 45 degrees

    Parameters
    ----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    return make_ellipses(n_points, flattening=3, rotation=np.pi/4.,
                         random_state=random_state)

def make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0):
    """Generate LS and TS for dataset 1

    Parameters
    ----------
    n_ts : int >0
        Size of TS
    n_ls : int >0
        Size of LS
    noise : float [0,1]
        Add some random noise on top of data (default: no noise)
    plot : bool
        Plot the LS.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X_train : array of shape [n_ls, 2]
        Traning input samples.
    y_train : array of shape [n_ls, 1]
        Training output samples.
    X_test : array of shape [n_ts, 2]
        Test input samples.
    y_test : array of shape [n_ts, 2]
        Test output samples.
    """
    X, y = make_data1_(n_ts+n_ls, random_state=random_state)
    X += noise*check_random_state(random_state).randn(X.shape[0],X.shape[1])
    X_train, y_train = X[:n_ls,:], y[:n_ls]
    X_test, y_test = X[n_ls:,:], y[n_ls:]
    
    if plot:
        plt.title('{} noise'.format(noise))
        X1 = X_train[y_train==0]
        X2 = X_train[y_train==1]
        plt.scatter(X1[:,0], X1[:,1], color="DodgerBlue")
        plt.scatter(X2[:,0], X2[:,1], color="orange")
        plt.show()
    
    return X_train, y_train, X_test, y_test

def make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0):
    """Generate LS and TS for dataset 2

    Parameters
    ----------
    n_ts : int >0
        Size of TS
    n_ls : int >0
        Size of LS
    noise : float [0,1]
        Add some random noise on top of data (default: no noise)
    plot : bool
        Plot the LS.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X_train : array of shape [n_ls, 2]
        Traning input samples.
    y_train : array of shape [n_ls, 1]
        Training output samples.
    X_test : array of shape [n_ts, 2]
        Test input samples.
    y_test : array of shape [n_ts, 2]
        Test output samples.
    """
    X, y = make_data2_(n_ts+n_ls, random_state=random_state)
    X += noise*check_random_state(random_state).randn(X.shape[0],X.shape[1])
    X_train, y_train = X[:n_ls,:], y[:n_ls]
    X_test, y_test = X[n_ls:,:], y[n_ls:]
    
    if plot:
        plt.title('{} noise'.format(noise))
        X1 = X_train[y_train==0]
        X2 = X_train[y_train==1]
        plt.scatter(X1[:,0], X1[:,1], color="DodgerBlue")
        plt.scatter(X2[:,0], X2[:,1], color="orange")
        plt.show()
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X_train, y_train, X_test, y_test = make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)
    X1 = X_test[y_test==0]
    X2 = X_test[y_test==1]
    plt.scatter(X1[:,0], X1[:,1], color="DodgerBlue")
    plt.scatter(X2[:,0], X2[:,1], color="darkorange")
    plt.plot([0], [0], marker="x", markersize=10)
    plt.grid(True)
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.show()

    X_train, y_train, X_test, y_test = make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)
    X1 = X_test[y_test==0]
    X2 = X_test[y_test==1]
    plt.scatter(X1[:,0], X1[:,1], color="DodgerBlue")
    plt.scatter(X2[:,0], X2[:,1], color="darkorange")
    plt.plot([0], [0], marker="x", markersize=10)
    plt.grid(True)
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.show()
