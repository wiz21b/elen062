from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import numpy.ma as ma


class residual_fitting(BaseEstimator, ClassifierMixin):

    def pearson(self, a, b):
        D = len(a)
        a = np.array(a)
        b = np.array(b)
        m_a = a - np.mean(a) * np.ones((1, D))
        m_b = b - np.mean(b) * np.ones((1, D))
        std_a = np.std(m_a)  # np.sqrt(np.sum(m_a*m_a) / D)
        std_b = np.std(m_b)  # np.sqrt(np.sum(m_b*m_b) / D)
        cov = np.sum(m_a*m_b) / (D-1)  # np.cov gives a matrix :-(

        if abs(cov) < 1e-10:
            # avoid div by zero
            return 0
        else:
            return cov / (std_a*std_b)

    def delta_ky(self, a_indices, y_observation, a_observation, w_params):
        a_obs = ma.masked_array(a_observation, mask=a_indices)
        wk = ma.masked_array(w_params, mask=a_indices)
        return y_observation - ma.sum(ma.multiply(wk, a_obs))

    def fit(self, X, y):
        """Fit a Residual fitting model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # I think X is quite cryptic => changing names...

        # One line per observation, one column per attribute
        observations = X
        # y values corresponding to each observation
        sy = y

        N = len(observations)

        self.attr_means = [ np.mean(row) for row in observations.T]
        self.attr_stds = [ np.std(row) for row in observations.T]

        # Standardizing attributes values
        # Using numpy's broadcasting a lot here
        observations = (observations - self.attr_means) / self.attr_stds

        # Add a "1" attribute to the right of the observations
        # for w_0 weight.
        observations = np.c_[np.ones(N), observations]

        # Number of attributes including the artificial 1 column.
        DIM = len(observations[0])

        # Forward-Stagewise Regression
        # It starts like forward-stepwise regression, with an in-
        # tercept equal to ȳ, and centered predictors (observations)
        # with coefficients (w_i) initially all 0.

        w = np.array([np.mean(sy)] + [0] * (DIM-1))

        attributes_mask = np.array([False] + [True]*(DIM-1))

        for new_a_ndx in range(1,DIM):

            # At each step the algorithm computes the best
            # w_k for a_k.

            residuals = [self.delta_ky(attributes_mask, sy[o_ndx], observations[o_ndx], w)
                         for o_ndx in range(N)]

            # We compute our stuff on all the observations

            # Pick the attribute of index new_a_ndx from
            # all the observations
            a_k = [observations[o_ndx][new_a_ndx] for o_ndx in range(N)]

            # Compute best fit according to slide's 20 formula
            w[new_a_ndx] = self.pearson(residuals, a_k) * np.std(residuals)
            attributes_mask[new_a_ndx] = False

        # Save the weigths vector for later (prediction)
        self.w = w

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # Before predicting, I have to standardize observations
        # like it was done during the fitting.

        npx = np.array(X,dtype=float)
        standardized_X = ( npx - self.attr_means) / self.attr_stds

        return np.sum( np.c_[np.ones(len(X)), standardized_X]*np.transpose(self.w), axis=1)


    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        p = self.predict(X)

        # Less than 0 is closer to 0 than 1.
        # More than 1 is closer to 1 than 0
        c = np.clip(p,0,1)

        return np.c_[ np.ones( p.shape[0]) - c, c  ]




if __name__ == "__main__":
    import os
    from data import make_data1, make_data2
    from plot import plot_boundary, plot_boundary_extended

    LINREG_PATH="plots/linreg"
    if not os.path.isdir(LINREG_PATH):
        os.makedirs(LINREG_PATH)

    LEARNING_SET_SIZE = 250
    TEST_SET_SIZE = 10000
    seed = 0

    # ----------------------------------------------------------------
    # Experiment 1, dataset 1

    inputs_ls, outputs_ls, dummy1, dummy2 = make_data1(
            TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed)

    rf = residual_fitting()
    clf = rf.fit( inputs_ls, outputs_ls)

    plot_boundary(f"{LINREG_PATH}/rl1a", clf,
                  inputs_ls, outputs_ls,
                  title=f"Experiment 1, dataset 1")

    # ----------------------------------------------------------------
    # Experiment 1, dataset 2

    inputs_ls, outputs_ls, dummy1, dummy2 = make_data2(
            TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed)

    rf = residual_fitting()
    clf = rf.fit( inputs_ls, outputs_ls)

    plot_boundary(f"{LINREG_PATH}/rl1b", clf,
                  inputs_ls, outputs_ls,
                  title=f"Experiment 1, dataset 2")

    # ----------------------------------------------------------------
    # Experiment 2, dataset 2

    inputs_ls, outputs_ls, dummy1, dummy2 = make_data2(
            TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed)

    X1 = inputs_ls[:,0]
    X2 = inputs_ls[:,1]

    inputs_ls = np.c_[ inputs_ls, X1*X1, X2*X2, X1*X2]

    rf = residual_fitting()
    clf = rf.fit( inputs_ls, outputs_ls)

    plot_boundary_extended(f"{LINREG_PATH}/rl2", clf,
                           inputs_ls, outputs_ls,
                           title=f"Experiment 2, dataset 2")