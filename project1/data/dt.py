"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary

RANDOM_SEED = 123456
DEC_TREES_PATH = "plots/dec_tree"


def dec_tree(seed, nb_generations, fix_data=True):

    LEARNING_SET_SIZE = 250
    TEST_SET_SIZE = 10000
    DEPTHS = [1, 2, 4, 8, None]
    FUNCS = [make_data1, make_data2]

    # For all problems
    for prob_ndx in range(len(FUNCS)):

        p_ndx = prob_ndx + 1  # python counts from 0 :-)

        print(f"Generating problem {p_ndx}")

        error_rates = ""

        inputs_ls, outputs_ls, dummy1, dummy2 = FUNCS[prob_ndx](
            TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed)

        for depth in DEPTHS:

            clf = DecisionTreeClassifier(max_depth=depth)
            clf = clf.fit(inputs_ls, outputs_ls)

            plot_boundary(f"{DEC_TREES_PATH}/p{p_ndx}_depth{depth}", clf,
                          inputs_ls, outputs_ls,
                          title=f"Problem {p_ndx}, depth {depth}")

            # plot_tree(clf, filled=True)
            # plt.show()

            ts_accuracies = []

            for gen in range(nb_generations):

                # Make sure we generate new test sets on each generation
                dummy1, dummy2, inputs_ts, outputs_ts = FUNCS[prob_ndx](
                    TEST_SET_SIZE, LEARNING_SET_SIZE,
                    random_state=seed+gen+prob_ndx)

                # Evaluate predictions on test set
                predictions = clf.predict(inputs_ts)

                # % of observations not predicted correctly (see course)
                ts_error_rate = 100.0 * np.sum(
                    outputs_ts != predictions) / len(outputs_ts)

                # Reporting ------------------------------------------

                ts_accuracies.append(ts_error_rate)

                if gen == 0:
                    # Evaluate predictions on learning set
                    predictions_ls = clf.predict(inputs_ls)
                    ls_error_rate = 100.0 \
                        * np.sum(outputs_ls != predictions_ls) \
                        / len(outputs_ls)

                    # Evaluate fitness
                    if ls_error_rate > 10:
                        overfit = "Underfit"
                    elif ls_error_rate < ts_error_rate - 2:
                        overfit = "Overfit"
                    else:
                        overfit = "-"

                    error_rates += "{} & {}\\% & {}\\% & {} \\\\\n".format(
                        depth, ls_error_rate, ts_error_rate, overfit)

            print("{} & {:.2f} & {:.2f} \\\\".format(
                depth, np.mean(ts_accuracies), np.std(ts_accuracies)))

        print(error_rates)
        print()


if __name__ == "__main__":
    if not os.path.isdir(DEC_TREES_PATH):
        os.makedirs(DEC_TREES_PATH)

    # Problem 1
    dec_tree(RANDOM_SEED, 5, False)
