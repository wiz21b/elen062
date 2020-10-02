"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from data.data import make_data1, make_data2
from data.plot import plot_boundary



RANDOM_SEED = 123456


def dec_tree(seed, nb_generations, fix_data=True):

    LEARNING_SET_SIZE=250
    TEST_SET_SIZE=10000
    DEPTHS = [1,2,4,8,None]
    FUNCS = [make_data1, make_data2]

    # For all problems
    for prob_ndx in range( len( FUNCS)):

        p_ndx = prob_ndx + 1 # python counts from 0 :-)

        print(f"Generating problem {p_ndx}")

        error_rates = ""

        inputs_ls, outputs_ls, dummy1, dummy2 = FUNCS[prob_ndx](TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed)

        for depth in DEPTHS:

            clf = DecisionTreeClassifier( max_depth= depth )
            clf = clf.fit(inputs_ls, outputs_ls)

            plot_boundary( f"plots/dec_tree/p{p_ndx}_depth{depth}", clf, inputs_ls, outputs_ls, title=f"Problem {p_ndx}, depth {depth}")

            # plot_tree(clf, filled=True)
            # plt.show()

            ts_accuracies = []

            for gen in range(nb_generations):

                # Make sure we generate new test sets on each generation
                dummy1, dummy2, inputs_ts, outputs_ts = FUNCS[prob_ndx](TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed+gen+prob_ndx)

                # Evaluate predictions on test set
                predictions = clf.predict( inputs_ts)

                # % of observations not predicted correctly (see course)
                ts_error_rate = 100.0 * np.sum( outputs_ts != predictions) / len(outputs_ts)

                # Reporting ------------------------------------------

                ts_accuracies.append( ts_error_rate)

                if gen == 0:
                    # Evaluate predictions on learning set
                    predictions_ls = clf.predict( inputs_ls)
                    ls_error_rate = 100.0 * np.sum( outputs_ls != predictions_ls) / len(outputs_ls)

                    # Evaluate fitness
                    if ls_error_rate > 10:
                        overfit = "Underfit"
                    elif ls_error_rate < ts_error_rate - 2:
                        overfit = "Overfit"
                    else:
                        overfit = "-"

                    error_rates += "{} & {}\% & {}\% & {} \\\\\n".format( depth, ls_error_rate, ts_error_rate, overfit)


            print("{} & {:.2f} & {:.2f} \\\\".format(depth, np.mean(ts_accuracies), np.std(ts_accuracies) ))

        print(error_rates)
        print()


#function for problem 2
def nneighbor(seed):
    #number of neigbors
    k = [1,5,10,75,100,150]
    for prob_ndx in range( len( FUNCS)):

        p_ndx = prob_ndx + 1 # python counts from 0 :-)

        print(f"Generating problem {p_ndx}")

        inputs_ls, outputs_ls, inputs_ts, outputs_ts = FUNCS[prob_ndx](TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=seed)
        for number in k:
            #init KNeighbors and fit data
            neigh = KNeighborsClassifier(n_neighbors=number)
            neigh.fit(inputs_ls,outputs_ls)
            predictions = neigh.predict(inputs_ts)
            plot_boundary(f"plots/neighbors/p{p_ndx}_#neighbors{number}",neigh, inputs_ls, outputs_ls, title=f"Problem {p_ndx}, #neighbors {number}")
            plt.show()

#def cross_validation(k):



if __name__ == "__main__":
    # Problem 1.1
    dec_tree( RANDOM_SEED, 5, False)


    # Problem 2

    #generate multiple generations of data
    # generations = 1
    # for i in range(0,generations):
    #     seed = 100000+i
    #     print(f"Generating data generation {i+1}")
    #     nneighbor(seed)
