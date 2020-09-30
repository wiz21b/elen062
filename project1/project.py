import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data.data import make_data1, make_data2
from data.plot import plot_boundary


LEARNING_SET_SIZE=250
TEST_SET_SIZE=10000


FUNCS = [make_data1, make_data2]

for prob_ndx in range( len( FUNCS)):

    p_ndx = prob_ndx + 1

    print(f"Generating problem {p_ndx}")
    inputs_ls, outputs_ls, inputs_ts, outputs_ts = FUNCS[prob_ndx](TEST_SET_SIZE, LEARNING_SET_SIZE, random_state=1000)

    DEPTHS = [1,2,4,8,None]

    for depth in DEPTHS:

        clf = DecisionTreeClassifier( max_depth= depth )
        clf = clf.fit(inputs_ls, outputs_ls)

        #plot_boundary( f"p{p_ndx}_depth{depth}", clf, inputs_ls, outputs_ls, title=f"Problem {p_ndx}, depth {depth}")

        # plot_tree(clf, filled=True)
        # plt.show()

        predictions = clf.predict( inputs_ts)

        ts_error_rate = 100.0 * np.sum( outputs_ts != predictions) / len(outputs_ts)

        predictions_ls = clf.predict( inputs_ls)
        ls_error_rate = 100.0 * np.sum( outputs_ls != predictions_ls) / len(outputs_ls)


        print( "{} \t Error rate LS : {:.2f}%\t TS : {:.2f}%".format( f"Problem {p_ndx}, depth {depth}", ls_error_rate, ts_error_rate ))
