import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data.data import make_data1, make_data2
from data.plot import plot_boundary


LEARNING_SET_SIZE=250
TEST_SET_SIZE=10000

LS_DATA = [ make_data1( LEARNING_SET_SIZE, random_state=1000),
            make_data2( LEARNING_SET_SIZE, random_state=1000) ]

TEST_DATA = [ make_data1( TEST_SET_SIZE, random_state=9999),
              make_data2( TEST_SET_SIZE, random_state=9999) ]


for prob_ndx in range( len( LS_DATA)):

    learning_set = LS_DATA[prob_ndx]
    p_ndx = prob_ndx + 1

    print(f"Generating problem {p_ndx}")
    inputs_ls = learning_set[0]
    outputs_ls = learning_set[1]

    DEPTHS = [1,2,4,8,None]

    for depth in DEPTHS:

        clf = DecisionTreeClassifier( max_depth= depth )
        clf = clf.fit(inputs_ls, outputs_ls)

        #plot_boundary( f"p{p_ndx}_depth{depth}", clf, inputs_ls, outputs_ls, title=f"Problem {p_ndx}, depth {depth}")

        # plot_tree(clf, filled=True)
        # plt.show()

        test_set = TEST_DATA[prob_ndx]
        inputs_ts = test_set[0]
        outputs_ts = test_set[1]
        predictions = clf.predict( inputs_ts)

        ts_success_rate = 100.0 * np.sum( outputs_ts == predictions) / len(outputs_ts)

        predictions_ls = clf.predict( inputs_ls)
        ls_success_rate = 100.0 * np.sum( outputs_ls == predictions) / len(outputs_ts)

        print( "{} \t Success rate LS : {}\t Success rate TS : {}%".format( f"Problem {p_ndx}, depth {depth}", ls_success_rate, ts_success_rate ))
