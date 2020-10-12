"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from data import make_data1, make_data2, make_data2_
from plot import plot_boundary


# (Question 2)

# Put your funtions here
# ...
#function for problem 2
def nneighbor(ls_size,ts_size,seed,plot21,plot22):
    #number of neigbors
    k = [1,5,10,75,100,150]
    FUNCS = [make_data1, make_data2]
    optimal_numbers = [0,0] 
    for prob_ndx in range( len( FUNCS)):
        error_rates = []
        p_ndx = prob_ndx + 1 # python counts from 0 :-)
        #keep track of optimal values
        min = 100
        n = 0
        
        print(f"Generating problem {p_ndx}")
        
        inputs_ls, outputs_ls, inputs_ts, outputs_ts = FUNCS[prob_ndx](ls_size,ts_size, random_state=seed)
        for number in k:
            #init KNeighbors and fit data
            neigh = KNeighborsClassifier(n_neighbors=number)
            neigh.fit(inputs_ls,outputs_ls)
            #predict outputs for test sample
            predictions = neigh.predict(inputs_ts)
            #calculate error rates
            error = np.sum(outputs_ts != predictions) / len(outputs_ts)
            error_rates.append(error)
            
            #keep track of optimal number of neighbors
            if error <= min:
                min = error
                n = number
            #plot boundaries
            if plot21:
                plot_boundary(f"../plots/neighbors/p{p_ndx}_neighbors{number}",neigh, inputs_ls, outputs_ls, title=f"Problem {p_ndx}, #neighbors {number}")
                plt.show()
        optimal_numbers[prob_ndx] = n
        
        if plot22:
             fig = plt.scatter(k,error_rates)
             plt.xlabel("Number of neighbors")
             plt.ylabel("Error rate")
             plt.title(f"Problem {p_ndx}, LS size {ls_size}")
             plt.savefig(f"../plots/neighbors/2.2/p{p_ndx}_neighbors{number}_ls{ls_size}")
             plt.clf()
             
    return optimal_numbers              
        
       
        

def cross_validation(seed):
    #generate data set for problem 2    
    inputs, outputs = make_data2_(10000, seed)
    
    #shuffle inputs and outputs in unison
    rng_state = np.random.get_state()
    np.random.shuffle(inputs)
    np.random.set_state(rng_state)
    np.random.shuffle(outputs)
    
    #create 5 random subsets
    elem_num = 2000
    ts_in = []
    ts_out = []
    for i in range(0,5):
        ts_in.append(inputs[elem_num*i:elem_num * (i+1),])
        ts_out.append(outputs[elem_num*i:elem_num * (i+1)])
    ls_in = []
    ls_out =[]
    #input sets for cross validation    
    ls_in.append(ts_in[1] + ts_in[2] + ts_in[3] + ts_in[4])
    ls_in.append(ts_in[0] + ts_in[2] + ts_in[3] + ts_in[4])
    ls_in.append(ts_in[0] + ts_in[1] + ts_in[3] + ts_in[4])
    ls_in.append(ts_in[0] + ts_in[1] + ts_in[2] + ts_in[4])
    ls_in.append(ts_in[0] + ts_in[1] + ts_in[2] + ts_in[3])
    
    #output sets for cross validation    
    ls_out.append(ts_out[1] + ts_out[2] + ts_out[3] + ts_out[4])
    ls_out.append(ts_out[0] + ts_out[2] + ts_out[3] + ts_out[4])
    ls_out.append(ts_out[0] + ts_out[1] + ts_out[3] + ts_out[4])
    ls_out.append(ts_out[0] + ts_out[1] + ts_out[2] + ts_out[4])
    ls_out.append(ts_out[0] + ts_out[1] + ts_out[2] + ts_out[3])
    
    #number of neigbors
    k = [1,5,10,75,100,150]
    #keeping track of best value for neighbors
    min = 100
    best = 0
    error_rates = [] 
    #iterate over neighbors
    for number in k:
        error = 0
        for i in range(0,5):
            #init KNeighbors and fit data
            neigh = KNeighborsClassifier(n_neighbors=number)
            neigh.fit(ls_in[i],ls_out[i])
            #predict outputs for test sample
            predictions = neigh.predict(ts_in[i])
            #add errors for every subset in order to compute mean
            error += 100.0 * np.sum(
                    ts_out[i] != predictions) / len(ts_out[i])
        #mean of error rates for model with k neighbors
        mean = error/5
        error_rates.append(mean)
        #check if error rate is lower compared to the ones seen before
        if mean <= min:
            min = mean
            best = number            
    #return best mean and optimal value for number of neighbors
    return (best,min)
        
if __name__ == "__main__":
    pass # Make your experiments here
    
    #for problem 2.1
    LEARNING_SET_SIZE = 250
    TEST_SET_SIZE = 10000
    seed = 1000
    nneighbor(LEARNING_SET_SIZE,TEST_SET_SIZE,seed,False,False)
    
    #for problem 2.2
    number, min = cross_validation(seed)
    print(f"Optimal value for neighors: {number}")
    print(f"Corresponding accuracy: {min}")
    #for problem 2.3
    LS_SIZES = [50,200,250,500]
    TEST_SET_SIZE = 500
    best_values = []
    for size in LS_SIZES:
        best_values.append(nneighbor(size,TEST_SET_SIZE,seed,False,True))
    best_values = np.array(best_values)
    #plot optimal values for neighbors with respect to LS Sizes
    plt.scatter(LS_SIZES,best_values[:,0], label="Problem 1")
    plt.scatter(LS_SIZES,best_values[:,1], label= "Problem 2")
    plt.ylabel("Neighbors")
    plt.xlabel("LS size")
    plt.legend()
    plt.title("Optimal value of neighbors")
    plt.savefig("../plots/neighbors/2.3/optimal_values.pdf")
        