# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:47:36 2018

@author: Ingvar
"""
import numpy as np
from collections import Counter
from random import gauss

def get_most_common(parameters, errors):
    # parameters is a list of tuples
    # errors is a list
    # the function returns the parameters which appear most often
    # if there is a tie the one with lower average validation error is returned
    most_common_params = Counter(parameters).most_common(len(parameters))
    most_comon_params = [i for i in most_common_params if most_common_params[0][1] == i[1]]
    if len(most_comon_params) == 1:
        # calc the average error rate
        idx = np.where(np.array(parameters) == most_comon_params[0][0])[0]
        avg_error = np.mean([errors[i] for i in idx])
        return most_comon_params[0][0], avg_error
    else:
        chosen_one = []
        for result in most_common_params:
            idx = np.where(np.array(parameters) == result[0])[0]
            avg_error = np.mean([errors[i] for i in idx])
            chosen_one.append((avg_error, result[0]))
        return min(chosen_one)[1], avg_error # 1 is for acessing the parameters



def get_min_index(error_rate, batch_sizes):
    # error_rate and batch_sizes are both lists
    # get the index with the lowest error rate
    # if there is a tie - select the one with higher batch 
    # as that one is less due to randomization

    error_rate, batch_sizes = np.array(error_rate), np.array(batch_sizes)
    all_min_idx = np.where(error_rate == error_rate.min())[0]
    best_of_ties = np.argmax([batch_sizes[idx] for idx in all_min_idx])
    return all_min_idx[best_of_ties]


def get_objective(X, y, w, batch_size, weight_decay):
    # X is a np matrix -- predictors
    # y is a np array -- target
    # w is a np array - the weights of the trained model
    objective = 0
    for i in range(len(y)):
        objective += np.log(1 + np.exp(-y[i] * w.T * X[i]))

    return sum(objective) / batch_size + weight_decay*np.linalg.norm(w, ord = 2)


def loss_derivative(X, y, w):
    # X, y, w  are np.arrays
    # the cross entropy function is the loss function in this case
    # with taregts -1 and 1
    # see http://cseweb.ucsd.edu/~kamalika/pubs/scs13.pdf 
    
    # we use gradient clipping to ensure that the gradient is less or equal than 1
    # we clipp the gradient by the l2 norm
    abs_max_value_for_exp = 709.7 # everythin bigger results in in inf -- which give numerical problems
    abs_max_value_for_exp = np.exp(709.7)
    #print('weights',w.T )
    result = np.array([0.0 for i in range(len(X[0]))])
    for i in range(len(y)):
        exponenet = np.clip(np.exp(y[i] * w.T * X[i] ), 0.0, abs_max_value_for_exp) # ensure we dont get infs
        derivative = -(X[i] * y[i] /(exponenet + 1))
        norm_clipper = max(1, np.linalg.norm(derivative, ord = 2))
     
        result += derivative / norm_clipper
        
    return result #sum([-(X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1)) / max(1, np.linalg.norm((X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1)), ord = 2)) for i in range(len(y))])

def sigmoid_prediction(X, w):
    # x, y are input arrays
    
    Xw = np.dot(X, w)
    prediction = np.round(1 / (1 + np.exp(-Xw )))
    
    # since the labels are -1 and 1 and sigmoid is in the range 0 to 1
    if prediction == 0.0:
        return -1
    return 1


# taken from: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
def make_rand_vector(dims):
    # dims is an int
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def add_noise(num_dimensions, epsilon):
    #  num_dimensions is int and epsilon is float
    
    # we sample a uniform vector in the unit ball
    v = make_rand_vector(num_dimensions)
    sensitivity = 2
    
    # sample the magnitude l from gamma (d, senitivity/eps)
    l =  np.array([np.random.gamma(num_dimensions, sensitivity / epsilon) for i in range(num_dimensions)])

    return l * v

def project_onto_unitball(X):
    # X is a np matrix
    
    # this method projects the data onto the l2 unitball
    # the math was found here: https://math.stackexchange.com/questions/627034/orthogonal-projection-onto-the-l-2-unit-ball
    
    for i in range(len(X)):
        X[i] = X[i] / max(1, np.linalg.norm(X[i], ord = 2))
    return X



