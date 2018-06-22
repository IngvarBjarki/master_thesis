# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict, Counter


def gradientDescentLasso(x, y, theta, learning_rate, n, numIterations, weight_decay, tol = 10**(-4)):

    Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    xTrans = x.transpose()
    cost = 9999 # due to minimization
    previous_cost = 0 # due to minimization
    i = 0
    # print('\n=============== gradientDecentLasso =========================')
    while(i < numIterations and abs(cost - previous_cost) > tol):
        guess = np.dot(x, theta)
        loss = guess - y
        previous_cost = cost
        # avg cost per example (the 2 in 2*n doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * n)
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / n
        # update
        theta = Gamma(theta - learning_rate * gradient)
        i += 1
    return theta


def get_gradient(x, y, theta, n, previous_cost, tol = 10**(-20)):
    
    guess = np.dot(x, theta)
    loss = guess - y
    cost = np.sum(loss ** 2) / (2 * n)
    gradient = np.dot(x.T, loss) / n

    if abs(cost - previous_cost) < tol:
        return(True, gradient, cost)
    else:
        return(False, gradient, cost)
		
def get_first_gradient(x, y, theta, n):
    xTrans = x.transpose()
    guess = np.dot(x, theta)
    loss = guess - y
    cost = np.sum(loss ** 2) / (2 * n)
    gradient = np.dot(xTrans, loss) / n
    return (gradient, cost)






def get_most_common(all_instances):
    parameters = all_instances['parameters']
    errors = all_instances['error_rates']
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
        return (most_comon_params[0][0], avg_error)
    else:
        chosen_one = []
        for result in most_common_params:
            idx = np.where(np.array(parameters) == result[0])[0]
            avg_error = np.mean([errors[i] for i in idx])
            chosen_one.append((avg_error, result[0]))
        return (min(chosen_one)[1], avg_error) # 1 is for acessing the parameters



