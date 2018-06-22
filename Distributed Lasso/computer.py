# -*- coding: utf-8 -*-
from optimization_algorithms import  gradientDescentLasso, get_gradient, get_first_gradient

class Computer():
    def __init__(self):
        # m is the number of data points
        self.cost = 999999999 # due to minimzation
        self.num_data_points = 0
        self.is_first_run = True
        self.is_cost_small_enough = False
    def get_gradients(self, X, y, theta):
        # X is the predictor variables, y is the response and theta are the weights
        # we check if we have run the method before by checking if
        # the cost has changed, if so we can pass in the cost to calculate
        # the cost difference between the runs, this helps us find out if we have converged
        #print('self.is_first_run', self.is_first_run)
        #print('self-cost', self.cost)
        if self.is_first_run:
            self.is_first_run = False
            is_converged = False
            gradient, cost = get_first_gradient(X, y, theta, self.num_data_points)
        else:
            is_converged, gradient, cost = get_gradient(X, y, theta, self.num_data_points, self.cost)
        self.cost = cost
        
        if self.cost < 10**(-4):
            self.is_cost_small_enough = True
        return(is_converged, gradient, self.is_cost_small_enough)
    
    def lasso_gradiants(self, X, y, theta, learning_rate, num_rounds, weight_decay):
        return gradientDescentLasso(X, y, theta,
                             learning_rate, self.num_data_points, num_rounds, weight_decay)
    
