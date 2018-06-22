"""
Created on Fri Mar  9 16:34:27 2018

@author: helga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:07:13 2018

@author: Ingvar
"""

import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import time
import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from multiprocessing import Pool
from computer import Computer
from collections import defaultdict
#%%
#sns.set_style(style = 'darkgrid')
def main(args):
    print('starting..!.')
    # get all the relevant data from the preprocessing
    X_train, y_train, X_test, y_test, num_splits, tunned_parameters, random_state = args
    
    # multiprocessing dose not update the seed - so we make sure its random
    np.random.seed(random_state)

    # shuffle to introduce randomness in the smaller samples
    X_train, y_train = shuffle(X_train, y_train)
    
    # init paramters to make code more readable
    num_test_samples = len(y_test)
    num_dimensions = len(X_train[0])
    max_iterations = 3000
    range_of_samples_to_check = 1000
    num_training_samples = len(y_train)
    #final_points = [2000, 4000,  6000, 8000, num_training_samples] # for PCA
    final_points = [2000,  6000,  num_training_samples] # without PCA
    num_splits -= len(final_points) # since we add the final points by hand
    
    # in each itreation we use different amount of data to see how the model improvese with increased data    
    total_amount_of_data = [int(range_of_samples_to_check / num_splits) for i in range(num_splits)] 
    total_amount_of_data_intervals = np.cumsum(total_amount_of_data).tolist()

    # we add final points to show that the model has converged
    total_amount_of_data_intervals += final_points 
    print('total_amount_of_data_intervals', total_amount_of_data_intervals)

    # make the two computer centers that run the program
    computer_1 = Computer()
    computer_2 = Computer()

    # container for the accuricies
    accuracies_distributed, accuracies_single, accuracies_central = [], [], []

    # container for the weights
    distributed_weights, single_weights, central_weights  = {}, {}, {}
    
    loop_start_time = time.time()
    for loop_num, n in enumerate(total_amount_of_data_intervals):
        
        # split the data differently according to if it is odd or even
        if n % 2 == 0:
            m = int(n / 2)
            
        else:
            # if odd, we add one ovservation to ensure that there is the same amount of 
            # observations in both places.
            m = int((n + 1) / 2)

			
        # init the computer before distributed run 
        computer_1.cost = 999999999 # due to minimization
        computer_1.num_data_points = m
        computer_1.is_first_run = True
        computer_1.is_cost_small_enough = False
        
        computer_2.cost = 999999999 # due to minimization
        computer_2.num_data_points = m    
        computer_2.is_first_run = True
        computer_2.is_cost_small_enough = False
		
        learning_rate =tunned_parameters['distributed'][str(n)]['learning_rate']
        weight_decay = tunned_parameters['distributed'][str(n)]['weight_decay']
        Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)     
        theta = np.zeros(num_dimensions)
        for i in range(max_iterations):
            is_converged_computer_1, computer_1_gradient, computer_1_cost_small_enough = computer_1.get_gradients(X_train[:m], y_train[:m], theta)
            is_converged_computer_2, computer_2_gradient, computer_2_cost_small_enough = computer_2.get_gradients(X_train[m:2*m], y_train[m:2*m], theta)
            total_gradients = computer_1_gradient + computer_2_gradient
            if is_converged_computer_1:
                # if either of the computers has converge we stopp
                print('distributed converge')
                theta = computer_1_gradient
                break
            elif is_converged_computer_2:
                print('distributed converge2')
                theta = computer_2_gradient
                break
            elif computer_1_cost_small_enough:
                print('distributed cost1')
                theta = computer_1_gradient
                break
            elif computer_2_cost_small_enough:
                print('distributed cost2')
                theta = computer_2_gradient
                break
            else:
                theta = Gamma(theta - learning_rate * total_gradients)
        if i == max_iterations - 1:
            print('distributed maxItr')

        # Evaluate the model -- check for error rate
        total_correct_distributed = 0
        for i in range(num_test_samples):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_distributed += 1
        
        distributed_weights[str(n)] = theta
        accuracies_distributed.append(1 - total_correct_distributed / num_test_samples)
        
    ############## If only one computer did the analysis on there own data ############################
        # initalize the data center
        computer_1.cost = 999999999 # due to minimization
        computer_1.num_data_points = m
        computer_1.is_first_run = True
        computer_1.is_cost_small_enough = False

        learning_rate = tunned_parameters['single'][str(n)]['learning_rate']
        weight_decay = tunned_parameters['single'][str(n)]['weight_decay']
        Gamma = lambda x: np.sign(x) * (abs(x) - weight_decay)        
        theta = np.zeros(num_dimensions)
        for i in range(max_iterations):
            is_converged, gradient, is_cost_small_enough = computer_1.get_gradients(X_train[:m], y_train[:m], theta)
            if is_converged:
                print('single converge')
                theta = gradient
                break
            elif is_cost_small_enough:
                theta = gradient
                print('single cost')
                break
            else:
                theta = Gamma(theta - learning_rate * gradient)
        if i == max_iterations - 1:
            print('single maxItr')
        
        # Evaluate the model -- check for error rate
        total_correct_single = 0
        for i in range(num_test_samples):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_single += 1

        single_weights[str(n)] = theta
        accuracies_single.append(1 - total_correct_single / num_test_samples)
    
    ############ If all data was at a centeralized location ###########################################
        # initalize the data center
        theta = np.zeros(num_dimensions)
        computer_1.cost = 999999999 # due to minimization
        computer_1.num_data_points = 2*m
        computer_1.is_first_run = True
        computer_1.is_cost_small_enough = False

        learning_rate = tunned_parameters['central'][str(n)]['learning_rate']
        weight_decay = tunned_parameters['central'][str(n)]['weight_decay']
        Gamma = lambda x: np.sign(x) * (abs(x) - weight_decay)
        for i in range(max_iterations):
            is_converged, gradient, is_cost_small_enough = computer_1.get_gradients(X_train[:2*m], y_train[:2*m], theta )
            if is_converged:
                theta = gradient
                print('central converge')
                break
            elif is_cost_small_enough:
                theta = gradient
                print('central cost')
                break
            else:
                theta = Gamma(theta - learning_rate * gradient)
        if i == max_iterations - 1:
            print('central maxItr')
		

        # Evaluate the model -- check for error rate
        total_correct_all_data = 0
        for i in range(num_test_samples):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_all_data += 1
        
        central_weights[str(n)] = theta
        accuracies_central.append(1 - total_correct_all_data / num_test_samples)
        print('loop {} out of {}, time from first loop: {}'.format(loop_num, len(total_amount_of_data_intervals), time.time() - loop_start_time))
    print('Leaving..!.1')
    return (
            {
           'distributed':np.array(accuracies_distributed),
           'single':np.array(accuracies_single),
           'central_all_data':np.array(accuracies_central),
           'total_amount_of_data_intervals':np.array(total_amount_of_data_intervals)},
           {
           'distributed': distributed_weights,
           'single': single_weights,
           'central':central_weights}
           )

if __name__ == '__main__':

    #!!! DEFINE CLASSIFICATION PROBLEM
    num1, num2 = 4, 9
    y_train = []
    X_train_without_bias = []
    with open("mnist_train.csv") as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_train.append(label)
                X_train_without_bias.append(features) 
    

    y_train = np.asarray(y_train)
    X_train_without_bias = np.asarray(X_train_without_bias)
    scalar = StandardScaler()
    scalar.fit(X_train_without_bias)
    X_train_without_bias = scalar.transform(X_train_without_bias)
    #X_train_without_bias = normalize(X_train_without_bias)
    
    y_train[y_train == num1] = -1
    y_train[y_train == num2] = 1
    


    y_test = []
    X_test_without_bias = []        
    with open("mnist_test.csv") as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_test.append(label)
                X_test_without_bias.append(features) 
    
    
    y_test = np.asarray(y_test)
    X_test_without_bias = np.asarray(X_test_without_bias)
    X_test_without_bias = scalar.transform(X_test_without_bias)
    #X_test_without_bias = normalize(X_test_without_bias)

    y_test[y_test == num1] = -1
    y_test[y_test == num2] = 1

    # ************** IS PCA ***********************************
    # we reduce the dimensionality with PCA since the gradient is very sensable in high dimensional space and to speeds things up!
    #pca = PCA(n_components=100)
    #pca.fit(X_train_without_bias)
    #X_train_without_bias = pca.transform(X_train_without_bias)
    #X_test_without_bias = pca.transform(X_test_without_bias)

# =============================================================
    # load the parameters from the tunned model
    parameters_from_tuning = 'parameters.json'#'parameters.json' # 'parameters_tunned_just_rate_Learning_rate_freq.json'
    with open(parameters_from_tuning) as f:
        tunned_parameters = json.load(f)
    
    print('data has been loaded')
    print('length of n={}'.format(len(y_train)))
    
    
    
    # we add bias term in front -- done for the gradient calculations
    records, attributes = np.shape(X_train_without_bias)
    X_train = np.ones((records, attributes + 1))
    X_train[:,1:] = X_train_without_bias
    
    
    
    records, attributes = np.shape(X_test_without_bias)
    X_test = np.ones((records, attributes + 1))
    X_test[:,1:] = X_test_without_bias
    
   
    # applay multiprocessing to make the analysis faster
    # here we do some number of total instances to average and see how the model
    # behaves, this is due to the randomness in the train, test split
    t1 = time.time()
    total_instances = 24
    p = Pool(total_instances)
    total_instances *= 2
    num_splits = 53 #105 
    max_integer_val = np.iinfo(np.int32).max
    args = [(X_train, y_train, X_test, y_test, num_splits, tunned_parameters, np.random.randint(max_integer_val)) for i in range(total_instances )]
    all_results = p.map(main, args)
    p.close()
    p.join()
    results, weights = zip(*all_results)
    print('Time taken for multiprocessing: {}'.format(time.time() - t1))

    average_results = {
            'distributed': np.zeros(num_splits),
            'single': np.zeros(num_splits),
            'central_all_data': np.zeros(num_splits),
            'total_amount_of_data_intervals': np.zeros(num_splits)
            }

    # calculate the average of all the instances
    for res in results:
        for key in res:
            average_results[key] += res[key]

    for key in average_results:
        average_results[key] /= total_instances 
		
    # make it json serilizable, change np array to list
    for key in average_results:
        average_results[key] =  average_results[key].tolist()
    
    with open('parameters_to_plot.json', 'w') as f:
	    json.dump(average_results, f)

##################### Calculate the standard devation #######################
    standard_devation_of_results = {
            'distributed': np.zeros(num_splits), 
            'single': np.zeros(num_splits),
            'central_all_data': np.zeros(num_splits),
            'total_amount_of_data_intervals': np.zeros(num_splits)
            }

    for res in results:
        for key in res:
            standard_devation_of_results[key] += (res[key] - average_results[key])**2

    for key in  standard_devation_of_results:
        standard_devation_of_results[key] /= (total_instances  - 1)
        standard_devation_of_results[key] = np.sqrt(standard_devation_of_results[key])
    # make it json serilizable, change np array to list
    for key in standard_devation_of_results:
        standard_devation_of_results[key] = standard_devation_of_results[key].tolist()
        
    with open('standard_devation_to_plot.json', 'w') as f:
        json.dump(standard_devation_of_results, f)

###################### Analyse the weights #################################
average_weights = {'distributed':defaultdict(list), 'single': defaultdict(list), 'central': defaultdict(list)}
for weight in weights:
    for method_key, method in weight.items():
        for num_data_key in method:
            if  num_data_key not in average_weights[method_key]:
                average_weights[method_key][num_data_key ] = method[num_data_key]
            else:
                average_weights[method_key][num_data_key ] += method[num_data_key]
 

#%%
       
for method_key, method_value in average_weights.items():
    for num_data_key in method_value:
        average_weights[method_key][num_data_key] = average_weights[method_key][num_data_key] / len(weights)
        # make it json serializable
        average_weights[method_key][num_data_key] = average_weights[method_key][num_data_key].tolist()
    
with open('average_weights.json', 'w') as f:
    json.dump(average_weights, f)
