# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:07:13 2018

@author: Ingvar
"""

import numpy as np
import time
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from multiprocessing import Pool
from collections import defaultdict, Counter

# my modules
from computer import Computer
from optimization_algorithms import get_most_common, get_best_param
#%%

def main(args):
    print('starting..!.')
    
    # get all the relevant data from the preprocessing
    X, y, num_splits, random_state = args

    # shuffle the data so we get different data for each run of this program
    # Then the program us run multiple times to get a good esitmate    
    # multiprocessing dose not do different seed, so we take a random number to start different seeds
    np.random.seed(random_state)
    X, y = shuffle(X, y)
    
    # init parameters to make code more readable
    max_iterations = 3000 
    num_samples = len(y)
    num_dimensions = len(X[0])
    range_of_samples_to_check = 1000 # since the model converges on very few images
	
    # in each itreation we use different amount of data to see how the model improvese with increased data    
    total_amount_of_data = [int(range_of_samples_to_check/num_splits) for i in range(num_splits)] 
    total_amount_of_data_in_intervals = np.cumsum(total_amount_of_data).tolist()

    # we add points to show that the model has converged
    #total_amount_of_data_in_intervals += [2000, 4000,  6000, 8000,  num_samples] # PCA
    total_amount_of_data_in_intervals  +=  [2000,  6000,  num_samples] 

    # initialize dictonaries that contain information from the cross validation
    distributed_solution = {}
    single_solution = {}
    all_data_solution = {}
	
    splited_data_intervals = []
	
    for n in total_amount_of_data_in_intervals:
        distributed_solution[n] = {'weight_decays': [], 'learning_rates': [], 'error_rates':[]}
        single_solution[n] = {'weight_decays': [], 'learning_rates': [], 'error_rates':[]}
        all_data_solution[n] = {'weight_decays': [], 'learning_rates': [], 'error_rates':[]}
        if n % 2 == 0:
            m = int(n / 2)
            splited_data_intervals.append(m)
		
        else:
            # if odd, we make the first data set have 1 more record than the second
            m = int((n + 1) / 2)
            splited_data_intervals.append(m)


    # make the two computer centers that run the program
    computer_1 = Computer() 
    computer_2 = Computer()
            
    
    loop_start_time = time.time()

    # tune the weight decay and learning rate with grid search and 5 fold cross validation
    learning_rates = [0.1, 0.05, 0.01, 0.005, 0.0001, 0.00005, 10**(-5), 5*10**(-6)]
    #weight_decays = [1.5, 1.0, 0.5, 0.1, 0.01]
    for splited_index, n in enumerate(total_amount_of_data_in_intervals):
        all_results_for_each_n_distributed = []
        all_results_for_each_n_single = []
        all_results_for_each_n_central = []
        half_data = splited_data_intervals[splited_index] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n < 360:
            weight_decays = [15, 10, 5, 1, 0.01, 10**(-5), 10**(-10), 10**(-20)]
        else:
            weight_decay = [10**(-5), 10**(-10), 10**(-20)]
        for weight_decay in weight_decays:
            Gamma = lambda x: np.sign(x) * (abs(x) - weight_decay)
            for learning_rate in learning_rates:
                # initalize lists to gather errors from the cross validation
                error_rates_distributed = []
                error_rates_single = []
                error_rates_central = []
                kf = KFold(n_splits = 5)
                for train_index, test_index in kf.split(X[:2*half_data]): 
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
					
                    num_test_samples = len(y_test)
                    m = int(len(y_train) / 2)                            
                    
                    # now we perform the distributed lasso regression
                    # initalize for the distributed algorithm
                    computer_1.cost = 999999999 # due to minimization
                    computer_1.num_data_points = m
                    computer_1.is_first_run = True
                    computer_1.is_cost_small_enough = False
                    
                    computer_2.cost = 999999999 # due to minimization
                    computer_2.num_data_points = m
                    computer_2.is_first_run = True
                    computer_2.is_cost_small_enough = False
                    theta = np.zeros(num_dimensions)
                    for i in range(max_iterations):
                        is_converged_computer_1, computer_1_gradient, computer_1_cost_small_enough = computer_1.get_gradients(X_train[:m], y_train[:m], theta )
                        is_converged_computer_2, computer_2_gradient, computer_2_cost_small_enough = computer_2.get_gradients(X_train[m:2*m], y_train[m:2*m], theta )
                        total_gradients = computer_1_gradient + computer_2_gradient
                        if is_converged_computer_1:
                            theta = computer_1_gradient
                            # if either of the computers has converge we stopp
                            break
                        elif is_converged_computer_2:
                            theta = computer_2_gradient
                            break
                        elif computer_1_cost_small_enough:
                            theta = computer_1_gradient
                            break
                        elif computer_2_cost_small_enough:
                            theta = computer_2_gradient
                            break
                        else:
                            theta  =  Gamma(theta - learning_rate * total_gradients)

                    # Evaluate the model -- check for error rate
                    total_correct_distributed = 0
                    for i in range(num_test_samples):
                        prediction = np.sign(np.dot(theta, X_test[i]))
                        if prediction == y_test[i]:
                            total_correct_distributed += 1
                    error_rates_distributed.append(1 - total_correct_distributed / num_test_samples)
                    
                    ############ If all data was at a centeralized location ###########################################
                    computer_1.num_data_points = 2 * m
                    computer_1.cost = 999999999
                    computer_1.is_first_run = True
                    computer_1.is_cost_small_enough = False
                
                    theta = np.zeros(num_dimensions)
                    for i in range(max_iterations):
                        is_converged, gradient, is_cost_small_enough = computer_1.get_gradients(X_train[:2*m], y_train[:2*m], theta)
                        if is_converged:
                            theta = gradient
                            print('converge all', (learning_rate, weight_decay))
                            break
                        elif is_cost_small_enough:
                            theta = gradient
                            print('cost all', (learning_rate, weight_decay))
                            break
                        else:
                            theta = Gamma(theta - learning_rate * gradient)

         
                    # Evaluate the model -- check for error rate
                    total_correct_all_data = 0
                    for i in range(num_test_samples):
                        prediction = np.sign(np.dot(theta, X_test[i]))
                        if prediction == y_test[i]:
                            total_correct_all_data += 1
                    
                    error_rates_central.append(1 - total_correct_all_data / num_test_samples)
                
                
                
                ############## If only one computer did the analysis on there own data ############################
                # we do another cross validation so that the ratio between training and testing is the same as for the other models
                kf = KFold(n_splits = 5)
                for train_index, test_index in kf.split(X[:half_data]): 
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
					
                    num_test_samples = len(y_test)
                    m = int(len(y_train) ) 
                    computer_1.cost = 999999999
                    computer_1.num_data_points = m
                    computer_1.is_first_run = True
                    computer_1.is_cost_small_enough = False
                    theta = np.zeros(num_dimensions)
                    for i in range(max_iterations):
                        is_converged, gradient, is_cost_small_enough = computer_1.get_gradients(X_train[:m], y_train[:m], theta)         
                        if is_converged:
                            theta = gradient
                            break
                        elif is_cost_small_enough:
                            theta = gradient
                            break
                        else:
                            theta = Gamma(theta - learning_rate * gradient)

               
                    # Evaluate the model -- check for error rate
                    total_correct_single = 0
                    for i in range(num_test_samples):
                        prediction = np.sign(np.dot(theta, X_test[i]))
                        if prediction == y_test[i]:
                            total_correct_single += 1
                    
                    error_rates_single.append(1 - total_correct_single / num_test_samples)
                

                
                # After cross validation is finsihed for a pair of weight decay and learning rate we save the results
                # we use tuples so we can find the min error rate, and at the same time have the correct weight decay and learning rate
                all_results_for_each_n_distributed.append((np.mean(error_rates_distributed), weight_decay, learning_rate))
                all_results_for_each_n_single.append((np.mean(error_rates_single), weight_decay, learning_rate))
                all_results_for_each_n_central.append((np.mean(error_rates_central), weight_decay, learning_rate))
                
        print('all_results_for_each_n_central', all_results_for_each_n_central)
        error_rate, weight_decay, learning_rate = min(all_results_for_each_n_distributed) 
        distributed_solution[n]['weight_decays'].append(weight_decay)
        distributed_solution[n]['learning_rates'].append(learning_rate)
        distributed_solution[n]['error_rates'].append(error_rate)
		
        error_rate, weight_decay, learning_rate = min(all_results_for_each_n_single) 
        single_solution[n]['error_rates'].append(error_rate)
        single_solution[n]['weight_decays'].append(weight_decay)
        single_solution[n]['learning_rates'].append(learning_rate)
		
        error_rate, weight_decay, learning_rate = min(all_results_for_each_n_central) 
        all_data_solution[n]['error_rates'].append(error_rate)
        all_data_solution[n]['weight_decays'].append(weight_decay)
        all_data_solution[n]['learning_rates'].append(learning_rate)
        print('\n{} out of {} total loop time: {}'.format(splited_index, len(total_amount_of_data_in_intervals), time.time() - loop_start_time), flush = True)
        #print('distributed_solution[n]', distributed_solution[n]['error_rates'])
        #print('single_solution[n]', single_solution[n])
        #print('all_data_solution[n]', all_data_solution[n], flush = True)
    #print('len(distributed_solution[n][error_rates])', len(distributed_solution[n]['error_rates']))
    #print('distributed_solution[n][error_rates]', distributed_solution[n]['error_rates'])
    print('Leaving..!.1')
    return (distributed_solution, single_solution, all_data_solution)
if __name__ == '__main__':
    print('I have been run')	
    num1 = 4
    num2 = 9
    
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
    
    scaler = StandardScaler()
    X_train_without_bias = scaler.fit_transform(X_train_without_bias)
    
    
    #********** UNCOMMENT TO DO PCA *******************
    # we do pca as the gradient is very sensitive in high dimensonal space
    #pca = PCA(n_components = 100) # from the graph in the report this is around 80% of the var    
    #pca.fit(X_train_without_bias)
    #X_train_without_bias = pca.transform(X_train_without_bias)
    y_train[y_train == num1] = -1
    y_train[y_train == num2] = 1

    print('data has been loaded')
            
    
    # we add bias term in front -- done for the gradient calculations
    records, attributes = np.shape(X_train_without_bias)
    X_train = np.ones((records, attributes + 1))
    X_train[:,1:] = X_train_without_bias

   
    # applay multiprocessing to make the analysis faster
    # here we do some number of total instances to average and see how the model
    # behaves, this is due to the randomness in the train, test split
    t1 = time.time()
    total_instances = 24 
    p = Pool(total_instances)
    num_splits = 50 #100 
    max_integer_val = np.iinfo(np.int32).max
    args = [(X_train, y_train, num_splits, np.random.randint(max_integer_val)) for i in range(total_instances*2)]
    result = p.map(main, args)
    
    
    distributed, single, central = zip(*result)
    p.close()
    p.join()
    print('Time taken for multiprocessing: {}'.format(time.time() - t1))
    print('')
    print(result)

    # flatten the data structure so it is wasier to work with and compair between diferenet runs
    all_instances_distributed = defaultdict(lambda: defaultdict(list))
    for i, item in enumerate(distributed):
        for key in item:
            all_instances_distributed[key]['error_rates'] += distributed[i][key]['error_rates']
            all_instances_distributed[key]['parameters'] += list(zip(distributed[i][key]['weight_decays'], distributed[i][key]['learning_rates']))
    
    
    all_instances_single = defaultdict(lambda: defaultdict(list))
    for i, item in enumerate(single):
        for key in item:
            all_instances_single[key]['error_rates'] += single[i][key]['error_rates']
            all_instances_single[key]['parameters'] += list(zip(single[i][key]['weight_decays'], single[i][key]['learning_rates']))
    
    all_instances_central= defaultdict(lambda: defaultdict(list))
    for i, item in enumerate(central):
        for key in item:
            all_instances_central[key]['error_rates'] += central[i][key]['error_rates']
            all_instances_central[key]['parameters'] += list(zip(central[i][key]['weight_decays'], central[i][key]['learning_rates']))
    
   
    
    
    # we pick the pair of parameters which come most often -- if there is a tie we select the one with
    # the lowest error rate..
    tunned_params = {'distributed':{}, 'single':{}, 'central':{}}
    for key in all_instances_distributed:
        parameters, error_rate = get_most_common(all_instances_distributed[key])
        weight_decay, learning_rate = parameters
        # key needs to be string for the json object
        tunned_params['distributed'][str(key)] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}
    
    for key in all_instances_single:
        parameters, error_rate= get_most_common(all_instances_single[key])
        weight_decay, learning_rate  = parameters
        # key needs to be string for the json object
        tunned_params['single'][str(key)] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}

    for key in all_instances_central:
        parameters, error_rate= get_most_common(all_instances_central[key])
        weight_decay, learning_rate = parameters
        # key needs to be string for the json object
        tunned_params['central'][str(key)] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}        

        
    # load all the best parameters into json so we can easly acess them in our main program 
    file_name = 'parameters12_3r_converge20_PCA.json'   
    with open(file_name, 'w') as f:
        json.dump(tunned_params, f)
        
    print('data has been loaded to {}'.format(file_name))




    tunned_params = {'distributed':{}, 'single':{}, 'central':{}}
    for key in all_instances_distributed:
        parameters, error_rate = get_best_param(all_instances_distributed[key])
        weight_decay, learning_rate = parameters
        # key needs to be string for the json object
        tunned_params['distributed'][str(key)] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}
    
    for key in all_instances_single:
        parameters, error_rate= get_best_param(all_instances_single[key])
        weight_decay, learning_rate  = parameters
        # key needs to be string for the json object
        tunned_params['single'][str(key)] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}

    for key in all_instances_central:
        parameters, error_rate= get_best_param(all_instances_central[key])
        weight_decay, learning_rate = parameters
        # key needs to be string for the json object
        tunned_params['central'][str(key)] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}        

    # load all the best parameters into json so we can easly acess them in our main program    
    file_name = 'parameters.json' 
    with open(file_name, 'w') as f:
        json.dump(tunned_params, f)
        
    print('data has been loaded to {}'.format(file_name))



    
    print('\n The total time of tunning was: {}'.format(time.time() - t1))
    

