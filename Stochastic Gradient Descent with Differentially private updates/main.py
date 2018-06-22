# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:30:51 2018

@author: Ingvar
"""
import numpy as np
import json
import time
from multiprocessing import Pool
from sklearn import random_projection
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# my libraries
import utils




def sgd(all_input_params):
    X_train_without_bias, y_train, X_test_without_bias, y_test, amount_in_interval, random_state, parameters = all_input_params
    # X are the predictors, come as np array
    # y are the targets, come as np array
    # amount_in_interval is the number of samples used to geneerate learning curve
    # multiprocessing dose not do different seed, so we take a random number to start different seeds
    np.random.seed(random_state)
    # do the random projection as they do in the paper -- second paper
    transformer = random_projection.GaussianRandomProjection(n_components = 50)
    transformer.fit(X_train_without_bias)
    X_train_without_bias = transformer.transform(X_train_without_bias)
    X_test_without_bias = transformer.transform(X_test_without_bias)
    
    # we add bias term in front -- done for the gradient decent
    records, attributes = np.shape(X_train_without_bias)
    X_train = np.ones((records, attributes + 1))
    X_train[:,1:] = X_train_without_bias
    
    records, attributes = np.shape(X_test_without_bias)
    X_test = np.ones((records, attributes + 1))
    X_test[:,1:] = X_test_without_bias
    
    

    
    # shuffle so different data will be used in each process
    X_train, y_train = shuffle(X_train, y_train)
    
    num_dimensions = len(X_train[0])
    epochs = 10
    epsilons = [float('Inf')] #[0.1, 1, 10, float('Inf')]# [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, float('Inf')] # inf makes the noise go to zero -- equal to having no noise
    learning_rates = [1/np.sqrt(t + 1) for i in range(epochs) for t in range(amount_in_interval[-1])]
    results = {}
    objective_info = {}
    for epsilon in epsilons:
        if epsilon not in results:
            results[epsilon] = {}
        for n in amount_in_interval:
            if n not in results[epsilon]:
                results[epsilon][n] = {}
                #results[epsilon][n]['noise'] = []
            
            weights = np.array([0.0 for i in range(num_dimensions)])
            # param is a list which has the order -> [learning_rate, batch_size, weight_decay]
            #learning_rate = parameters[epsilon][n]['parameters'][0]
            batch_size  = parameters[epsilon][n]['parameters'][0]
            weight_decay = parameters[epsilon][n]['parameters'][1]
            
            # this if sentance is just so we can invetegate some properties only for the last model
            # where it is trained on all avilable data
            if n != amount_in_interval[-1]:
                t = 0
                for i in range(epochs):
                    # shuffle the data so the minibatch takes different data in each epoch
                    X_train_in_use, y_train_in_use = shuffle(X_train[:int(n)], y_train[:int(n)])
                    for j in range(0, len(y_train_in_use), batch_size):
                        X_batch = X_train_in_use[j:j+batch_size]
                        y_batch = y_train_in_use[j:j+batch_size]
                                    
                        # claculate the derative of the l2 norm of the weights -- regularize 
                        l2_derivative = sum(weights)
                        
                        # get the noise for all dimensions
                        noise = utils.add_noise(num_dimensions, epsilon)
                        
                        # get the objective derivative value -- look at convergance
                        objective_derivative = weight_decay * l2_derivative  + utils.loss_derivative(X_batch, y_batch, weights) / batch_size + noise / batch_size
                        
                        
                        # take a step towrads the optima
                        weights -= learning_rates[t] *(objective_derivative)
                        
                        # keep all the noise added so we can investegate it's distribution
                        #results[epsilon][n]['noise'] += noise.tolist()
                        t += 1
            else:
                print('n != amount_in_interval[-1] = {}, n {}, amount_in_interval[-1] {}'.format(n != amount_in_interval[-1], n, amount_in_interval[-1]))
                # we want to investegate how the objective changes thorugh iterations only for
                # the models which are trained on all the data
                if epsilon not in objective_info:
                    objective_info[epsilon] = {}
                    objective_info[epsilon]['objective'] = []
                    objective_info[epsilon]['gradient'] = []
                    objective_info[epsilon]['num_points'] = []
                t = 0
                for i in range(epochs):
                    if objective_info[epsilon]['num_points']:
                        points_from_last_epoch = objective_info[epsilon]['num_points'][-1]   
                    else:
                        points_from_last_epoch = 0
                        
                    # shuffle the data so the minibatch takes different data in each epoch
                    X_train_in_use, y_train_in_use = shuffle(X_train[:int(n)], y_train[:int(n)])
                    print(len(y_train ))
                    for j in range(0, len(y_train_in_use), batch_size):
                        X_batch = X_train_in_use[j:j+batch_size]
                        y_batch = y_train_in_use[j:j+batch_size]
                                    
                        # claculate the derative of the l2 norm of the weights -- regularize 
                        l2_derivative = sum(weights)
                        
                        # get the noise for all dimensions
                        noise = utils.add_noise(num_dimensions, epsilon)
                        
                        # get the objective value
                        objective = utils.get_objective(X_batch, y_batch, weights, batch_size, weight_decay)
                        
                        
                        # get the objective derivative value -- look at convergance
                        objective_derivative = weight_decay * l2_derivative  + utils.loss_derivative(X_batch, y_batch, weights) / batch_size + noise / batch_size
                        
                        # take a step towrads the optima
                        weights -= learning_rates[t] *(objective_derivative)
                        
                        objective_info[epsilon]['objective'].append(objective)
                        objective_info[epsilon]['gradient'].append(np.linalg.norm(objective_derivative, ord = 2))
                     
                        
                        objective_info[epsilon]['num_points'].append(j+batch_size + points_from_last_epoch) # if we go to the next epoch we keep on couniting
                        #results[epsilon][n]['noise'] += noise.tolist()
                        t += 1
                    
                    print('num_points', objective_info[epsilon]['num_points'], flush = True)
                    
            

            # now we predict with the trained weights, using logistic regression
            num_correct = 0
            avg_error = 0
            for i in range(len(y_test)):
                if y_test[i] == utils.sigmoid_prediction(X_test[i], weights):
                    num_correct += 1
            avg_error = num_correct/len(y_test)
            
            results[epsilon][n]['error_rate'] = 1 - avg_error
            
            
            # take the last iteration of the noise and find its magnitude
            # this is done to compare it to the wegiths to see how it influences
            # the decision process -- when epsilon is inf no noise is added and we can see how the weights are
            if epsilon == float('Inf'):
                results[epsilon][n]['noise_and_weights_magnitude'] = sum(abs(weights))
            else:
                results[epsilon][n]['noise_and_weights_magnitude'] = sum(abs(noise))
                
            # lets investegate how the noise affects the weights .. by looking at how the final weights are after
            # each noise level
            
            results[epsilon][n]['weights'] = sum(abs(weights))
                    
    
    return (results, objective_info)




if __name__ == '__main__':
    debugging = False
    small_intervals_in_begining = False
    if debugging:
        # get the data and preprocess it
        digits = load_digits()
        n_samples = len(digits.images)
        X_without_bias = digits.images.reshape((n_samples, -1))
        y = digits.target
         
        # now we only want to do binary classification of two numbers
        num1, num2 = 4, 9
        
        index_of_num1 =  np.flatnonzero( y == num1 ) #returns the indexes
        index_of_num2 = np.flatnonzero( y == num2 )
    
        # merge the two together and  sort them
        new_indexes = np.concatenate((index_of_num1, index_of_num2), axis=0)
        new_indexes = np.sort(new_indexes)
        y = y[new_indexes]
        X_without_bias = X_without_bias[new_indexes]
        
        # since we are classifying with the sign - we translate the y vector  to -1 to 1
        y[y == num1] = -1
        y[y == num2] = 1
        X_train_without_bias , X_test_without_bias, y_train, y_test = train_test_split(X_without_bias , y)
        
        
    else:
        num1, num2 = 4, 9
        y_train, X_train_without_bias = [], []
        with open('mnist_train.csv') as l:
            for i, line in enumerate(l):
                line = line.split(',')
                label = int(line[0])
                if label == num1 or label == num2:
                    features = [float(i) for i in line[1:]]
                    y_train.append(label)
                    X_train_without_bias.append(features)
        y_train = np.asarray(y_train)
        X_train_without_bias = np.asarray(X_train_without_bias)
        
        y_train[y_train == num1] = -1
        y_train[y_train == num2] = 1
        
        y_test, X_test_without_bias = [], []
        with open('mnist_test.csv') as l:
            for i, line in enumerate(l):
                line = line.split(',')
                label = int(line[0])
                if label == num1 or label == num2:
                    features = [float(i) for i in line[1:]]
                    y_test.append(label)
                    X_test_without_bias.append(features)
        y_test = np.asarray(y_test)
        X_test_without_bias = np.asarray(X_test_without_bias)
        
        y_test[y_test == num1] = -1
        y_test[y_test == num2] = 1
        
    
    # standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train_without_bias)
    X_train_without_bias = scaler.transform(X_train_without_bias)
    X_test_without_bias = scaler.transform(X_test_without_bias)
    
    # project the data onto the unitball
    X_train_without_bias = utils.project_onto_unitball(X_train_without_bias)
    X_test_without_bias = utils.project_onto_unitball(X_test_without_bias)
    
    

    
    
    # split the data upp so to get the learning curve
    num_samples = len(y_train) 
    num_splits = 50
    if small_intervals_in_begining:
        samples_to_check = 1000
        amount_of_data_in_interval = np.cumsum([int(samples_to_check / num_splits) for i in range(num_splits - 5)]).tolist()
        amount_of_data_in_interval += [2000, 4000, 6000, 8000, num_samples]
    else:
        amount_of_data_in_interval = np.cumsum([int(num_samples / num_splits) for i in range(num_splits)])
        print('amount_of_data_in_interval', amount_of_data_in_interval)
    max_integer_val = np.iinfo(np.int32).max
    
    
    if small_intervals_in_begining:
        json_file = 'parameters_tight_begining.json'
    else:
        json_file = 'parameters.json'
        json_file = 'parameters_anotherLearningRate.json'
        json_file = 'parameters_more_iterations.json'
    # get the parameters from training
    with open(json_file) as json_data:
        parameters = json.load(json_data)
    
    # change the keys from string to numbers
    param_keys = parameters
    for epsilon in list(param_keys):
        try:
            new_epsilon = eval(epsilon)
        except:
            # eval dose not recognize inf
            new_epsilon = float('Inf')
        parameters[new_epsilon] = {}
        n_keys = parameters[epsilon] 
        for n in list(n_keys):
            new_n = eval(n)
            parameters[new_epsilon][new_n] = parameters[epsilon][n]
            del parameters[epsilon][n]
        del parameters[epsilon]
    

        
    
    if debugging:
        
        args = (X_train_without_bias, y_train, X_test_without_bias, y_test, amount_of_data_in_interval,  np.random.randint(max_integer_val), parameters)
        args2 = (X_train_without_bias, y_train, X_test_without_bias, y_test, amount_of_data_in_interval,  np.random.randint(max_integer_val), parameters)
        args3 = (X_train_without_bias, y_train, X_test_without_bias, y_test, amount_of_data_in_interval,  np.random.randint(max_integer_val), parameters)
        args4 = (X_train_without_bias, y_train, X_test_without_bias, y_test, amount_of_data_in_interval,  np.random.randint(max_integer_val), parameters)
        all_results = [sgd(args), sgd(args2), sgd(args3), sgd(args4)]
        num_processes = len(all_results)
        results, objective_infos = zip(*all_results)
        
    else:
        # we run mulitiprocessing when we are not debuging
        num_threads = 24
        args = [(X_train_without_bias, y_train, X_test_without_bias, y_test, amount_of_data_in_interval,  np.random.randint(max_integer_val), parameters) for i in range(num_threads*2)] 
        t1 = time.time()
        p = Pool(num_threads)
    
        all_results = p.map(sgd, args)# [sgd(X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))] 
        results, objective_infos = zip(*all_results)
        
        p.close()
        p.join()
        print('multiporcessing finsihed, time: {}'.format(time.time() - t1))
        
#%%        
    # make the data ready for plotting
    results_flatten = {}
    for result in results:
        for epsilon in result:
            epsilon_key = str(epsilon) # json wants its key as string 
            if epsilon_key not in results_flatten:
                results_flatten[epsilon_key] = {}
            for n in result[epsilon]:
                n_key = str(n) # json wants its key as string 
                if n_key not in results_flatten[epsilon_key]:
                    results_flatten[epsilon_key][n_key] = {}
                    results_flatten[epsilon_key][n_key]['error_rate'] = [result[epsilon][n]['error_rate']]
                    #results_flatten[epsilon_key][n_key]['noise'] = result[epsilon][n]['noise']
                    results_flatten[epsilon_key][n_key]['noise_and_weights_magnitude'] = [result[epsilon][n]['noise_and_weights_magnitude']]
                    results_flatten[epsilon_key][n_key]['weights'] = [result[epsilon][n]['weights']]
                else:
                    results_flatten[epsilon_key][n_key]['error_rate'].append(result[epsilon][n]['error_rate'])
                    #results_flatten[epsilon_key][n_key]['noise'] += result[epsilon][n]['noise']
                    results_flatten[epsilon_key][n_key]['noise_and_weights_magnitude'].append(result[epsilon][n]['noise_and_weights_magnitude'])
                    results_flatten[epsilon_key][n_key]['weights'].append(result[epsilon][n]['weights'])
                    
        
    # save the data as json and use it in the plot_results.py file
    file_name = 'results_moreItertaions.json'
    
    with open(file_name, 'w') as f:
        json.dump(results_flatten, f)
    print('error rate and niose results saved in: {}'.format(file_name))
        
#%%
    objective_info_flatten ={}
    for i, objective_info in enumerate(objective_infos):
        for  epsilon in objective_info:
            if epsilon not in objective_info_flatten:
                objective_info_flatten[epsilon] = {'objective': [], 'gradient': [], 'num_points': []}
            for j in range(len(objective_info[epsilon]['objective'])):
                # we introduce list of list inside the objective_info_flatten
                # each list is for iteration number (num data points), then later it is possible to
                # calculate variance and mean for each iteration count
                if i == 0:
                    
                    # first time trough the loop
                    objective_info_flatten[epsilon]['objective'].append([objective_info[epsilon]['objective'][j]])
                    objective_info_flatten[epsilon]['gradient'].append([objective_info[epsilon]['gradient'][j]])
                    objective_info_flatten[epsilon]['num_points'].append(objective_info[epsilon]['num_points'][j])
                else:
                    objective_info_flatten[epsilon]['objective'][j].append(objective_info[epsilon]['objective'][j])
                    objective_info_flatten[epsilon]['gradient'][j].append(objective_info[epsilon]['gradient'][j])
            
            
    # save as json nad use it in plot_results.py file
    file_name = 'objective_info_moreItertaions.json'
    with open(file_name, 'w') as f:
        json.dump(objective_info_flatten, f)
    print('information on gradient and objective {}'.format(file_name))
            
                
