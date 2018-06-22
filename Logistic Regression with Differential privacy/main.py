# -*- coding: utf-8 -*-

import numpy as np

import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from collections import defaultdict
from multiprocessing import Pool
from sklearn.utils import shuffle
from functools import partial

np.seterr(all='ignore')


def main(all_args):

    print("starting....")
    debugg = False

    X_train, y_train, X_test, y_test, total_amount_of_data_in_interval, dimensionality, epsilons, random_state, weight_decays  = all_args 
    num_test_samples = len(y_test)
    # since multiprocessing and np do not work well together we need to make a tandom seed for all simulations    
    np.random.seed(random_state)
    # shuffle the data for randomnes in the smaller values of n
    X_train, y_train = shuffle(X_train, y_train, random_state = random_state)
        
    all_accuracies = defaultdict(list)
    noise_and_weights = defaultdict(partial(defaultdict, list)) # defaultdict inside defaultdict
   
    for n in total_amount_of_data_in_interval:
        regularization_constant = 1 / weight_decays[str(n)]['weight_decay']
        clf = LogisticRegression(penalty="l2", C= 1 / regularization_constant)
        clf.fit(X_train[:n], y_train[:n])
        # we did the list procedure to copy the weights
        weights = np.asarray(list(clf.coef_[0])) 
        if debugg:
            print(clf.score(X_test, y_test))
            print(len(X_train))
            scikit_proba = clf.predict_proba(X_test)
            scikit_predict = clf.predict(X_test)
        
           
        all_accuracies[(9999, 'Without DP')].append(1 - clf.score(X_test, y_test))
        
            
        ############# add differential privacy #########################
        sensitivity = 2 / (n * regularization_constant)
        for epsilon in epsilons:
            noise = np.array([np.random.laplace(0, sensitivity / epsilon) for i in range(dimensionality)])
            clf.coef_[0] = weights + noise

            # first index has the lowest n and then it increases
            all_accuracies[(epsilon, '$\epsilon$ = {}'.format(epsilon))].append(1 - clf.score(X_test, y_test))
            noise_and_weights[n][epsilon] = noise.tolist()
        noise_and_weights[n][99999999999999999999] = weights.tolist() # add the weights at the end for plottting

    print("leaving!!!")
    return (all_accuracies, noise_and_weights)
            
if __name__ == '__main__':
    print("hallo")
    
    # load the data and select the binary classificatio problem
    num1 = 4
    num2 = 9
    
    y_train = []
    X_train = []
    with open('mnist_train.csv') as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_train.append(label)
                X_train.append(features) 
    

    y_train = np.asarray(y_train)
    X_train = np.asarray(X_train)
    
    X_train = normalize(X_train)
    y_train[y_train == num1] = -1
    y_train[y_train == num2] = 1
    


    y_test = []
    X_test = []        
    with open('mnist_test.csv') as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_test.append(label)
                X_test.append(features) 
    
    
    y_test = np.asarray(y_test)
    X_test = np.asarray(X_test)
    
    X_test = normalize(X_test)
    y_test[y_test == num1] = -1
    y_test[y_test == num2] = 1
   
    with open('logistic_results.json') as l:
        weight_decays = json.load(l)

    print('Data has ben loaded..')
    
    
    # The epsilons we are going to try to differential privacy
    epsilons = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]
    dimensionality = len(X_train[0])
    number_of_training_samples = len(X_train)
    
    
    # in each itreation we use different amount of data to see how the model improvese with increased data
    num_splits = 50    
    total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] 
    total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)
    
    # Lets do multi-threading to speed things up!
    t1 = time.time()
    total_instances =  48
    p = Pool(total_instances)
    max_integer_val = np.iinfo(np.int32).max
    args = [(X_train, y_train, X_test, y_test, total_amount_of_data_in_interval, dimensionality, epsilons, np.random.randint(max_integer_val),  weight_decays ) for i in range(total_instances)]
    results_and_weights_perturb = p.map(main, args)
    p.close()
    p.join()

    print('Time taken for multiprocessing: {}'.format(time.time() - t1))

    # get three list out of a list with tuples of three
    results,  noise_and_weights = zip(*results_and_weights_perturb)
    print('results:')
    for res in results:
        for key in res:
            if key == (0.0005, '$\\epsilon$ = 0.0005'):
                print(res[key])
                print('--------------------------')
    average_results = defaultdict(lambda:np.array([0.0 for i in range(num_splits)]))
    for result in results:
        for item in result:
            # we do str because of json only accepts strings as keys, and if we do str streight upp, the latex code meshes up
            key = '(' + str(item[0]) + ',' + '"' + item[1] + '"' + ')'
            average_results[key] += np.array(result[item])
    
    
    for result in sorted(average_results):
        average_results[result] /= total_instances
    
    
    # make average_results json seralizable
    for key in average_results:
        average_results[key] = average_results[key].tolist() 
    
    with open('results.json', 'w') as f:
        json.dump(average_results, f)
    

    # get the statndard devation of the results
    standard_devations = defaultdict(lambda:np.array([0.0 for i in range(num_splits)]))
    for result in results:
        for item in result:
            key = '(' + str(item[0]) + ',' + '"' + item[1] + '"' + ')'
            standard_devations[key] += (np.array(result[item]) - np.array(average_results[key]))**2
    
    for key in standard_devations:
        standard_devations[key] /= (total_instances - 1)
        standard_devations[key] = np.sqrt(standard_devations[key])
    # make it json serilizable
    for key in standard_devations:
        standard_devations[key] = standard_devations[key].tolist()

    with open('standard_devations.json', 'w') as f:
        json.dump(standard_devations, f) 

    
    # make a josn out of the noieses and the weights
    
    noise_and_weights_combined = defaultdict(lambda: defaultdict(list))
    for i, noise in enumerate(noise_and_weights):
        for n in noise:
            item = noise[n]
            for eps in item:
                noise_and_weights_combined[n][eps] = noise_and_weights_combined[n][eps] + noise_and_weights[i][n][eps]
                
    # make it json serializable
    noise_and_weights_combined_json = defaultdict(lambda: defaultdict(list))
    for n, value in noise_and_weights_combined.items():
        for eps in value:
            noise_and_weights_combined_json[str(n)][str(eps)] =noise_and_weights_combined[n][eps]
   
    
    with open('noise_and_weights.json', 'w') as f:
        json.dump(noise_and_weights_combined_json, f)
        
    additional_params = {'total_amount_of_data_in_interval': total_amount_of_data_in_interval.tolist(), 'epsilons': epsilons}
    with open('additional_params.json', 'w') as f:
        json.dump(additional_params, f)
