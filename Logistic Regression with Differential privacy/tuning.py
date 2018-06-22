# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:55:32 2018

@author: Ingvar
"""
import numpy as np
import json
import time
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from collections import defaultdict
from multiprocessing import Pool




def main(args):
    
    print('le go')
    X, y, total_amount_of_data_in_interval, random_state = args
    # since multiprocessing and np do not work well together we need to make a tandom seed for all simulations    
    np.random.seed(random_state)
    # shuffle the data so no simulation will be the same 
    X, y = shuffle(X, y, random_state = random_state)
    
    results = {}
    loop_start_time = time.time()
    for i, n in enumerate(total_amount_of_data_in_interval):
        param_gird = {'C':  [i for i in range(2, 65, 2)]} #[2**i for i in range(-8, 8)]
        log_regress = LogisticRegression()
        clf = GridSearchCV(estimator = log_regress, param_grid = param_gird, cv = 5)
        clf.fit(X[:n], y[:n])
        best_weight_decay = clf.best_estimator_.C
        results[str(n)] = (best_weight_decay, clf.best_score_)
        print('loop {} out of {}, time in loop: {}'.format(i + 1, len(total_amount_of_data_in_interval), time.time() - loop_start_time))
    return results

if __name__ == '__main__':

    
    num1 = 4
    num2 = 9
    y = []
    X = []
    path = "mnist_train.csv"
    with open(path) as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(num) for num in line[1:]]
                y.append(label)
                X.append(features)
    
    
    y = np.asarray(y)
    X = np.asarray(X)
    X = normalize(X)    
    
    y[y == num1] = -1
    y[y == num2] = 1
    
    print('data is ready!!')
    number_of_training_samples = len(y) 
    # in each itreation we use different amount of data to see how the model improvese with increased data
    num_splits = 50    
    total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] #not lin space i numpy..
    total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)
    
    total_instances = 48
    p = Pool(total_instances)
    max_integer_val = np.iinfo(np.int32).max
    args = [(X, y, total_amount_of_data_in_interval, np.random.randint(max_integer_val)) for i in range(total_instances)]
    all_results = p.map(main, args)
    
    p.close()
    p.join()
    
    print('done!')
    

    all_results_flatten = defaultdict(list)
    for simulation in all_results:
        for key, value in simulation.items():
            all_results_flatten[key].append(value)
    
    
    
    # for each n find the weight decay that appers most often, and select that one as the best
    containers_for_score_and_weight_decay = [{} for i in range(num_splits)]
    logistic_results = dict(zip(all_results_flatten.keys(), containers_for_score_and_weight_decay))
    for key, value in all_results_flatten.items():
        weight_decays, scores = zip(*value)
        best_weight_decay = max(set(weight_decays), key = weight_decays.count)
        best_score = np.mean([score for weight_decay, score in value if weight_decay == best_weight_decay])
        logistic_results[key]['weight_decay'] = best_weight_decay
        logistic_results[key]['score'] = best_score 
        
        
    json_string = 'logistic_results.json' 
    with open(json_string, 'w') as f:
        json.dump(logistic_results, f)
    
    print('results saved in {}'.format(json_string))
    
   
