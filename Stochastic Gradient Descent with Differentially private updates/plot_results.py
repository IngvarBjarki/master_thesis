# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:05:15 2018

@author: Ingvar
"""








#%%
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

#colors = sns.color_palette('Set1', n_colors = 9) + [(1.0, 191/255, 0.0)] + sns.color_palette('Set2', n_colors = 3)[0:3:2]

#colors = [sns.color_palette('Set1', n_colors = 9)[-4]] + sns.color_palette('Set1', n_colors = 9)[-2:] + [(1.0, 191/255, 0.0)]



colors = [(0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
          (1.0, 0.4980392156862745, 0.0),
          (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
          (1.0, 191/255, 0.0)
]


sns.set_palette(colors)
sns.set_style('darkgrid')

# get the data

with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\SGD\results.json") as f:
    results = json.load(f)


with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\SGD\objective_info.json") as f:
    objective_info = json.load(f)
    
    
with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\SGD\results_dim.json") as f:
    results_dim = json.load(f)
    
    
with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\SGD\parameters_more_iterations.json") as f:
    param_more_iterations = json.load(f)
    
    
#file:///L:/stochasticGradientDecent/SGD/results_moreItertaions.json
#file:///L:/stochasticGradientDecent/SGD/objective_info_moreItertaions.json
with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\SGD\results_moreItertaions.json") as f:
    param_more_iterations = json.load(f)
    
num_simulations = 24
t_critical = stats.t.ppf(q = 0.95, df = num_simulations - 1)

#!! s;mrama optimal!!!!!!!
get_label = {
                '0.0005':'$\epsilon$ = 0.0005',
                '0.001': '$\epsilon$ = 0.001',
                '0.005' : '$\epsilon$ = 0.005',
                '0.01': '$\epsilon$ = 0.01',
                '0.05': '$\epsilon$ = 0.05',
                '0.1': '$\epsilon$ = 0.1',
                '0.5': '$\epsilon$ = 0.5',
                '1': '$\epsilon$ = 1',
                '10': '$\epsilon$ = 10',
                'inf': 'Without DP',
                'Infinity': 'Without DP'
        }


#%%
for i in range(2):
    if i == 1:
        print('plot on log axis')
    all_limits = []
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    # look at the error rate    
    for epsilon in results:
        if float(epsilon) > 0.05:
            error_rates_mean = []
            error_rate_interval = []
            num_points = []
            limits = []
            for n in results[epsilon]:
                num_points.append(int(n))
                error_rate_mean = np.mean(results[epsilon][n]['error_rate'])
                
                # find confidance interval
                error_rate_std = np.std(results[epsilon][n]['error_rate'])
                limit = t_critical * error_rate_std / np.sqrt(num_simulations)
                
                error_rates_mean.append(error_rate_mean)
                limits.append(limit)
            all_limits.append((epsilon, list(limits)))
            plt.errorbar(num_points, error_rates_mean, yerr = limits, label =get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)   
        #plt.plot(num_points, error_rates_mean, 'o--', label =get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.65), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.xlabel('Amount of training data [N]')
    
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(Error rate)')
        plt.savefig('ErrorRateLog.png')
    else:
        plt.ylabel('Error rate')
        plt.savefig('ErrorRate.eps', format = 'eps')
    
    plt.show()
    
    # look at the size of the error bar
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon, limit in all_limits:
        plt.plot(limit, '*--', label=get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.65), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(upper limit - lower limit)') #!!!!!!!!!!! samr;mma!!
    else:
        plt.ylabel('upper limit - lower limit') #!!!!!!!!!!! samr;mma!!
    plt.show()
    

#%%

plotting_n = [0, int(len(results[epsilon]) / 2), int(len(results[epsilon])) - 1]
for i in range(len(plotting_n)):
    noise_magnitude = []
    x_labels = []
    # look at the noise magnitude
    for epsilon in results:
        
        for j, n in enumerate(results[epsilon]):
            
            if j == plotting_n[i]:
                if epsilon == '1':
                    # print once...
                    print('\n n = {}'.format(n))
                
                
                #noise_magnitude.append(results[epsilon][n]['weights'])
                noise_magnitude.append(results[epsilon][n]['noise_and_weights_magnitude'])
    
        x_labels.append(get_label[epsilon])
    
    
    
    sns.barplot(data = noise_magnitude, estimator = sum)
    plt.yscale('log')
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.show()





#%%
print('Investegade weights..')
# look at how the noise influences the weights by injecting into the training
plotting_n = [0, int(len(results[epsilon]) / 2), int(len(results[epsilon])) - 1]
for i in range(len(plotting_n)):
    noise_magnitude = []
    x_labels = []
    # look at the noise magnitude
    for epsilon in results:
        
        for j, n in enumerate(results[epsilon]):
            
            if j == plotting_n[i]:
                if epsilon == '1':
                    # print once...
                    print('\n n = {}'.format(n))
                
                
                noise_magnitude.append(results[epsilon][n]['weights'])
                #noise_magnitude.append(results[epsilon][n]['noise_and_weights_magnitude'])
    
        x_labels.append(get_label[epsilon])
    
    
    
    sns.barplot(data = noise_magnitude, estimator = sum)
    plt.yscale('log')
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.show()





#%%
# look at the convergance of the objective function
for i in range(2):
    all_limits = []
    if i == 1:
        print('plot on log axis')
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon in objective_info:
        if float(epsilon) > 0.05:
            # hversu margir punktar gerdu tetta... num sim..
            means = [np.mean(objective_info[epsilon]['objective'][i]) for i in range(len(objective_info[epsilon]['objective']))]
            stds = [np.std(objective_info[epsilon]['objective'][i]) for i in range(len(objective_info[epsilon]['objective']))]
            limits = [t_critical * std / np.sqrt(num_simulations - 1) for std in stds]
            all_limits.append((epsilon,limits))
            num_points = objective_info[epsilon]['num_points']
            #plt.plot(num_points, means, label = get_label[epsilon])
            num_points = [x if x <= 11791 else 11791 for x in num_points]
            
            for point in num_points:
                if point > 11750:
                    print(point)
            
            plt.errorbar(num_points, means, yerr = limits, label = get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(Objective value)')
        plt.savefig('ObjectiveLog.png')
    else:
        plt.ylabel('Objective value')
        plt.savefig('Objective.eps', format = 'eps')
    plt.show()
    
    print('look at CI')
    # look at the size of the error bar
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon, limit in all_limits:
        num_points = objective_info[epsilon]['num_points']
        plt.plot(num_points, limit, '*--', label=get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(upper limit - lower limit)') #!!!!!!!!!!! samr;mma!!
    else:
        plt.ylabel('upper limit - lower limit') #!!!!!!!!!!! samr;mma!!
    plt.show()

#%%   
# look at the gradient
for i in range(2):
    all_limits = []
    if i == 1:
        print('plot on log axis')
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon in objective_info:
        if float(epsilon) > 0.05:
            means = [np.mean(objective_info[epsilon]['gradient'][i]) for i in range(len(objective_info[epsilon]['gradient']))]
            
            stds = [np.std(objective_info[epsilon]['gradient'][i]) for i in range(len(objective_info[epsilon]['gradient']))]
            limits = [t_critical * std / np.sqrt(num_simulations - 1) for std in stds]
            all_limits.append((epsilon, limits))
            num_points = objective_info[epsilon]['num_points']
            

            plt.errorbar(num_points, means, yerr = limits, label = get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(Gradient)')
        plt.savefig('logGradient.png')
    else:
        plt.ylabel('gradient')
        plt.savefig('Gradient.eps', format = 'eps')
    plt.show()
    
    print('look at CI')
    # look at the size of the error bar
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon, limit in all_limits:
        num_points = objective_info[epsilon]['num_points']
        plt.plot(num_points, limit, '*--', label=get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(upper limit - lower limit)') #!!!!!!!!!!! samr;mma!!
    else:
        plt.ylabel('upper limit - lower limit') #!!!!!!!!!!! samr;mma!!
    plt.show()
    
    
    
#%% More iterations


fig = plt.figure(figsize=(7, 4))
ax = plt.subplot(111)  
all_limits = []
for epsilon in param_more_iterations:
    data_points = []
    means = []
    limits = []
    for n in  param_more_iterations[epsilon]:
        data_points.append(int(n))
        error_rate = param_more_iterations[epsilon][n]['error_rate']
        means.append(np.mean(error_rate))
        std = np.std(error_rate)
        limit = t_critical * error_rate_std / np.sqrt(num_simulations)
        limits.append(limit)

    plt.errorbar(data_points, means, yerr = limits, label = get_label[epsilon] + '-10 epoch' ,  fmt = '--s', capsize = 2, markersize = 5, color = (0.40, 120/255, 0.8))
    data_points = []
    means = []
    limits = []
    for n in results['inf']:
        data_points.append(int(n))
        error_rate = results[epsilon][n]['error_rate']
        means.append(np.mean(error_rate))
        std = np.std(error_rate)
        limit = t_critical * error_rate_std / np.sqrt(num_simulations)
        limits.append(limit)

    plt.errorbar(data_points, means, yerr = limits, label = get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5, color = (1.0, 191/255, 0.0))
plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0.)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.savefig('10EpochVS1.eps', format = 'eps')
plt.show()
    
    
    
    
    
#%% Shows how dimensions affects privacy


points = {}


shapes = ['--o', '--s', '--^', '--d', '--v']
epsilon_colors = [[ (0.7371472510572856, 0.895517108804306, 0.7108342945021145),
                   (0.5573241061130334, 0.8164244521337947, 0.546958861976163),
                   (0.3388235294117647, 0.7117262591311034, 0.40584390618992694),
                   (0.17139561707035755, 0.581514801999231, 0.2979008073817762),
                   (0.017762399077277974, 0.44267589388696654, 0.18523644752018453)],

                ['#fdd0a2',
                 '#fdae6b',
                 '#fd8d3c',
                 '#f16913',
                 '#d94801'],
                
                [  (0.9882352941176471, 0.6866743560169165, 0.5778854286812765),
                 (0.9865897731641676, 0.5067281814686659, 0.38123798539023457),
                 (0.9570011534025374, 0.3087120338331411, 0.22191464821222606),
                 (0.8370472895040368, 0.13394848135332565, 0.13079584775086506),
                 (0.6663437139561708, 0.06339100346020761, 0.08641291810841982)],
                     ['#ffe99c','#ffdf83','#ffd15f','#ffc53a','#ffbb00']]
for d in results_dim:
    for epsilon in results_dim[d]:
        if epsilon not in points:
            points[epsilon] = {}
        if d not in points[epsilon]:
            points[epsilon][d] = {'means':[], 'limits':[]}
            data_points = []
        for n in results_dim[d][epsilon]:
            data_points.append(int(n))
            a = results_dim[d][epsilon][n]['error_rate']
            points[epsilon][d]['means'].append(np.mean(a))
            error_rate_std = np.std(a)
            limit = t_critical * error_rate_std / np.sqrt(num_simulations)
            points[epsilon][d]['limits'].append(limit)
        
        
for i, epsilon in enumerate(points):
    fig = plt.figure(figsize=(14, 4))
    ax = plt.subplot(111)
    plt.subplot(1,2,1)
    for j, d in enumerate(points[epsilon]):
            plt.plot(data_points, points[epsilon][d]['means'], shapes[j], label = '$\epsilon = $' +epsilon +' - '+'$d$ = ' +d, color = epsilon_colors[i][j])
    plt.ylabel('Error rate')
    plt.xlabel('Amount of training data [N]')
    box = ax.get_position()
    plt.subplot(1,2,2)
    for j, d in enumerate(points[epsilon]):
        if d != 'all':
            plt.plot(data_points, points[epsilon][d]['limits'], shapes[j], label = '$\epsilon = $' +epsilon +', '+'$d$ = ' +d, color = epsilon_colors[i][j])
        else:
            plt.plot(data_points, points[epsilon][d]['limits'], shapes[j], label = '$\epsilon = $' +epsilon +', '+'$d$ = 784', color = epsilon_colors[i][j])
    plt.ylabel('Confidence interval')
    plt.xlabel('Amount of training data [N]')
    plt.legend(loc='upper center', bbox_to_anchor=(-0.10, -0.12),
          fancybox=True, shadow=True, ncol=5, borderaxespad=0.)
    box = ax.get_position()
    plt.savefig('ErrorRateDims_eps{}.eps'.format(epsilon), format = 'eps', bbox_inches='tight')
    plt.show()
    
    
