# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:35:03 2018

@author: s161294
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# the color palette dose not have enough  colors so we add colors that go well with it
#colors = sns.color_palette("Set1", n_colors = 9) + [(1.0, 191/255, 0.0)] + sns.color_palette("Set2", n_colors = 3)[0:3:2]
colors = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
 (	95/255, 158/255, 160/255),
 (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
 (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
 (0.6, 0.6, 0.6),
 (1.0, 0.4980392156862745, 0.0),
 (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
 (1.0, 191/255, 0.0)
 ] 
#colors = sns.color_palette("Set1", n_colors = 9) + [(1.0, 191/255, 0.0)] + sns.color_palette("Set2", n_colors = 3)[0:3:2]
sns.set_palette(colors)
sns.set_style('darkgrid')


with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\differential_privacy_logistic_regression\results.json", 'r') as f:
    results = json.load(f)
    
with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\differential_privacy_logistic_regression\standard_devations.json", 'r') as f:
    standard_devations = json.load(f)
    
with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\differential_privacy_logistic_regression\noise_and_weights.json") as f:
    noise_and_weights = json.load(f)
    
with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\differential_privacy_logistic_regression\additional_params.json") as f:
    additional_params = json.load(f)

total_amount_of_data_in_interval = additional_params['total_amount_of_data_in_interval']
epsilons =  additional_params['epsilons']
num_splits = len(total_amount_of_data_in_interval)
num_simulations = 48



#%%   


# get the dict on nice format
keys = list(results.keys())
for key in keys:
    print(key)
    new_key = eval(key)
    results[new_key] = results.pop(key)
    standard_devations[new_key] = standard_devations.pop(key)
    




#%%    
# start by plotting the results


    
fig = plt.figure(figsize=(7,4))
ax = plt.subplot(111)


all_limits = []
# we use the student t distribution as we use the sample mean and sigma
t_critical =  stats.t.ppf(q = 0.95, df= num_simulations - 1) 
for i, result in enumerate(sorted(results)):
    limits = []
    for j in range(len(standard_devations[result])):
        limit = t_critical * standard_devations[result][j] / np.sqrt(num_simulations)
        limits.append(limit)
    # result[1] is the string represantation of the result
    ax.errorbar(total_amount_of_data_in_interval, results[result], yerr= limits, label = result[1], color = colors[i],\
                fmt='-o',capsize=2, markersize=5)

    all_limits.append(limits)

plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)

#Shrink current axis by 25%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.ylabel('Error rate')
plt.xlabel('Amount of training data [N]')
plt.title('Regularized Logistic Regression with Differential privacy')

#%%
# =============================================================================
# # to make the plot look better in power point
# plt.rcParams.update({'text.color' : "white",
#                      'axes.labelcolor' : "white",
#                      'xtick.color':'white',
#                      'ytick.color':'white',
#                      'figure.facecolor':'#485d70'})
# 
# =============================================================================

#plt.savefig('error_rate_log_regress.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig('error_rate_log_regress.eps', format = 'eps')

    
   
plt.show()


# close look at the ones closes to the weights
fig = plt.figure(figsize=(7,4))
ax = plt.subplot(111)

num_worst_to_skip = 3
for i, result in enumerate(sorted(results)):
    if i > num_worst_to_skip:
        # result[1] is the string represantation of the result
        ax.errorbar(total_amount_of_data_in_interval, results[result], yerr= all_limits[i], label = result[1], color = colors[i], fmt='-o',capsize=2, markersize=5)

ylim = ax.get_ylim()     
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0., fontsize = 12)

#Shrink current axis by 25%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.yscale('log')
plt.ylabel('log(Error rate)', fontsize = 12)
plt.xlabel('Amount of training data [N]', fontsize = 12)

plt.savefig('error_rate_log_regress2.eps', format = 'eps')
plt.show()

fig = plt.figure(figsize=(7,4))
ax = plt.subplot(111)

num_worst_to_skip = 1
for i, result in enumerate(sorted(results)):
    if i > num_worst_to_skip:
        ax.errorbar(total_amount_of_data_in_interval, results[result], yerr= all_limits[i], label = result[1], color = colors[i], fmt='-o',capsize=2, markersize = 5)

        
plt.legend(bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0., fontsize = 12)

#Shrink current axis by 25%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.ylabel('log(Error rate)', fontsize = 12)
plt.xlabel('Amount of training data [N]', fontsize = 12)

#plt.ylim(ylim)     
plt.yscale('log')

plt.savefig('error_rate_log_regress3.eps', format = 'eps')
plt.show()

#%%

fig = plt.figure(figsize=(7,4))
ax = plt.subplot(111)
keys = sorted(list(results.keys()))
for i, lim in enumerate(all_limits):
    ax.plot(total_amount_of_data_in_interval, lim, '-*',color = colors[i], label = keys[i][1], markersize = 5)


plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)

#Shrink current axis by 25%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
plt.xlabel('Amount of training data [N]')
plt.ylabel('Confidance interval')
plt.savefig('LogisticConfidenceMagnitude.eps', format = 'eps')
plt.show()


#%%
# investegate the weights




#%%

# plot the magnitude and the distributtion of all the weights and noises generated
x_labels = ['$\epsilon = {}$'.format(eps) for eps in epsilons]
x_labels.append('weights')

# we know that the first run has the greates axis, so we capture it
biggest_axis_boxplot = None
biggest_axis_boxplot_sinh = None
biggest_axis_magnitude = None
for i, n in enumerate(noise_and_weights):
    item = noise_and_weights[n]

    if int(n) == total_amount_of_data_in_interval[0] or int(n) == total_amount_of_data_in_interval[int(num_splits / 2)] or int(n) == total_amount_of_data_in_interval[-1]:
        noise_and_weights_distribution = []
        noise_and_weights_magnitude = []
        for eps in item:
            noise_and_weights_distribution.append(noise_and_weights[n][eps])
            noise_and_weights_magnitude.append([abs(value) for value in noise_and_weights[n][eps]])
        num_labels = len(noise_and_weights_distribution)
        #plt.title('Distribution of noise and weights for n = {}'.format(n))
        ax = sns.boxplot(data=noise_and_weights_distribution)
        plt.xticks(range(num_labels), x_labels, rotation=45, fontsize = 12)
        #plt.savefig('distributionOfNoiseWeights_n={}.png'.format(n))
        plt.savefig('distributionOfNoiseWeights_n={}.eps'.format(n), format = 'eps', bbox_inches="tight")
        if i == 0:
            biggest_axis_boxplot = ax.get_ylim()
        else:
            ax.set_ylim(biggest_axis_boxplot)
        plt.show()

        
        # lets do inverse hyperbolic transformation
        inv = [np.arcsinh(i) for i in noise_and_weights_distribution]
        #plt.title('Distribution of noise and weights for n = %s with $\mathrm{sinh}^{-1}$ transformation' % n)
        ax = sns.boxplot(data=inv)

        plt.xticks(range(num_labels), x_labels, rotation=45, fontsize = 12)
        
        if i == 0:
            biggest_axis_boxplot_sinh =  ax.get_ylim()
        else:
            ax.set_ylim(biggest_axis_boxplot_sinh)
        #plt.savefig('distributionOfNoiseWeightsLog_n={}.png'.format(n))
        plt.savefig('distributionOfNoiseWeightsLog_n={}.eps'.format(n), format = 'eps', bbox_inches="tight")
        plt.show()
        
        

        #plt.title('Magnitude off noise and the weights.. n = {} with log transformation'.format(n))
        ax = sns.barplot(data=noise_and_weights_magnitude , estimator = sum)
        plt.yscale('log')
        plt.xticks(range(num_labels), x_labels, rotation=45, fontsize = 12)
        if i == 0:
            biggest_axes_magnitude = ax.get_ylim()
        else:
            print('third')
            ax.set_ylim(biggest_axes_magnitude)
        #plt.savefig('magnitudeOfNoiseAndWeights_n_{}.png'.format(n))
        plt.savefig('magnitudeOfNoiseAndWeights_n_{}.eps'.format(n), format = 'eps', bbox_inches="tight")
        
        plt.show()

#%%
     # write variances of the noise and mean of the weights to pandas inorder to make
     # a excel file to copy into latex.....
x_labels = ['$\epsilon = {}$'.format(eps) for eps in epsilons]
x_labels.append('E(weights)')
x_labels.append('var(weights)')
names = ['interval'] + x_labels
statistics = [] 
for i, n in enumerate(noise_and_weights):
    item = noise_and_weights[n]
    statistics.append([])
    statistics[-1].append(total_amount_of_data_in_interval[i])
    for j, eps in enumerate(item):
        name = x_labels[j]
        print(name)
        # if the name stars with $ we know it is an epsilon
        if name[0] == '$':
            # get the variance of all the noise's
            statistics[-1].append(np.var(noise_and_weights[n][eps]))
        else:
            # get the variance and the mean of the weights
            statistics[-1].append(np.mean(noise_and_weights[n][eps]))
            statistics[-1].append(np.var(noise_and_weights[n][eps]))


statistics = pd.DataFrame(statistics, columns = names) 
writer = pd.ExcelWriter('output.xlsx')
statistics.to_excel(writer, 'Sheet1')
writer.save()
print('done')
# 
# 
# =============================================================================







