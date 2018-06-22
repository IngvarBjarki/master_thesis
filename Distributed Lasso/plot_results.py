# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:36:54 2018

@author: s161294
"""
#%%
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import defaultdict

#%%



colors = sns.color_palette("Set1", n_colors = 9) + [(1.0, 191/255, 0.0)] + sns.color_palette("Set2", n_colors = 3)[0:3:2] 
sns.set_style('darkgrid')
sns.set_palette(colors[:3])



shapes = ['--^', '--o', '--*']
labels = {'distributed': 'Distributed', 'single': 'N/2 at one center', 'central_all_data': 'N at one center'}

# SELECT WHICH TO PLOT
path_PCA = r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\distributed_lasso\now\parameters_to_plot_123r.json"
path_noPCA =r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\distributed_lasso\now\parameters_to_plot_123rNoPCA.json"


with open(path_noPCA) as f:
    average_results = json.load(f)

path_PCAStandardDev = r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\distributed_lasso\now\standard_devation_to_plot_123r.json"
path_noPCAStandardDev =r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\distributed_lasso\now\standard_devation_to_plot_123rNoPCA.json"
with open(path_noPCAStandardDev) as f:
    standard_dev_res = json.load(f)

num_simulations = 24
t_critical =  stats.t.ppf(q = 0.95, df= num_simulations - 1)



# ****************************** close upp *******************************************
fig = plt.figure(figsize=(7,4))

ax = plt.subplot(111)
   
start = 0
end = 20

shape_index = 0    
for key in standard_dev_res:
    limits = []
    if not key == 'total_amount_of_data_intervals':
        for i in range(len(standard_dev_res[key])):
            limit = t_critical * standard_dev_res[key][i] / np.sqrt(num_simulations)
            limits.append(limit)
        print(key, average_results[key][start:end])
        ax.errorbar(average_results['total_amount_of_data_intervals'][start:end],average_results[key][start:end], yerr= limits[start:end],\
                    label = labels[key],fmt=shapes[shape_index],capsize=2, markersize=5)
        shape_index += 1

plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0.)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.ylabel('Error rate')
plt.xlabel('Amount of training data [N]')
plt.savefig('distributed_lasso_noPCA_Final.eps', format = 'eps')

plt.show()



# ****************** Log axis ***************************************


fig = plt.figure(figsize=(7,4))
ax = plt.subplot(111)
shape_index = 0

data_for_single_table = {}
data_for_central_table = {}   
# the average and the standard division have the same keys
for key in standard_dev_res:
    upper_limits = []
    lower_limits = []
    limits = []
    if not key == 'total_amount_of_data_intervals':
        for i in range(len(standard_dev_res[key])):
            # 95% confidance interval...
            limits = t_critical * standard_dev_res[key][i] / np.sqrt(num_simulations)
            upper_limit = average_results[key][i] + limits
            lower_limit = average_results[key][i] - limits
            upper_limits.append(upper_limit) 
            lower_limits.append(lower_limit)
        ax.errorbar(average_results['total_amount_of_data_intervals'],average_results[key], yerr= limits,\
            label = labels[key],fmt=shapes[shape_index],capsize=2, markersize=5)
        plt.plot(average_results['total_amount_of_data_intervals'],average_results[key])
        
        shape_index += 1
        
        if key == 'distributed':
            data_for_dist_table = zip(average_results['total_amount_of_data_intervals'], lower_limits, average_results['distributed'], upper_limits)
        elif key == 'single':
             data_for_single_table = zip(average_results['total_amount_of_data_intervals'], lower_limits, average_results['single'], upper_limits)
        elif key == 'central_all_data':
            data_for_central_table = zip(average_results['total_amount_of_data_intervals'], lower_limits, average_results['central_all_data'], upper_limits)
            
plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0., fontsize = 13)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.ylabel('log(Error rate)', fontsize = 13)
plt.xlabel('Amount of training data [N]', fontsize = 13)
plt.yscale('log')

plt.savefig('distributed_lasso_noPCA_logScaleFinal.eps', format = 'eps')
plt.show()



############### ANALYSE THE WEIGHTS ##########################################
#%%

path_weights_PCA = r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\distributed_lasso\now\average_weights_123r.json"
path_weights_noPCA =r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\distributed_lasso\now\average_weights_123rNoPCA.json"
with open(path_weights_noPCA) as f:
    average_weights = json.load(f)
    
    
threshold = 10**(-2)
number_of_weights_terminated = defaultdict(list)       
for key, value in average_weights.items():
    for key2 in sorted(value, key=int):
        value2 = value[key2]
        #print(key2)
        number_of_weights_terminated[key].append(sum(1 for i in value2 if abs(i) < threshold))
    #print('NEXT')
        

        
N = len(number_of_weights_terminated['single'][::10])
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)

yvals = number_of_weights_terminated['distributed'][::10]
ax.bar(ind, yvals, width, label = 'Distributed')
zvals = number_of_weights_terminated['single'][::10]
ax.bar(ind+width, zvals, width, label = 'N/2 at one center')
kvals = number_of_weights_terminated['central'][::10]
ax.bar(ind+width*2, kvals, width, label = 'N at one center')

ax.set_ylabel('Count')
ax.set_xlabel('Amount training data [N]')
ax.set_xticks(ind+width)
ax.set_xticklabels([int(i) for i in average_results['total_amount_of_data_intervals'][::10]])


plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0.)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ylim = ax.get_ylim()
#plt.savefig('lasso_feature_selection.eps', format='eps') 
plt.show() 





threshold = 10**(-100)
number_of_weights_terminated = defaultdict(list)       
for key, value in average_weights.items():
    for key2 in sorted(value, key=int):
        value2 = value[key2]
        number_of_weights_terminated[key].append(sum(1 for i in value2 if abs(i) < threshold))


        
N = len(number_of_weights_terminated['single'][::10])
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)

yvals = number_of_weights_terminated['distributed'][::10]
ax.bar(ind, yvals, width, label = 'Distributed')
zvals = number_of_weights_terminated['single'][::10]
ax.bar(ind+width, zvals, width, label = 'N/2 at one center')
kvals = number_of_weights_terminated['central'][::10]
ax.bar(ind+width*2, kvals, width, label = 'N at one center')

ax.set_ylabel('Count')
ax.set_xlabel('Amount of training data [N]')
ax.set_xticks(ind+width)
ax.set_xticklabels([int(i) for i in average_results['total_amount_of_data_intervals'][::10]])


plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0.)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.set_ylim(ylim)

#plt.savefig('lasso_feature_selection_smallerThres_normal.eps', format='eps')
plt.show() 






all_weights = []       
for key, value in average_weights.items():
    for key2 in sorted(value, key=int):
        all_weights += (abs(i) for i in value[key2])
        
print('max', max(all_weights))
print('min', min(all_weights))


#%%
# make excel tables

names = ['interval', 'lower', 'mean', 'upper']

distributed = pd.DataFrame(list(data_for_dist_table), columns = names) 
writer = pd.ExcelWriter('output_distributed_normal.xlsx')
distributed.to_excel(writer, 'Sheet1')
writer.save()

single = pd.DataFrame(list(data_for_single_table), columns = names) 
writer = pd.ExcelWriter('output_single_normal.xlsx')
single.to_excel(writer, 'Sheet1')
writer.save()

central = pd.DataFrame(list(data_for_central_table), columns = names) 
writer = pd.ExcelWriter('output_central_normal.xlsx')
central.to_excel(writer, 'Sheet1')
writer.save()


print('done')
#%%
# make a table out the best selected parameters

with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\differential_privacy_logistic_regression\logistic_results_addaptive_second.json") as f:
    param = json.load(f)

table = []
for key in param:
    table.append([key, param[key]['weight_decay'], param[key]['score']])
    
names = ['interval', 'weight decay', 'score']

table = pd.DataFrame(table, columns = names)
writer = pd.ExcelWriter('optimal_parameters.xlsx')
table.to_excel(writer, 'Sheet1')
writer.save()




#%%

# SET 1 IF THE WEIGHT IS USED OTHERWISE 0
keys = [i for i in range(len(average_weights['distributed']['20']))]
start_values = [0 for i in range(len(average_weights['distributed']['20']))]
weight_used = dict(zip(keys,start_values))
threshold = 10**(-2)
for key, value in average_weights.items():
    for n in value:
        for i in range(len(value[n])):
            if abs(value[n][i]) > threshold:
                weight_used[i] += 1
                
                


#%% The importance coefff
                
# SET 1 IF THE WEIGHT IS USED OTHERWISE 0
keys = [i for i in range(len(average_weights['distributed']['20']))]
start_values = [0 for i in range(len(average_weights['distributed']['20']))]
weight_used = dict(zip(keys,start_values))
threshold = 10**(-2)
for key, value in average_weights.items():
    for n in value:
        for i in range(len(value[n])):
            if abs(value[n][i]) > threshold:
                weight_used[i] += abs(value[n][i])
                
                



#%%
threshold = 10**(-100)
number_of_weights_terminated = defaultdict(list)  
pixels_to_delete = []     
for key, value in average_weights.items():
    for key2 in sorted(value, key=int):
        value2 = value[key2]
        #print(key2)
        number_of_weights_terminated[key].append(sum(1 for i in value2 if abs(i) < threshold))
        
        if key == 'distributed' and len(number_of_weights_terminated['distributed']) == 53: # we know we did 53 models
            pixels_to_delete = [i-1 for i in range(len(value2)) if abs(value2[i]) < threshold] # we do -1 because the bias is not a feature
    #print('NEXT')
        


#%%
train = []
train_names = []
X_train = []
y_train = []
X_train_four_or_nine = []
y_train_four_or_nine = []
X_train_zero_or_one = []
y_train_zero_or_one = []
school = "C:/Users/s161294/OneDrive - Danmarks Tekniske Universitet/"
with open(school + 'mnist_train.csv') as l:
    for i , line in enumerate(l):
        line = line.split(",")
        features = [float(i) for i in line[1:]]
        target = int(line[0]) 
        row = [target] + features
        train.append(row)
        X_train.append(features)
        y_train.append(target)
        if target == 4 or target == 9:
            X_train_four_or_nine.append(features)
            y_train_four_or_nine.append(target)
        elif target == 0 or target ==1:
            X_train_zero_or_one.append(features)
            y_train_zero_or_one.append(target)


#%%
# after looking at the data we know that the first image is a 4 and the second one is 9

# look at number 4
X_train_four_or_nine = np.asarray(X_train_four_or_nine)
img = X_train_four_or_nine[0].reshape(28, 28) 
stacked_img = np.stack((img,)*3, -1)

value2_magnitudes = [abs(val) for val in value2]
best_pixels = zip(value2_magnitudes, [i-1 for i in range(len(value2))]) #-1 because of the bias
best_pixels = sorted(best_pixels, reverse=True)

best_pixels, best_pixels_idx = zip(*best_pixels)

num_best_pixels_to_show = 30

pixl_num = 0
for i, row in enumerate(stacked_img):
    for j in range(len(row)):
        
        if pixl_num in pixels_to_delete:
            stacked_img[i][j] = np.array([1.0, 0.0, 0.0])
        elif pixl_num in best_pixels_idx[:num_best_pixels_to_show]:
            stacked_img[i][j] = np.array([0.0, 1.0, 1.0])
                
        pixl_num += 1
    

plt.imshow(stacked_img)
plt.axis('off')
plt.savefig('fourFeatureSelection.eps', format='eps')
plt.show()
print('alrithy')
# look at the 9
X_train_four_or_nine = np.asarray(X_train_four_or_nine)
img = X_train_four_or_nine[1].reshape(28, 28) 
stacked_img = np.stack((img,)*3, -1)

value2_magnitudes = [abs(val) for val in value2]
best_pixels = zip(value2_magnitudes, [i-1 for i in range(len(value2))]) #-1 because of the bias
best_pixels = sorted(best_pixels, reverse=True)

best_pixels, best_pixels_idx = zip(*best_pixels)

num_best_pixels_to_show = 15

pixl_num = 0
for i, row in enumerate(stacked_img):
    for j in range(len(row)):
        
        if pixl_num in pixels_to_delete:
            stacked_img[i][j] = np.array([1.0, 0.0, 0.0])
        elif pixl_num in best_pixels_idx[:num_best_pixels_to_show]:
            stacked_img[i][j] = np.array([0.0, 1.0, 1.0])
        
                
        pixl_num += 1
    

plt.imshow(stacked_img)
plt.axis('off')
plt.savefig('nineFeatureSelection.eps', format='eps')
plt.show()


#%%


fig = plt.figure(figsize=(14,4))

ax = plt.subplot(111)
   
start = 0
end = 20
plt.subplot(1,2,1)
shape_index = 0    
for key in standard_dev_res:
    limits = []
    if not key == 'total_amount_of_data_intervals':
        for i in range(len(standard_dev_res[key])):
            limit = t_critical * standard_dev_res[key][i] / np.sqrt(num_simulations)
            limits.append(limit)
        print(key, average_results[key][start:end])
        plt.errorbar(average_results['total_amount_of_data_intervals'][start:end],average_results[key][start:end], yerr= limits[start:end],\
                    label = labels[key],fmt=shapes[shape_index],capsize=2, markersize=5)
        shape_index += 1
#plt.set_ylim([0, 1])

#plt.legend(bbox_to_anchor=(1.05, 0.55), loc=2, borderaxespad=0.)
box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

plt.ylabel('Error rate', fontsize = 12)
plt.xlabel('Amount of training data [N]', fontsize = 12)
#plt.savefig('distributed_lasso_noPCA_first20.eps', format = 'eps')

#plt.show()



# ****************** Log axis ***************************************


#fig = plt.figure(figsize=(14,4))
#ax = plt.subplot(111)
shape_index = 0
plt.subplot(1,2,2)
data_for_single_table = {}
data_for_central_table = {}   
# the average and the standard division have the same keys
for key in standard_dev_res:
    upper_limits = []
    lower_limits = []
    limits = []
    if not key == 'total_amount_of_data_intervals':
        for i in range(len(standard_dev_res[key])):
            # 95% confidance interval...
            limits = t_critical * standard_dev_res[key][i] / np.sqrt(num_simulations)
            upper_limit = average_results[key][i] + limits
            lower_limit = average_results[key][i] - limits
            upper_limits.append(upper_limit) 
            lower_limits.append(lower_limit)
        plt.errorbar(average_results['total_amount_of_data_intervals'],average_results[key], yerr= limits,\
            label = labels[key],fmt=shapes[shape_index],capsize=2, markersize=5)
        
        shape_index += 1
        
        if key == 'distributed':
            data_for_dist_table = zip(average_results['total_amount_of_data_intervals'], lower_limits, average_results['distributed'], upper_limits)
        elif key == 'single':
             data_for_single_table = zip(average_results['total_amount_of_data_intervals'], lower_limits, average_results['single'], upper_limits)
        elif key == 'central_all_data':
            data_for_central_table = zip(average_results['total_amount_of_data_intervals'], lower_limits, average_results['central_all_data'], upper_limits)
            
box = ax.get_position()

plt.ylabel('log(Error rate)', fontsize = 12)
plt.xlabel('Amount of training data [N]', fontsize = 12)
plt.yscale('log')

plt.show()








