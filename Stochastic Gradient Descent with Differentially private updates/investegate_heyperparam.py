# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:29:42 2018

@author: Ingvar
"""
#%%
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

with open(r"C:\Users\s161294\OneDrive - Danmarks Tekniske Universitet\Thesis\SGD\parameters.json") as f:
    params = json.load(f)
    
colors = sns.color_palette('Set1', n_colors = 9) + [(1.0, 191/255, 0.0)] + sns.color_palette('Set2', n_colors = 3)[0:3:2]

colors = [sns.color_palette('Set1', n_colors = 9)[-4]] + sns.color_palette('Set1', n_colors = 9)[-2:] + [(1.0, 191/255, 0.0)]


colors = [(0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
          (1.0, 0.4980392156862745, 0.0),
          (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
          (1.0, 191/255, 0.0)
]


sns.set_palette(colors)
sns.set_style('darkgrid')
# get the data

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
batch_sizes = {}
weight_decays = {}
for epsilon in params:
    print('epsilon', epsilon)
    batch_sizes[epsilon] = []
    weight_decays[epsilon] = []
    for n in params[epsilon]:
        batch_size = params[epsilon][n]['parameters'][0]
        weight_decay = params[epsilon][n]['parameters'][1]
        
        batch_sizes[epsilon].append((eval(n), batch_size))
        weight_decays[epsilon].append((eval(n), weight_decay))
    
    amount_data, batcher = zip(*sorted(batch_sizes[epsilon]))
    amount_data_b, weight_decaysn = zip(*sorted(weight_decays[epsilon]))
    print('===========================================')
    plt.plot(amount_data, batcher, 'o--')
    plt.xlabel('N')
    plt.ylabel('batch')
    plt.show()
    
    plt.plot(amount_data, weight_decaysn, 'o--', color = 'green')
    plt.xlabel('N')
    plt.ylabel('w-decay')
    plt.show()
    
    plt.plot(batcher, weight_decaysn, 'o', color = 'orange')
    plt.xlabel('batch')
    plt.ylabel('w-decay')
    plt.show()
    
    print('===========================================')
    
    
    
#%%
fig = plt.figure(figsize=(7, 4))
ax = plt.subplot(111)
batch_sizes = {}
color = 0
for epsilon in params:
    batch_sizes[epsilon] = []
    weight_decays[epsilon] = []
    for n in params[epsilon]:
        batch_size = params[epsilon][n]['parameters'][0]
        batch_sizes[epsilon].append((eval(n), batch_size))
    amount_data, batcher = zip(*sorted(batch_sizes[epsilon]))
    print(epsilon, batcher[-1] )

    
    plt.plot(amount_data, batcher, 'o', label = get_label[epsilon], color = colors[color])
    color += 1
plt.xlabel('Amount of training data [N]',fontsize = 12)
plt.ylabel('$b$', fontsize = 12)
plt.legend(bbox_to_anchor=(1.05, 0.65), loc=2, borderaxespad=0., fontsize = 12)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.savefig('batches_training.eps', format = 'eps')
plt.show()
    



fig = plt.figure(figsize=(7, 4))
ax = plt.subplot(111)
weight_decays = {}
for epsilon in params:
    batch_sizes[epsilon] = []
    weight_decays[epsilon] = []
    for n in params[epsilon]:
        batch_size = params[epsilon][n]['parameters'][0]
        weight_decay = params[epsilon][n]['parameters'][1]
        
        weight_decays[epsilon].append((eval(n), weight_decay))
    
    amount_data_b, weight_decaysn = zip(*sorted(weight_decays[epsilon]))
    print(epsilon, weight_decaysn[-1] )
    plt.plot(amount_data, weight_decaysn, 'o', label = get_label[epsilon])
plt.xlabel('Amount of training data [N]', fontsize = 12)
plt.ylabel('$\lambda$', fontsize = 12)
plt.legend(bbox_to_anchor=(1.05, 0.65), loc=2, borderaxespad=0.,fontsize = 12)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.savefig('weight_decays_tuning.eps', format = 'eps')
plt.show()

#%%
fig = plt.figure(figsize=(7, 4))
ax = plt.subplot(111)
batch_sizes = {}
weight_decays = {}
for epsilon in params:
    batch_sizes[epsilon] = []
    weight_decays[epsilon] = []
    for n in params[epsilon]:
        batch_size = params[epsilon][n]['parameters'][0]
        weight_decay = params[epsilon][n]['parameters'][1]
        
        
        batch_sizes[epsilon].append((eval(n), batch_size))
        weight_decays[epsilon].append((eval(n), weight_decay))
    
    amount_data, batcher = zip(*sorted(batch_sizes[epsilon]))
    amount_data_b, weight_decaysn = zip(*sorted(weight_decays[epsilon]))
    
    c = Counter(zip(batcher, weight_decaysn))
    s = [20*c[xx, yy] for xx, yy in zip(batcher, weight_decaysn)]
    
    plt.scatter(batcher , weight_decaysn, s = s, label = get_label[epsilon], alpha = 0.6)
plt.xlabel('batch size')
plt.ylabel('weight decay')
plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.savefig('weightDecay_learningRate_tuning.png')
plt.show()

#%%
batch_sizes = []
weight_decays = []
nums = []
for epsilon in params:
    for n in params[epsilon]:
        batch_sizes.append(params[epsilon][n]['parameters'][0])
        weight_decays.append(params[epsilon][n]['parameters'][1])
        nums.append(int(n))
        
b = Counter(batch_sizes).most_common(len(set(batch_sizes)))
print('batches', b)
print('-----------------------------------------------')
x, y = zip(*b)
plt.bar(x, y, width=5)
plt.xlabel('batch size')
plt.ylabel('Count')
plt.savefig('batchSizeCounts.png')
plt.show()
w = Counter(weight_decays).most_common(len(set(weight_decays)))
x, y = zip(*w)
plt.bar(x, y, color = [i for i in 'rgbkymc'], width=0.03)
plt.xlabel('weight decay')
plt.ylabel('Count')
plt.savefig('weightDecayCount.png')
plt.show()
print('w-decay', w)
print('-----------------------------------------------')

a = Counter(list(zip(nums, weight_decays))).most_common(len(set(list(zip(nums, weight_decays)))))

print(sorted(a))
#%%
fig = plt.figure(figsize=(22, 8))
ax = plt.subplot(111)
pairs = []
for epsilon in params:
    for n in params[epsilon]:
        batch = params[epsilon][n]['parameters'][0]
        decay = params[epsilon][n]['parameters'][1]
        pairs.append((batch, decay))
        
        
pairs = Counter(pairs).most_common(len(set(pairs)))
print('-----------------------------------------------')
x, y = zip(*sorted(pairs, key=lambda element: (element[0], element[1])))
plt.bar([i for i in range(0, 2*len(x), 2)], y, width=1)
plt.xticks([i for i in range(0, 2*len(x), 2)], x, rotation = 90)
plt.xlabel('batch size & weight decay')
plt.ylabel('Count')
plt.savefig('pairsCount.png')
plt.show()




fig = plt.figure(figsize=(7, 4))
ax = plt.subplot(111)
batch_sizes = {}
weight_decays = {}
for epsilon in params:
    batch_sizes[epsilon] = []
    weight_decays[epsilon] = []
    for n in params[epsilon]:
        batch_size = params[epsilon][n]['parameters'][0]
        weight_decay = params[epsilon][n]['parameters'][1]
        
        
        batch_sizes[epsilon].append((eval(n), batch_size))
        weight_decays[epsilon].append((eval(n), weight_decay))
    
    amount_data, batcher = zip(*sorted(batch_sizes[epsilon]))
    amount_data_b, weight_decaysn = zip(*sorted(weight_decays[epsilon]))
    
    c = Counter(zip(batcher, weight_decaysn))
    s = [20*c[xx, yy] for xx, yy in zip(batcher, weight_decaysn)]
    
    plt.scatter(batcher , weight_decaysn, s = s, label = get_label[epsilon])
plt.xlabel('$b$', fontsize = 12)
plt.ylabel('$\lambda$', fontsize = 12)
plt.legend(bbox_to_anchor=(1.05, 0.65), loc=2, borderaxespad=0., fontsize = 12)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.savefig('weightDecay_learningRate_tuning.eps', format = 'eps')
plt.show()



