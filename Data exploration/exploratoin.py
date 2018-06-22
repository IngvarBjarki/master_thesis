# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:37:54 2018

@author: s161294
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
sns.set_context("paper")

# load the data and put it into a pandas data frame
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


test = []
test_names = []
with open(school + 'mnist_test.csv') as l:
    for i , line in enumerate(l):
        line = line.split(",")
        row = [int(line[0])] + [float(i) for i in line[1:]]
        test.append(row)



names = ['y'] + ['X{}'.format(i) for i in range(784)] # we know there are 784 features
df_train = pd.DataFrame(train, columns = names )
df_test = pd.DataFrame(test, columns = names )

#%%
# plot the images to see how they look

images_to_show = list(range(10))
for i, target in enumerate(y_train):
    if target in images_to_show:
        sns.set_style("ticks")
        pxles = np.array(X_train[i])
        pxles = pxles.reshape((28, 28))        
        plt.imshow(pxles, cmap = 'gray')
        plt.axis('off')
        plt.savefig('number_{}.eps'.format(target), format='eps')
        plt.show()
        images_to_show.remove(target)

#%%
# plot the images to see how they look into a subplot
fig = plt.figure(figsize=(5,2))
counter = 1
images_to_show = list(range(10))
for i, target in enumerate(y_train):
    if target in images_to_show:
        sns.set_style("ticks")
        pixles = np.array(X_train[i])
        pixles = pixles.reshape((28, 28))
        ax = fig.add_subplot(2, 5, 1 + target)        
        ax.imshow(pixles, cmap = 'gray')
        plt.axis('off')
        
        images_to_show.remove(target)
fig.tight_layout()
plt.show()
        
        
#%%
# find out how many data points are in each category of the targets
print('Categories in train set')
print(df_train['y'].value_counts())
print('\n Categories in test set')
print(df_test['y'].value_counts())

#%%
# create histogram of the number of times a value comes upp in pandas..
###! spa i pixlunum!!!!
sns.set_style("darkgrid")
sns.distplot(np.asarray(X_train).flatten(), color =  sns.hls_palette(8, l=.3, s=.8)[0])
plt.title('The density of pixels color values')
plt.xlabel('color value')
plt.ylabel('ratio of pixels in each bin')
plt.savefig('density_pixel_colors.png', format='png')
plt.show()
plt.hist(np.asarray(X_train).flatten(), bins = 254)
plt.show()

#%%
# calculate the eculution distance from the mean
sns.set_style("darkgrid")
colors = ['#e6194b',
          '#0082c8',
          '#d2f53c',
          '#3cb44b',
          '#f032e6',
          '#911eb4',
          '#46f0f0',
          '#f58231', 
          '#008080',
          '#ffe119']
# find all the values that have specifict value..
data_eculidian_distance = []
for digit in range(10):
    df_digit = df_train.loc[df_train['y'] == digit]
    means = np.array(df_digit[['X{}'.format(i) for i in range(784)]].mean())
    digit_distance = (df_digit[['X{}'.format(i) for i in range(784)]].sub(means)).pow(2).sum(1).pow(0.5)
    data_eculidian_distance.append(digit_distance.tolist())

sns.boxplot(data = data_eculidian_distance)
plt.xlabel('digits')
plt.ylabel('Eculidian distance')
plt.title('Eulidian distance to typical digit')
plt.show()

sns.violinplot(data = data_eculidian_distance)
plt.xlabel('digits')
plt.ylabel('Eculidian distance')
plt.title('Eulidian distance to typical digit')

#plt.savefig('violinplot_eculidian_dist.eps', format='eps')
plt.show()



#%%
# find the outleirs and plot them
num_bad_images = 5
counter = 1
fig = plt.figure(figsize=(num_bad_images,10))

for digit in range(10):
    for i in range(num_bad_images):
        df_digit = df_train.loc[df_train['y'] == digit]
        index = data_eculidian_distance[digit].index(max(data_eculidian_distance[digit]))
        num = np.asarray(df_digit.iloc[index][1:])
        num = num.reshape((28, 28))
        print('digit {}'.format(digit))

        ax = fig.add_subplot(10, num_bad_images, counter)
        ax.imshow(num, cmap = 'gray')
        plt.axis('off')
        counter +=1

        del data_eculidian_distance[digit][index]
fig.tight_layout()
plt.savefig('bad_images.eps', format='eps')
plt.show()

#%%
# lets do PCA
sns.set_style("darkgrid")


colors = ['#e6194b',
          '#0082c8',
          '#d2f53c',
          '#3cb44b',
          '#f032e6',
          '#911eb4',
          '#46f0f0',
          '#f58231', 
          '#008080',
          '#ffe119']
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)

# visulize all attributes in the data set
num_components = 2
pca = PCA(n_components = num_components)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
print('var', sum(pca.explained_variance_ratio_)) 

total_explained_variance = sum(pca.explained_variance_ratio_)

train_pca = pca.transform(X_train) # make this more pretty l8ter
records, attributes = np.shape(train_pca)
train_pca_ones = np.ones((records, attributes + 1))
train_pca_ones[:,1:] = train_pca
train_pca_ones[:,0] = y_train



names = ['y'] # we know y is the first one
for i in range(num_components):
    names.append('pca{}'.format(i+1))

df_pca = pd.DataFrame(train_pca_ones, columns = names)


sns.lmplot("pca1", "pca2", data=df_pca, hue='y', fit_reg=False)
#plt.title('explained variance = {}'.format(sum(pca.explained_variance_ratio_)))
plt.savefig('pca_plot_all.eps', format='eps')
plt.show()
#%%
# visulize only 4 and 9
scalar = StandardScaler()
X_train_four_or_nine  = scalar.fit_transform(X_train_four_or_nine )

num_components = 2
pca = PCA(n_components = num_components)
pca.fit(X_train_four_or_nine)
print(pca.explained_variance_ratio_)
print('var', sum(pca.explained_variance_ratio_)) 

train_pca = pca.transform(X_train_four_or_nine) # make this more pretty l8ter
records, attributes = np.shape(train_pca)
train_pca_ones = np.ones((records, attributes + 1))
train_pca_ones[:,1:] = train_pca
train_pca_ones[:,0] = y_train_four_or_nine



names = ['y'] # we know y is the first one
for i in range(num_components):
    names.append('pca{}'.format(i+1))

df_pca = pd.DataFrame(train_pca_ones, columns = names)


sns.lmplot("pca1", "pca2", data=df_pca, hue='y', fit_reg=False, palette = [sns.color_palette()[4], sns.color_palette()[9]])
#plt.title('explained variance = {}'.format(sum(pca.explained_variance_ratio_)))
plt.savefig('pca_plot_four_and_nine.eps', format='eps')
plt.show()

#%%
# visulize only 0 and 1

num_components = 2

#X_train_zero_or_one = normalize(X_train_zero_or_one)
scalar = StandardScaler()
X_train_zero_or_one = scalar.fit_transform(X_train_zero_or_one)

pca = PCA(n_components = num_components)
pca.fit(X_train_zero_or_one)
print(pca.explained_variance_ratio_)
print('var', sum(pca.explained_variance_ratio_)) 

train_pca = pca.transform(X_train_zero_or_one) # make this more pretty l8ter
records, attributes = np.shape(train_pca)
train_pca_ones = np.ones((records, attributes + 1))
train_pca_ones[:,1:] = train_pca
train_pca_ones[:,0] = y_train_zero_or_one



names = ['y'] # we know y is the first one
for i in range(num_components):
    names.append('pca{}'.format(i+1))

df_pca = pd.DataFrame(train_pca_ones, columns = names)


sns.lmplot("pca1", "pca2", data=df_pca, hue='y', fit_reg=False, palette = [sns.color_palette()[0], sns.color_palette()[1]])
#plt.title('explained variance = {}'.format(sum(pca.explained_variance_ratio_)))
plt.savefig('pca_plot_zero_and_one.eps', format='eps')
plt.show()





#%%
# explained variance 
fig = plt.figure(figsize=(5,3))
ax = plt.subplot(111)

pca = PCA()
pca.fit(X_train_zero_or_one)
explained_var = np.cumsum(pca.explained_variance_ratio_)
ax.plot(range(1, len(explained_var) + 1), explained_var, label = '0 and 1')



pca = PCA()
pca.fit(X_train_four_or_nine)
explained_var = np.cumsum(pca.explained_variance_ratio_)
ax.plot(range(1, len(explained_var) + 1), explained_var, label = '4 and 9', color = sns.color_palette()[3])

pca = PCA()
pca.fit(X_train)
explained_var = np.cumsum(pca.explained_variance_ratio_)
ax.plot(range(1, len(explained_var) + 1), explained_var, label = 'All numbers', color = sns.color_palette()[2])


plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.title('Explained variance')
plt.yticks(np.arange(0.0, 1.1, 0.1))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
plt.savefig('explained_variance2.eps',  format='eps')
#fig.savefig('explained_variance.eps',  format='eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

#%% how much variance dose the first 100 principal componenets account for

num_components = 100
pca = PCA(n_components = num_components)
pca.fit(X_train_four_or_nine)
print(pca.explained_variance_ratio_)
print('var', sum(pca.explained_variance_ratio_)) 






