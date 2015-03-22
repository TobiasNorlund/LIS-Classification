# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:38:09 2015

@author: Borislav
"""
# Allows us to create custom scoring functions
import sklearn.metrics as skmet
import sklearn.svm as svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.grid_search as skgs

import numpy as np
import matplotlib.pyplot as plt


MAX_TRAIN_SAMPLES = 5000
""" Read Data """
data_X = np.genfromtxt('project_data/train.csv', delimiter=',',dtype=int)[0:MAX_TRAIN_SAMPLES]
data_Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')[0:MAX_TRAIN_SAMPLES]
#data_X_val = np.genfromtxt('project_data/validate.csv', delimiter=',')
#data_X_test = np.genfromtxt('project_data/test.csv', delimiter=',')

print('Shape of data_X:', data_X.shape)
print('Shape of data_Y:', data_Y.shape)
print('Data loaded sucessfully')

""" Feature Extraction """
num_features = [0]
cat_features = range(9,14)+range(14,32)
features = num_features + cat_features
X = data_X[:,features]
Y = data_Y
#X_val = X_val[:,features]
#X_test = X_test[:,features]

""" Normalization """
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
means[len(num_features):] = 0
stds[len(num_features):] = 1
stds[stds == 0] = 1

X = (X-means)/stds
#X_val = (X_val-means)/stds
#X_test = (X_test - means)/stds

""" Plotting """
# weights_cat_0 = [100,40,70,60,60,70,60,70]
# weights_cat_1 = [60,0 but uneven distribution,0 but,20 but,10 but,0 but,0 but,20 but]
X_hist1 = X[np.where(Y[:,1]==1)]
X_hist2 = X[np.where(Y[:,1]==2)]
X_hist3 = X[np.where(Y[:,1]==3)]
X_hist4 = X[np.where(Y[:,1]==4)]
X_hist5 = X[np.where(Y[:,1]==5)]
X_hist6 = X[np.where(Y[:,1]==6)]
X_hist7 = X[np.where(Y[:,1]==7)]
#plt.plot(Y[:,1],X[:,2],'bo')
#feature = 3
#plt.hist(X_hist1[:,feature],alpha=0.2)
#plt.hist(X_hist2[:,feature],alpha=0.2)
##plt.hist(X_hist3[:,feature])
#plt.hist(X_hist4[:,feature],alpha=0.2)
#plt.hist(X_hist5[:,feature],alpha=0.2)
#plt.hist(X_hist6[:,feature],alpha=0.2)
#plt.hist(X_hist7[:,feature],alpha=0.2)

""" Score Function """
def score_fn(gtruth, pred):
    mis_class = np.isclose(gtruth,pred).astype(int)*(-1)+1
    score = np.sum(mis_class)/(mis_class.shape[0]*mis_class.shape[1])
    print('score: ', score)
    return score
# Define score function
scorefun = skmet.make_scorer(score_fn, greater_is_better=False)

""" Classification """
""" OneVsRest """
svc_clf = svm.SVC()
clf = OneVsRestClassifier(svc_clf)

# Perform gris search
param_grid = {}

grid_search = skgs.GridSearchCV(clf, param_grid, scoring=score_fn, cv=4)
grid_search.fit(X, Y[:,0])
best = grid_search.best_estimator_

print(best)
print('best score =', grid_search.best_score_)
print('SVs: ', best.support_.shape)

