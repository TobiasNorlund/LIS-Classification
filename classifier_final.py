# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:38:09 2015

@author: Borislav
"""
import numpy as np
import matplotlib.pyplot as plt


MAX_TRAIN_SAMPLES = 14514 #14514
""" Read Data """
X = np.genfromtxt('project_data/train.csv', delimiter=',',dtype=int)[0:MAX_TRAIN_SAMPLES]
Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')[0:MAX_TRAIN_SAMPLES]
#X_val = np.genfromtxt('project_data/validate.csv', delimiter=',')
X_test = np.genfromtxt('project_data/test.csv', delimiter=',')

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

""" Normalization """
full_num_features = range(9)
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
means[len(full_num_features):] = 0
stds[len(full_num_features):] = 1
stds[stds == 0] = 1

X = (X-means)/stds
#X_val = (X_val-means)/stds
X_test = (X_test - means)/stds

""" Score Function """
def score(estimator,x_test, y_pred):
    y_test = estimator.predict(x_test)
    score = np.sum(y_test == y_pred)/float(y_test.shape[0])
#    print('score: ', score)
    return score

""" Feature Extraction """
features = range(0,53)
num_features_cat1 = [0,1,2,3,4,5,6,7,8]
cat_features_cat1 = range(9,14)+range(14,53)
features_cat1 = num_features_cat1 + cat_features_cat1

num_features_cat2 = [0,1,2,3,4,5,6,7,8]
cat_features_cat2 = range(9,14)+range(14,53)
features_cat2 = num_features_cat2 + cat_features_cat2

#X_val = X_val[:,features]
X_test = X_test[:,features]

""" Classification """
from sklearn.ensemble import *

clf = {
'extra trees': ExtraTreesClassifier(n_estimators=412),     #0.195 cat_1
'bagging': BaggingClassifier(n_estimators=412)         #0.192 cat_2
}

clf['extra trees'].fit(X[:,features_cat1], Y[:,0])
cat1_score = clf['extra trees'].score(X[:,features_cat1], Y[:,0])
print 'cat_1 score =' , cat1_score

clf['bagging'].fit(X[:,features_cat2], Y[:,1])
cat2_score = clf['bagging'].score(X[:,features_cat2], Y[:,1])
print 'cat_2 score =' , cat2_score

print 'best score =', (cat1_score+cat2_score)/2

""" Predict validation set """
#Y_val1 = clf['extra trees'].predict(X_val)
#Y_val2 = clf['bagging'].predict(X_val)
#np.savetxt('result_validation_final.txt', np.transpose(np.vstack((Y_val1, Y_val2))), fmt='%i', delimiter=',')
#print 'Saved Validation result!'

""" Predict test set """
Y_test1 = clf['extra trees'].predict(X_test)
Y_test2 = clf['bagging'].predict(X_test)
np.savetxt('result_test_final.txt', np.transpose(np.vstack((Y_test1, Y_test2))), fmt='%i', delimiter=',')
print 'Saved Test result!'