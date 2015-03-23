# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:38:09 2015

@author: Borislav
"""
# Allows us to create custom scoring functions
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.svm as svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.grid_search as skgs

import numpy as np
import matplotlib.pyplot as plt


MAX_TRAIN_SAMPLES = 10000 #14514
""" Read Data """
data_X = np.genfromtxt('project_data/train.csv', delimiter=',',dtype=int)[0:MAX_TRAIN_SAMPLES]
data_Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')[0:MAX_TRAIN_SAMPLES]
#data_X_val = np.genfromtxt('project_data/validate.csv', delimiter=',')
#data_X_test = np.genfromtxt('project_data/test.csv', delimiter=',')

print('Shape of data_X:', data_X.shape)
print('Shape of data_Y:', data_Y.shape)
print('Data loaded sucessfully')

""" Feature Extraction """
num_features = [0,1,2,3,4,5,6,7,8]
cat_features = range(9,14)+range(14,53)
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
#plt.hist(Y[:,1])


""" Score Function """
def score(estimator,x_test, y_pred):
    y_test = estimator.predict(x_test)
    score = np.sum(y_test != y_pred)/float(2*y_test.shape[0])
#    print('score: ', score)
    return score
scorefun = skmet.make_scorer(score, greater_is_better=False)

""" Classification """
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.tree import *

""" OneVsRest """
#clf = svm.SVC(kernel='rbf')
#clf = svm.SVC(kernel='rbf',class_weight={0: 2.7,1: 6.5,2: 0.9})
#clf = sklin.SGDClassifier()
""" Ensemble """
clf = [
#BaggingClassifier(),            #0.192
#ExtraTreesClassifier(),         #0.195
#RandomForestClassifier(),       #0.196
#KNeighborsClassifier(),         #0.241
#svm.SVC(kernel='rbf'),          #0.257
#RidgeClassifier(),              #0.281
#AdaBoostClassifier(),           #bad
#GradientBoostingClassifier(),   #slow
#OneVsRestClassifier(BaggingClassifier()),   #0.198
#OneVsOneClassifier(BaggingClassifier()),    #0.212
#DecisionTreeClassifier(),       #0.270
#OneVsRestClassifier(LogisticRegression()),  #0.260
BaggingClassifier()            #0.192
]


""" Grid Search """
for i in range(len(clf)):
    param_grid = {}
    grid_search = skgs.GridSearchCV(clf[i], param_grid,scoring=score, cv=4)
    grid_search.fit(X, Y[:,0])
    best1_estm = grid_search.best_estimator_
    best1_score = grid_search.best_score_
    grid_search.fit(X, Y[:,1])
    best2_estm = grid_search.best_estimator_
    best2_score = grid_search.best_score_
    print('best score =', best1_score+best2_score)

#""" Predict validation set """
#Y_val1 = best1.predict(X_val)
#Y_val2 = best2.predict(X_val)
#np.savetxt('result_validation_1.txt', np.transpose(np.vstack((Y_val1, Y_val2))), fmt='%i', delimiter=',')


