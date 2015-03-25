
# Load classification problem
import classification_base
import numpy as np
import sklearn.metrics as skmet

(X, Y, X_val, X_test) = classification_base.load(load_val=False, load_test=False)

def scale(X, eps = 0.001):
	# scale the data points s.t the columns of the feature space
	# (i.e the predictors) are within the range [0, 1]
	return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) - np.min(X, axis = 0) + eps)

def transform_y(Y):
    return Y[:,1]*7 + Y[:,0] # from 0-22

def transform_y_back(Yt):
    Y1 = Yt % 3
    Y2 = np.floor(Yt/7)
    return np.transpose(np.vstack((Y1, Y2)))

def score(gtruth, pred):
    pred = transform_y_back(pred)
    gtruth = transform_y_back(gtruth)
    return classification_base.score(gtruth, pred)
scorefun = skmet.make_scorer(score, greater_is_better=False)

X = scale(X)
#Y = transform_y(Y)
#X_val = scale(X_val)

# ---------

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
import time
import sklearn.grid_search
import sklearn.cross_validation

# perform a grid search on the learning rate, number of
# iterations, and number of components on the RBM and
# C for Logistic Regression
print("SEARCHING RBM + LOGISTIC REGRESSION")
params = {

}
params_logistics = {
    "C": [10.0, 100.0, 1000.0]
    }

# Models we will use
logistic = sklearn.linear_model.LogisticRegression(C=100.0)
rbm1 = BernoulliRBM(learning_rate=0.1, n_iter=40, n_components=50, random_state=0, verbose=True)
rbm2 = BernoulliRBM(learning_rate=0.1, n_iter=40, n_components=40, random_state=0, verbose=True)
rbm3 = BernoulliRBM(learning_rate=0.1, n_iter=40, n_components=30, random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])

#rbm.fit(X_val)
#h = rbm.transform(X)
#h = np.loadtxt('h.txt')

#scores = sklearn.cross_validation.cross_val_score(logistic, h, Y, scoring=scorefun, cv=4)
#print(scores)

# perform a grid search over the parameter
start = time.time()
gs = sklearn.grid_search.GridSearchCV(classifier, params, verbose = 1, scoring=classification_base.scorefun)
gs.fit(X, Y[:,0])
 
# print diagnostic information to the user and grab the
# best model
print("\ndone in %0.3fs" % (time.time() - start))
print("best score: %0.3f" % (gs.best_score_))
print("RBM + LOGISTIC REGRESSION PARAMETERS")
bestParams = gs.best_estimator_.get_params()
 
# loop over the parameters and print each of them out
# so they can be manually set
for p in sorted(params.keys()):
	print("\t %s: %f" % (p, bestParams[p]))