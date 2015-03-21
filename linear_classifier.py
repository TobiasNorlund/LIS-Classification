
# Load classification problem
import classification_base

(X, Y, X_val, X_test) = classification_base.load(load_val=False, load_test=False)

# ---------

import sklearn.linear_model as sklin
from sklearn.multiclass import OneVsRestClassifier
import sklearn.grid_search as skgs
import numpy as np

# Create linear classifier
base_classifier1 = sklin.SGDClassifier('perceptron', penalty=None)
base_classifier2 = sklin.SGDClassifier('perceptron', penalty=None)

classifier1 = OneVsRestClassifier(base_classifier1)
classifier2 = OneVsRestClassifier(base_classifier2)

# Fit classifier
param_grid = { }

grid_search1 = skgs.GridSearchCV(classifier1, param_grid, scoring=classification_base.scorefun, cv=4)
grid_search1.fit(X, Y[:,0])

grid_search2 = skgs.GridSearchCV(classifier2, param_grid, scoring=classification_base.scorefun, cv=4)
grid_search2.fit(X, Y[:,1])

best1 = grid_search1.best_estimator_
best2 = grid_search2.best_estimator_

print('best score =', -grid_search2.best_score_ + (-grid_search1.best_score_))


# Predict validation set
#Y_val1 = best1.predict(X_val)
#Y_val2 = best2.predict(X_val)
#np.savetxt('result_validation.txt', np.transpose(np.vstack((Y_val1, Y_val2))), fmt='%i', delimiter=',')

# Predict test set
#Y_test = best.predict(X_test)
#np.savetxt('result_test.txt', Y_test)
