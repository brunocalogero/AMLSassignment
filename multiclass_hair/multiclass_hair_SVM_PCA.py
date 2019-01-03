import time
import os
import cv2

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,  GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, accuracy_score

# setting user chosen vars
labels_filename = 'attribute_list.csv'
# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1

# import data (no one hot encoding)
data_train = np.load('../pca_dataset/pca_train.npz')
X_train = data_train['name1']
y_train = data_train['name2']

data_test = np.load('../pca_dataset/pca_test.npz')
X_test = data_test['name1']
y_test = data_test['name2']

# sanity check
print('X_train of shape:', X_train.shape)
print('y_train of shape:', y_train.shape)
print('X_test of shape:', X_test.shape)
print('y_test of shape:', y_test.shape)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# sanity check
print('X_train of shape:', X_train.shape)
print('y_train of shape:', y_train.shape)
print('X_val of shape:', X_val.shape)
print('y_val of shape:', y_val.shape)
print('X_test of shape:', X_test.shape)
print('y_test of shape:', y_test.shape)


# creating dummy SVM classifier for hyperparameterization
classifier = svm.SVC()

n_folds = 6
# choosing different parameter combinations to try
param_grid = {'C': [0.01, 0.1, 1, 10],
              'gamma': [0.004, 0.001, 0.01, 0.1],
              'kernel': ['rbf', 'linear', 'poly'],
             }

# type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)


# run grid search
start_time = dt.datetime.now()
print('Start grid search at {}'.format(str(start_time)))

grid_search = GridSearchCV(classifier, param_grid, cv=n_folds, scoring=acc_scorer, n_jobs=4)
grid_obj = grid_search.fit(X_val, y_val)
# get grid search results
print(grid_obj.cv_results_)

# set the best classifier found for rbf
clf = grid_obj.best_estimator_
print(clf)
end_time = dt.datetime.now()
print('Stop grid search {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed grid search time {}'.format(str(elapsed_time)))


# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start learning with best params at {}'.format(str(start_time)))

clf.fit(X_train, y_train)

end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning time {}'.format(str(elapsed_time)))

# predict using test set
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

# Now predict the value of the test
expected = y_test

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predictions)))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

# plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predictions)))
