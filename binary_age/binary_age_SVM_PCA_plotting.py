import time
import os
import cv2
import csv

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,  GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score

# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'
# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1
label = 'young'

# learning curve pulled from the web (implements cross validation, fitting(training) and inference)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 15)):
    """
    Generate a simple plot of the test and training learning curve.
    StratifiedKFold Validation is used.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, test_scores

print('loading train and test PCA based sets...')
# import PCA data (no one hot encoding)
data_train = np.load('../pca_dataset/pca_{0}_train.npz'.format(label))
X_train = data_train['name1']
y_train = data_train['name2']

data_test = np.load('../pca_dataset/pca_{0}_test.npz'.format(label))
X_test = data_test['name1']
y_test = data_test['name2']

# test that loading is working fine
data_train_index = np.load('../pca_dataset/pca_{0}_train_index.npz'.format(label))
X_train_index = data_train_index['name1']

data_test_index = np.load('../pca_dataset/pca_{0}_test_index.npz'.format(label))
X_test_index = data_test_index['name1']

# sanity check
print('X_train of shape:', X_train.shape)
print('y_train of shape:', y_train.shape)
print('X_test of shape:', X_test.shape)
print('y_test of shape:', y_test.shape)
print('X_test of shape:', X_train_index.shape)
print('y_test of shape:', X_test_index.shape)

title = "Learning Curves for {0} binary Linear Kernel SVM (250 First components (PCA) from 256x256 RGB data)".format(label)
# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start cross val with SVM linear {}'.format(str(start_time)))

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=0)
estimator = svm.SVC(kernel='poly', C=0.01, gamma=0.004, cache_size=1500)
plt, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, test_scores = plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

end_time = dt.datetime.now()
print('Stop cross val {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print('Elapsed cross val time {}'.format(str(elapsed_time)))

# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start learning SVM linear {}'.format(str(start_time)))

estimator.fit(X_train, y_train)

end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning time {}'.format(str(elapsed_time)))

# inference: predict using test set
predictions = estimator.predict(X_test)
accuracy_scr = accuracy_score(y_test, predictions)
print(accuracy_scr)

# Now predict the value of the test
expected = y_test

print("Classification report for classifier %s:\n%s\n"
      % (estimator, metrics.classification_report(expected, predictions)))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

plt.show()

print('creating csv for inference')
with open('./result_logs/inference.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow([accuracy_scr, '.'])
    for i, file in enumerate(X_test_index):
        filewriter.writerow(['{0}.png'.format(file), predictions[i]])
