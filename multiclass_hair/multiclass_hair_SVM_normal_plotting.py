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
from sklearn.model_selection import train_test_split,  GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score

# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'
# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1

# Import whole pre-processed Dataset (training and test)

def pull_dataset():
    '''
    Pulls the full dataset (without outliers).
    Used as a helper function to obtain full dataset that will be then split to user
    desired train/validation/test(inference) sets.
    The examples are shuffled.
    '''

    # lists keep the order
    full_dataset = []
    full_labels = []

    # collect labels
    df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
    newdf = df['hair_color']

    # collect pre-processed images and sort them to labels
    for (root, dirs, dat_files) in os.walk('{0}'.format(images_dir)):

        for file in dat_files:

            int_file = int(file[:-4])
            # removed -1 labeled images for hair colour
            if df.loc[int_file, 'hair_color'] != -1:
                # image grayscaling at import
                img = cv2.imread('{0}/{1}'.format(images_dir, file), grey_scale)
                # image equalisation (to be inserted if interesting)
                # rescaling image
                res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
                # turn to float for zero centering
                res = res.astype(float)
                full_dataset.append(res)
                full_labels.append(int_file)

    # only select rows of interest (none outliers) and only keep 'hair_color' feature to be evaluated (removed -1 labeled images for hair colour)
    full_labels = newdf.loc[full_labels]
    full_labels = full_labels.values.tolist()

    # now both of our dataset and labels are ordered

    # numpy array conversion
    full_dataset = np.array(full_dataset)
    full_labels = np.array(full_labels)

    print('full dataset of shape:', full_dataset.shape)
    print('full labels of shape:', full_labels.shape)

    # Reshuffling data (for extra randomness)
    X_data, Y_data = shuffle(full_dataset, full_labels, random_state=0)

    print('X_data of shape:', X_data.shape)
    print('Y_data of shape:', Y_data.shape)

    return X_data, Y_data

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


# import data (no one hot encoding)
X_data, y_data = pull_dataset()


# perform train and test split (random state set to 1 to ensure same distribution accross different sets)
# this split is obviously case specific! but cross validation allows us to avoid over-fitting so lets make sure we have a validation set ready.
# Since the dataset is not extrememly large i'll be using a 60/20 split, the 60 will be further split during cross_validation
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)


# X_train = X_train[:500]
# X_test = X_test[:200]
# y_train = y_train[:500]
# y_test = y_test[:200]

# sanity check
print('X_train of shape:', X_train.shape)
print('y_train of shape:', y_train.shape)
print('X_test of shape:', X_test.shape)
print('y_test of shape:', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# sanity check
print('X_train of shape:', X_train.shape)
print('X_test of shape:', X_test.shape)

# Preprocessing: subtract the mean image

# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)

# second: subtract the mean image from train and test data
X_train -= mean_image
X_test -= mean_image

# sanity check
print('X_train of shape:', X_train.shape)
print('X_test of shape:', X_test.shape)


title = "Learning Curves for Multi-class Hair Linear Kernel SVM (zero-centered 128x128 RGB data)"
# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start cross val with SVM linear {}'.format(str(start_time)))

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=0)
estimator = svm.SVC(kernel='linear', C=0.01)
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
print(accuracy_score(y_test, predictions))

# Now predict the value of the test
expected = y_test

print("Classification report for classifier %s:\n%s\n"
      % (estimator, metrics.classification_report(expected, predictions)))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

plt.show()
