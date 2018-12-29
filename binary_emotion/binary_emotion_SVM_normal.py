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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'
# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1

# Import whole pre-processed Dataset (training and test)

# lists keep the order
full_dataset = []
full_labels = []

# collect labels
df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
newdf = df[df.columns[2]]

# collect pre-processed images and sort them to labels
for (root, dirs, dat_files) in os.walk('{0}'.format(images_dir)):

    for file in dat_files:
        # image grayscaling at import
        img = cv2.imread('{0}/{1}'.format(images_dir, file), grey_scale)
        # image equalisation
        # rescaling image (you can use cv2)
        res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        # turn to float for zero centering
        res = res.astype(float)
        full_dataset.append(res)
        full_labels.append(int(file[:-4]))

# only select rows of interest (none outliers) and only keep 'smiling' feature to be evaluated
full_labels = newdf.loc[full_labels]
full_labels = full_labels.values.tolist()

# now both of our dataset and labels are ordered

# numpy array conversion
full_dataset = np.array(full_dataset)
full_labels = np.array(full_labels)

print('full dataset of shape:', full_dataset.shape)
print('full labels of shape:', full_labels.shape)

# plt.imshow(full_dataset[0])
full_dataset[0]


# Reshuffling data (for extra randomness)
X_data, Y_data = shuffle(full_dataset, full_labels, random_state=0)

print('X_data of shape:', X_data.shape)
print('Y_data of shape:', Y_data.shape)

# perform train and test split (random state set to 1 to ensure same distribution accross different sets)
# this split is obviously case specific! but cross validation allows us to avoid over-fitting so lets make sure we have a validation set ready.
# Since the dataset is not extrememly large i'll be using a 60/20/20 split, meaning more or less 1000 validation and test examples and 3000 training examples, to be tested: 75/10/15
# in this case we are a little less concerned since we are evaluating smiles which are present in every case, unlike glasses
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# sanity check
print('X_train of shape:', X_train.shape)
print('y_train of shape:', y_train.shape)
print('X_val of shape:', X_val.shape)
print('y_val of shape:', y_val.shape)
print('X_test of shape:', X_test.shape)
print('y_test of shape:', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((128,128,3)).astype('uint8')) # visualize the mean image
plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image


# creating dummy SVM classifier for hyperparameterization
classifier = svm.SVC()

n_folds = 5
# choosing different parameter combinations to try
param_grid = {'C': [0.01, 0.1, 1, 10],
              'gamma': [0.00002, 0.0001, 0.001, 0.01],
              'kernel': ['rbf', 'linear'],
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


# predict using validation set
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
