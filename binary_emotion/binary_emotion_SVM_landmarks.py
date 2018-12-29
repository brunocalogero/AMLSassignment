import time
import math
import os
import cv2
import dlib

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

##################
# list of emotions, you can reduce this to see how your accuracy varies
emotions = ["smile", "no_smile"]
detect_obj = dlib.get_frontal_face_detector()
predict_obj = dlib.shape_predictor("../preprocessing/models/shape_predictor_68_face_landmarks.dat")
##################


# final feature enhance obtain_landmarks function
def obtain_landmarks(frame):
    '''
    https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    returns array of new features found for face
    '''
    detections = detect_obj(frame, 1)
     # For each face detected in given frame
    for k,dots in enumerate(detections):
         # Get landmark coords in image
        shape = predict_obj(frame, dots)
        xlist = []
        ylist = []
        # Store x and y coordinates in respective lists
        for i in range(1,68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        # get means of dots to obtain center of gravity
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)

        # get the distance wrt dots
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        # divide by zero catch, these dots correspond to that of the nosebone and top right midpoint of eyebrow  (29 -> mid nose, 26 -> mid-eyebrow)
        if xlist[26] == xlist[29]:
            noseangle = 0
        else:
            noseangle = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)
         #getting offset for nosebone tilt so that nosebone is vertical in image
        if noseangle < 0:
            noseangle += 90
        else:
            noseangle -= 90

        # setting up the ladmark vectors
        landmarks_vector = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vector.append(x)
            landmarks_vector.append(y)

            # calculate eucledian distance between point and centre of gravity
            mean_eu = np.asarray((ymean,xmean))
            coor_eu = np.asarray((z,w))
            distance = np.linalg.norm(coor_eu-mean_eu)

            # tilted image nosebone based correction, we tilt the point so that the nose bone alignes with the vertical
            relative_angle = (math.atan2(y, x)*360)/(2*math.pi) # (math.atan((z-ymean)/(w-xmean))*180/math.pi) - noseangle
            landmarks_vector.append(distance)
            landmarks_vector.append(relative_angle)

    # error handling if no faces detected
    if len(detections) < 1:
        return False
    else:
        return landmarks_vector

def pull_dataset():
    '''
    Pulls the full dataset (without outliers)
    Used as a helper function to obtain full dataset that will be then split to user
    desired train/validation/test(inference) sets.
    '''
    # lists keep the order
    full_dataset = []
    full_labels = []
    lost_features = []

    # collect labels
    df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
    newdf = df[df.columns[2]]

    # collect pre-processed images and sort them to labels
    for (root, dirs, dat_files) in os.walk('{0}'.format(images_dir)):

        print('starting gatherer')

        for file in dat_files:
            # image grayscaling at import
            img = cv2.imread('{0}/{1}'.format(images_dir, file), grey_scale)
            # image equalisation (to be added by user if equalisation is esteemed to be useful)
            features = obtain_landmarks(img)
            if features is False:
                lost_features.append(file)
            else:
                full_dataset.append(features)
                full_labels.append(int(file[:-4]))

    # only select rows of interest (none outliers) and only keep 'smiling' feature to be evaluated
    full_labels = newdf.loc[full_labels]
    full_labels = full_labels.values.tolist()

    # numpy array conversion
    full_dataset = np.array(full_dataset)
    full_labels = np.array(full_labels)

    return full_dataset, full_labels, lost_features

# RUNNING CODE

full_dataset, full_labels, lost_features = pull_dataset()

print('Sanity Check')
print('full dataset of shape:', full_dataset.shape)
print('full labels of shape:', full_labels.shape)

print('TOTAL NUMBER OF FACES NOT DETECTED WITH OUR LANDMARKS DETECTOR (IN-BUILT, pre-trained model): {0}'.format(len(lost_features)))
# # creating classifier object as an SVM (support vector machine) probabilistic model, you can change this to any other type of classifier
# classifier = SVC(kernel='linear', probability=True, tol=1e-3)

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



# creating dummy SVM classifier for hyperparameterization
classifier = svm.SVC()

n_folds = 5
# choosing different parameter combinations to try
param_grid = {'C': [0.01, 0.1, 1, 10],
              'gamma': [0.0001, 0.003, 0.0037, 0.001, 0.01],
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


# predict using test set
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

# Now predict the value of the test
expected = y_test

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predictions)))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

# plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predictions)))
