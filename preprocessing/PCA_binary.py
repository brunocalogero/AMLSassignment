'''
This code can be made much more efficient
'''

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'
# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1


# function to transform each image from dataset to principle PCA components
def pca_transform(X_train, X_test):
    '''
    Transforms each image from dataset to principle PCA components
    '''
    # standardizing dataset
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)

    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA decomposition
    # Make an instance of the Model (there are 589 components for 95%, we reduce to only keep the main ones)
    pca = PCA(250)
    pca.fit(X_train)

    # check how many components retained
    print(pca.n_components_)

    # Apply the mapping (transform) to both the training set and the test set.
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    print('X_train after transformation:', X_train)
    print('X_test after transformation:', X_test)

    return X_train, X_test
# Import whole pre-processed Dataset (training and test)
def pull_dataset(label):
    '''
    Pulls the full dataset (without outliers).
    Used as a helper function to obtain full dataset that will be then split to user
    desired train/validation/test(inference) sets.
    The examples are shuffled.
    '''

    # lists keep the order
    full_dataset = []
    full_labels = []
    full_dataset_name = []

    # collect labels
    df = pd.read_csv(labels_filename, skiprows=1, index_col='file_name')
    newdf = df[label]

    # collect pre-processed images and sort them to labels
    for (root, dirs, dat_files) in os.walk('{0}'.format(images_dir)):

        for file in dat_files:

            int_file = int(file[:-4])
            # removed -1 labeled images for hair colour
                # image grayscaling at import
            img = cv2.imread('{0}/{1}'.format(images_dir, file), grey_scale)
            # image equalisation (to be inserted if interesting)
            # rescaling image
            # res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
            # turn to float for zero centering
            # img = img.astype(float)
            full_dataset.append(img)
            full_dataset_name.append(int_file)
            full_labels.append(int_file)

    # only select rows of interest (none outliers) and only keep 'hair_color' feature to be evaluated (removed -1 labeled images for hair colour)
    full_labels = newdf.loc[full_labels]
    full_labels = full_labels.values.tolist()

    # now both of our dataset and labels are ordered

    # numpy array conversion
    full_dataset_name = np.array(full_dataset_name)
    full_dataset = np.array(full_dataset)
    full_labels = np.array(full_labels)

    print('full dataset of shape:', full_dataset.shape)
    print('full full_dataset_name of shape:', full_dataset_name.shape)
    print('full labels of shape:', full_labels.shape)


    # Reshuffling data (for extra randomness) (notice the same random state)
    X_data_index, Y_data_index = shuffle(full_dataset_name, full_labels, random_state=1)
    X_data, Y_data = shuffle(full_dataset, full_labels, random_state=1)

    print('X_data of shape:', X_data.shape)
    print('Y_data of shape:', Y_data.shape)
    print('X_data_index of shape:', X_data_index.shape)
    print('Y_data_index of shape:', Y_data_index.shape)

    # Sanity Check: testing that y_data_index has been shuffled in the same way as y_data to make sure all the data is associated correctly
    print('Sanity Check: shuffling and indexes of images for future inference')
    print(len(Y_data_index) == len(Y_data))
    print(any(Y_data_index == Y_data))


    # perform train and test split (random state set to 1 to ensure same distribution accross different sets)
    # this split is obviously case specific! but cross validation allows us to avoid over-fitting so lets make sure we have a validation set ready.
    # Since the dataset is not extrememly large i'll be using a 60/20/20 split, meaning more or less 1000 validation and test examples and 3000 training examples, to be tested: 75/10/15
    # in this case we are a little less concerned since we are evaluating smiles which are present in every case, unlike glasses
    X_data_train_index, X_data_test_index, Y_data_train_index, Y_data_test_index = train_test_split(X_data_index, Y_data_index, test_size=0.2, random_state=1, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1, shuffle=False)

    print('Sanity Check: shuffling and indexes of images for future inference')
    print(any((Y_data_train_index == y_train)))
    print(len(Y_data_train_index) == len(y_train))
    print(any((Y_data_test_index == y_test)))
    print(len(Y_data_test_index) == len(y_test))

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print('Starting PCA Transform')
    X_train, X_test = pca_transform(X_train, X_test)

    # sanity check
    print('X_train of shape:', X_train.shape)
    print('y_train of shape:', y_train.shape)
    print('X_test of shape:', X_test.shape)
    print('y_test of shape:', y_test.shape)

    return X_train, X_test, y_train, y_test, X_data_train_index, X_data_test_index, Y_data_train_index, Y_data_test_index

labels = ['eyeglasses', 'smiling', 'young', 'human']



for label in labels:

    print('Computing PCA for label:', label)
    # import data (no one hot encoding)
    X_train, X_test, y_train, y_test, X_data_train_index, X_data_test_index, Y_data_train_index, Y_data_test_index = pull_dataset(label)

    print('saving training PCA based sets...')
    # saving PCA based training set
    np.savez('../pca_dataset/pca_{0}_train.npz'.format(label), name1=X_train, name2=y_train)

    print('saving test PCA based sets...')
    # saving PCA based test set
    np.savez('../pca_dataset/pca_{0}_test.npz'.format(label), name1=X_test, name2=y_test)

    print('saving training PCA based sets indexes...')
    # saving PCA based training set
    np.savez('../pca_dataset/pca_{0}_train_index.npz'.format(label), name1=np.array(X_data_train_index), name2=np.array(Y_data_train_index))

    print('saving test PCA based sets indexes...')
    # saving PCA based test set
    np.savez('../pca_dataset/pca_{0}_test_index.npz'.format(label), name1=np.array(X_data_test_index), name2=np.array(Y_data_test_index))


    print('loading train and test PCA based sets...')
    # test that loading is working fine
    data_train = np.load('../pca_dataset/pca_{0}_train.npz'.format(label))
    print(len(data_train['name2']))

    data_test = np.load('../pca_dataset/pca_{0}_test.npz'.format(label))
    print(len(data_test['name2']))

    print('loading train and test PCA based sets...')
    # test that loading is working fine
    data_train_index = np.load('../pca_dataset/pca_{0}_train_index.npz'.format(label))
    print(any(data_train_index['name2'] == data_train['name2']))

    data_test_index = np.load('../pca_dataset/pca_{0}_test.npz'.format(label))
    print(any(data_test_index['name2'] == data_test['name2']))
