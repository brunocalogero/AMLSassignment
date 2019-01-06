import numpy
import os
import cv2
import time
import keras

import datetime as dt
import tensorflow as tf
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten, Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import to_categorical


from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from matplotlib import pyplot

# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1
# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'

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
    newdf = df['human']

    # collect pre-processed images and sort them to labels
    for (root, dirs, dat_files) in os.walk('{0}'.format(images_dir)):

        for file in dat_files:

            int_file = int(file[:-4])

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

    full_labels[full_labels == -1] = 0
    print(full_labels)

    print('full dataset of shape:', full_dataset.shape)
    print('full labels of shape:', full_labels.shape)

    # Reshuffling data (for extra randomness)
    X_data, Y_data = shuffle(full_dataset, full_labels, random_state=0)

    print('X_data of shape:', X_data.shape)
    print('Y_data of shape:', Y_data.shape)

    return X_data, Y_data

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', learn_rate=0.001, amsgrad=False, activation=tf.nn.leaky_relu):

    channel_1, channel_2, channel_3, num_classes =  32, 16, 8, 2
    # create model
    model = Sequential()

    model.add(Conv2D(channel_1, (3, 3), padding='SAME', activation=activation,  input_shape=(128, 128, 3), data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Dropout(0.2))

    model.add(Conv2D(channel_2, (3, 3), padding='SAME', activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    # model.add(Conv2D(channel_3, (3, 3), padding='SAME', activation=activation))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(lr=learn_rate, amsgrad=amsgrad )
    # Compile model (sparse cross-entropy can be used if one hot encoding not used)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

    return model

def batch_epoch_hyper(n_folds):
    '''
    This function finds the best hyperparameters among those given by the user for batch size and epoch size
    '''

    print('hyperparameterization: batch_size and epochs')
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the grid search parameters
    batch_size = [32, 64, 128]
    epochs = [10, 15, 20]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds, n_jobs=4)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def optimization_func_hyper(n_folds):
    '''
    This function finds the best hyperparameters among those given by the user for the chosen optimization function
    '''
    print('hyperparameterization: optimization function')
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64 ,verbose=0)

    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds, n_jobs=4)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def learning_rate_hyper(n_folds):
    '''
    This function finds the best hyperparameters among those given by the user for the chosen learning rate
    '''
    print('hyperparameterization: learning_rate / amsgrad')
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

    # define the grid search parameters
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    amsgrad = [False, True]
    param_grid = dict(learn_rate=learn_rate, amsgrad=amsgrad)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds, n_jobs=4)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def activation_hyper(n_folds):
    '''
    This function finds the best hyperparameters among those given by the user for the chosen learning rate
    '''

    print('hyperparameterization: activation function')
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

    # define the grid search parameters
    activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds, n_jobs=4)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# NOTE (TBD): hyperparameterization: number of neurons per layer to be added (if time allows)


# import data
X_data, y_data_orig = pull_dataset()
# one hot encode labels (for categorical cross-entropy)
y_data = to_categorical(y_data_orig, num_classes=2)

# Perform train and test split (random state set to 1 to ensure same distribution accross different sets).
# This split is obviously case specific! but cross validation allows us to avoid over-fitting so lets make sure we have a validation set ready.
# Since the dataset is not extremely large i'll be using a 60/20/20 split, meaning more or less 1000 validation and test examples and 3000 training examples, to be tested: 75/10/15.
# In this case we are a little less concerned since we are evaluating smiles which are present in every case, unlike glasses.
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_data, y_data_orig, test_size=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
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
# plt.figure(figsize=(4,4))
# plt.imshow(mean_image.reshape((128,128,3)).astype('uint8')) # visualize the mean image
# plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)


# Preprocessing: reshape back into three channels
X_train = np.reshape(X_train, (X_train.shape[0], 128, 128, 3))
X_val = np.reshape(X_val, (X_val.shape[0], 128, 128, 3))
X_test = np.reshape(X_test, (X_test.shape[0], 128, 128, 3))
# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)


# # declaring number of folds for cross_validation
# n_folds = 8   # if using hyperparameterisation, please uncomment
epochs = 8
batch_size = 64

# retrieve model
model = create_model()
model.summary()

# do inference with tensorboard and best parameters + validation set
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start learning with best params at {}'.format(str(start_time)))

model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[TensorBoard(log_dir='tf_logs/8/train')])

end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning time {}'.format(str(elapsed_time)))


predictions = model.predict(X_test, batch_size=batch_size)

predictions_sparse = []

for one_hot_pred in predictions:
    if one_hot_pred[0] > one_hot_pred[1]:
        predictions_sparse.append(0)
    else:
        predictions_sparse.append(1)

print(len(predictions_sparse) == len(y_test_orig))


print(predictions_sparse)
print(y_test_orig)

print(accuracy_score(y_test_orig, predictions_sparse))

# Now predict the value of the test
expected = y_test_orig

print("Classification report for classifier:\n%s\n", metrics.classification_report(expected, predictions_sparse))

cm = metrics.confusion_matrix(expected, predictions_sparse)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predictions_sparse)))
