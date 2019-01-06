
import numpy
import os
import cv2
import time
import keras
import shutil
import csv

import datetime as dt
import tensorflow as tf
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.applications import VGG16

from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt

 # user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1
# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'
test_images_dir = '../testing_dataset'


class Dataset(object):

    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

    def get_size(self):
        '''
        gets the size of input data
        '''
        return len(self.X)


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
    newdf = df['smiling']

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


def pull_test_set():
    '''
    Pulls the full dataset (without outliers).
    Used as a helper function to obtain full dataset that will be then split to user
    desired train/validation/test(inference) sets.
    The examples are shuffled.
    '''

    # lists keep the order
    full_dataset = []
    full_labels = []


    # collect pre-processed images and sort them to labels
    for (root, dirs, dat_files) in os.walk('{0}'.format(test_images_dir)):

        for file in dat_files:

            int_file = int(file[:-4])

            # image grayscaling at import
            img = cv2.imread('{0}/{1}'.format(test_images_dir, file), grey_scale)
            # image equalisation (to be inserted if interesting)
            # turn to float for zero centering
            res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
            res = res.astype(float)
            full_dataset.append(res)
            full_labels.append(int_file)

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


def extract_features(dataset , sample_count):
    # Pass data through convolutional base
    i = 0
    features = []
    labels = []
    for inputs_batch, labels_batch in dataset:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# import data
X_data, y_data_orig = pull_dataset()
# one hot encode labels (for categorical cross-entropy)
y_data = y_data_orig # to_categorical(y_data_orig, num_classes=2)


# import recently given test set
X_inference, inference_indexes = pull_test_set()

# Perform train and test split (random state set to 1 to ensure same distribution accross different sets).
# This split is obviously case specific! but cross validation allows us to avoid over-fitting so lets make sure we have a validation set ready.
# Since the dataset is not extremely large i'll be using a 75/15/20 split, meaning more or less 1000 validation and test examples and 3000 training examples, to be tested: 75/10/15.
# In this case we are a little less concerned since we are evaluating smiles which are present in every case, unlike glasses.
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_data, y_data_orig, test_size=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

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
X_inference = np.reshape(X_inference, (X_inference.shape[0], -1))

# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)
print('X_inference of shape:', X_inference.shape)

# Preprocessing: subtract the mean image

# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
std_image = np.std(X_train, axis=0)


# do it on inference test set
mean_image_test = np.mean(X_inference, axis=0)
std_image_test = np.std(X_inference, axis=0)

# second: subtract the mean image from train and test data
X_train -= mean_image
X_train /= std_image
X_val -= mean_image
X_val /= std_image
X_test -= mean_image
X_test /= std_image
X_inference -= mean_image
X_inference /= std_image

# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)
print('X_inference of shape:', X_inference.shape)

# Preprocessing: reshape back into three channels
X_train = np.reshape(X_train, (X_train.shape[0], 128, 128, 3))
X_val = np.reshape(X_val, (X_val.shape[0], 128, 128, 3))
X_test = np.reshape(X_test, (X_test.shape[0], 128, 128, 3))
X_inference = np.reshape(X_inference, (X_inference.shape[0], 128, 128, 3))


# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)
print('X_inference of shape:', X_inference.shape)

# # declaring number of folds for cross_validation
# n_folds = 8   # if using hyperparameterisation, please uncomment
epochs = 85
batch_size = 64
img_width = 128
img_height = 128


# Getting batches setup from Dataset Class
train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)



conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures

# Check architecture
conv_base.summary()


# Extract features

train_features, train_labels = extract_features(train_dset, train_dset.get_size())
validation_features, validation_labels = extract_features(val_dset, val_dset.get_size())
test_features, test_labels = extract_features(test_dset, test_dset.get_size())

# Debug
print('extract features debug')

print(validation_features)
print(validation_labels)



# model =Sequential()
# model.add(Flatten(input_shape=(4,4,512)))
# model.add(Dense(256, activation='relu', input_dim=(4*4*512)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(4,4,512)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer = Adam(),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# Train model
history = model.fit(np.array(train_features), np.array(train_labels),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(np.array(validation_features), np.array(validation_labels)))

# print('saving model')
# model.save('./saved_models/VGGnet_save.h5')

print('printing training/validation curves')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


print('starting fit on inference data given to us on Friday before the deadline')

prediction = []

for example in X_inference:

    # Extract features
    features = conv_base.predict(example.reshape(1,img_width, img_height, 3))

    # Make prediction
    try:
        prediction.append(model.predict(features))
    except:
        prediction.append(model.predict(features.reshape(1, 4*4*512)))


prediction = np.array(prediction)
prediction_new = []

print(prediction)

for i, values in enumerate(prediction):
    if values < 0.5:
        prediction_new.append('not_smiling')
    else:
        prediction_new.append('smilinig')


print('creating csv for inference')
with open('./result_logs/inference_VGG.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i, file in enumerate(list(range(1,101))):
        filewriter.writerow(['{0}.png'.format(file), prediction_new[i]])
