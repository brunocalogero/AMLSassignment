import time
import os
import math
import cv2
import sklearn.preprocessing

import numpy as np
import datetime as dt
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

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

def check_accuracy(sess, dset, x, scores, merge, is_training=None):
    """
    Check accuracy on a classification model. (training accuracy)

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        summary, scores_np = sess.run([merge, scores], feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    with tf.name_scope("cost"):
        acc = float(num_correct) / num_samples
        # add scalar summary for accuracy tensor
        tf.summary.scalar("cost", acc)
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

def model_init_fn(inputs):
    """
    Defining our Tensorflow model (for the moment 4 different layers (including last FC layer))
    """
    channel_1, channel_2, channel_3, num_classes =  64, 32, 16, 6
    # consider using initializer for conv layers (variance scaling)

    conv1 = tf.layers.conv2d(inputs, channel_1, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
    bn1 = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(bn1, 2, 2)

    # maybe add a dropout/dropconnect at some point here

    conv2 = tf.layers.conv2d(pool1, channel_2, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
    bn2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(bn2, 2, 2)

    # conv3 = tf.layers.conv2d(pool2, channel_3, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
    # bn3 = tf.layers.batch_normalization(conv3)
    # pool3 = tf.layers.max_pooling2d(bn3, 2, 2)

    conv3_flattened = tf.layers.flatten(pool2)
    fc = tf.layers.dense(conv3_flattened, num_classes)

    return fc

# user choses grey scale or not, 0 for yes, 1 for no
grey_scale = 1
# setting user chosen vars
images_dir = '../new_dataset'
labels_filename = 'attribute_list.csv'

# import data
X_data, y_data = pull_dataset()
# one hot encode labels (for categorical cross-entropy)
# label_binarizer = sklearn.preprocessing.LabelBinarizer()
# label_binarizer.fit(range(max(y_data)+1))
# y_data = label_binarizer.transform(y_data)


# sanity check for one-hot encoding
print('y_data of shape:', y_data.shape)
print('y_data of type:', type(y_data))
print(y_data[0])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# NOTE: Set up global variables For GPU use
USE_GPU = False

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

print('Using device: ', device)


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



# NOTE: constant to control the learning rate of the optimizer
learning_rate = 0.001
# NOTE: Constant to control how often we print when training models
print_every = 2
# NOTE: constant to control the number of epochs we want to train for
num_epochs = 10
# NOTE: constant to control the rate at which we save the model
save_every = 250

# Getting batches setup from Dataset Class
train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)

# sanity check
print(train_dset)


# START TRAINING
tf.reset_default_graph()
with tf.device(device):
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    y = tf.placeholder(tf.int32, [None])
    print(y.name)
    is_training = tf.placeholder(tf.bool, name='is_training')
    _ = tf.Variable(initial_value='fake_variable')

    with tf.name_scope("cost"):
        scores = model_init_fn(x)
        loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss   = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        # add scalar summary for cost tensor
        tf.summary.scalar("cost", loss)


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter( './tf_logs/1/train ', sess.graph)

    t = 0
    for epoch in range(num_epochs):
        print('Starting epoch %d' % epoch)
        for x_np, y_np in train_dset:
            # passing merge tensor to session for summary
            print('HELOOOOOO')
            print(y_np.shape)
            print(type(y_np))
            merge = tf.summary.merge_all()
            feed_dict = {x: x_np, y: y_np, is_training:1}
            summary, loss_np, _ = sess.run([merge, loss, train_op], feed_dict=feed_dict)
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss_np))
                summary = check_accuracy(sess, val_dset, x, scores, merge, is_training=is_training)
                train_writer.add_summary(summary, t)
            if t % save_every == 0:
                print('saving model..')
                save_path = tf.train.Saver().save(sess, "saved_model/network_weights.ckpt")
                print("Model saved in file: %s" % save_path)
            t += 1
    print('saving FINAL model..')
    save_path = tf.train.Saver().save(sess, "saved_model/network_weights.ckpt")
    print("FINAL Model saved in file: %s" % save_path)
