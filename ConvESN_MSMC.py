####################################################################
# ConvESN model using multi-step and multi-channel fusion approach
####################################################################
# Run this module with the respective dataset as a command-line arg.
# For example, if using the MSR Daily Activity 3D dataset:
# 1. Run <python Load_MSRDailyActivity3D.py>
# 2. Run <python Padding.py MSRDailyActivity>
# 3. Run <python ConvESN_MSMC.py MSRDailyActivity>
#
#
# Input data is expected to be a .p file composed of a list:
# [left_hand_skeleton, right_hand_skeleton, left_leg_skeleton, 
#    right_leg_skeleton, central_trunk_skeleton, labels]
# 
# The first five elements have shape: (num_samples, time_length, num_joints)
# The last element (labels) has shape: (num_samples,)
####################################################################
# Based on the architecture described in the paper:
# "WALKING WALKing walking: Action Recognition from Action Echoes"
# 
# by Qianli Ma, Lifeng Shen, Enhuan Chen, Shuai Tian, Jiabing Wang, 
# Garrison W. Cottrell
####################################################################
# Code written by Lifeng Shen, Qianli Ma? with modifications by
# Jenny Hamer
####################################################################

# Imports and dependencies
import numpy as np
import pickle as cp
import sys

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D

import reservoir
import utils


print('Loading data...')
"""
Input: a p file composed of a list:
[left_hand_skeleton, right_hand_skeleton, left_leg_skeleton, right_leg_skeleton, central_trunk_skeleton, labels]

- The shape of the first five elements: (num_samples, time_length, num_joints)
- The shape of the last element (labels): (num_samples,)
"""

# filepath_train = './dataset/MSRAction3D_real_world_P4_Split_AS3_train.p'
# filepath_test = './dataset/MSRAction3D_real_world_P4_Split_AS3_test.p'
# data_train = cp.load(open(filepath_train, 'rb'))
# skeletons_train = data_train[0:5]
# labels_train = data_train[5]
# data_test = cp.load(open(filepath_test, 'rb'))
# skeletons_test = data_test[0:5]
# labels_test = data_test[5]


# Check that a dataset was specified: if not, throw an exception
if len(sys.argv) < 3:
    raise Exception("Please specify the name of the dataset and desired data split ('all', 'AS1', etc.) and run again")
else: 
    dataset_name = sys.argv[1]

# Load in the desired data split: combine all splits or use one of the 3
if sys.argv[2] == "all":
    # Load in the 3 training and testing files
    filepaths_train = []
    filepaths_test = []

    filepaths_train.append("./data/" + dataset_name + "_P4_Split_AS1_train.p")
    filepaths_test.append("./data/" + dataset_name + "_P4_Split_AS1_test.p")
    filepaths_train.append("./data/" + dataset_name + "_P4_Split_AS2_train.p")
    filepaths_test.append("./data/" + dataset_name + "_P4_Split_AS2_test.p")
    filepaths_train.append("./data/" + dataset_name + "_P4_Split_AS3_train.p")
    filepaths_test.append("./data/" + dataset_name + "_P4_Split_AS3_test.p")

    data_train = cp.load(open(filepaths_train[0], 'rb'))
    skeletons_train = np.array(data_train[0:5])
    labels_train = data_train[5]

    data_test = cp.load(open(filepaths_test[0], 'rb'))
    skeletons_test = np.array(data_test[0:5])
    labels_test = data_test[5]


    # Combine all of the train and test splits into one training and one test set
    for i in range(len(filepaths_train)-1):
        data_train = cp.load(open(filepaths_train[i+1], 'rb'))
        skeletons_train = np.hstack((skeletons_train, np.array(data_train[0:5])))
        labels_train = np.hstack((labels_train, data_train[5]))

        data_test = cp.load(open(filepaths_test[i+1], 'rb'))
        skeletons_test = np.hstack((skeletons_test, np.array(data_test[0:5])))
        labels_test = np.hstack((labels_test, data_test[5]))

# Otherwise, load in the specified training/test data split
else:
    filepath = "./data/" + dataset_name + "_P4_Split_" + sys.argv[2]
    data_train = cp.load(open(filepath + "_train.p", "rb"))
    skeletons_train = data_train[0:5]
    labels_train = data_train[5]

    data_test = cp.load(open(filepath + "_test.p", "rb"))
    skeletons_test = data_test[0:5]
    labels_test = data_test[5]

    
print("Dimens of train labels", labels_train.shape, "dimens of test labels", labels_test.shape)
print('Transfering labels...')
labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

"""
Set parameters of reservoirs, create five reservoirs 
and get echo states of five skeleton parts
"""
num_samples_train = labels_train.shape[0]
num_samples_test = labels_test.shape[0]

print("Num training samples:", num_samples_train)
print("Num testing samples:", num_samples_test)


# Total number of training samples, max number of frames across all train/test videos
num_samples, time_length, n_in = skeletons_train[0].shape
print("From training set: num samples", num_samples, "time_length", time_length, "n_in", n_in)
print("From test set: num samples, time length, and n_in:", skeletons_train[0].shape)

n_res = n_in * 3
IS = 0.1
SR = 0.9 # 0.99 in the paper 
sparsity = 0.3
leakyrate = 1.0

# Create five different reservoirs, one for a skeleton part
reservoirs = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate) for i in range(5)]

print('Getting echo states...')
echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(5)]
echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(5)]

for i in range(5):
    # FOR DEBUG:
#     print("dimen of echo_states_train at index i:", echo_states_train[i][:, 0, :, :].shape)
#     print("dimen of reservoirs get echo states of skeletons_train index at i:", reservoirs[i].get_echo_states(skeletons_train[i]).shape)
#     print("dimen of echo_states_test at index i:", echo_states_test[i][:, 0, :, :].shape)
#     print("dimen of skeletons_test at index i:", skeletons_test[i].shape)
#     print("dimen of reservoirs get echo states of skeletons_train index at i:", reservoirs[i].get_echo_states(skeletons_test[i]).shape)
    echo_states_train[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_train[i])
    echo_states_test[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_test[i])
    
echo_states_train = [np.concatenate(echo_states_train[0:2], axis=1), 
                     np.concatenate(echo_states_train[2:4], axis=1), echo_states_train[4]]

echo_states_test = [np.concatenate(echo_states_test[0:2], axis=1), 
                    np.concatenate(echo_states_test[2:4], axis=1), echo_states_test[4]]

"""
Set hyperparameters of convolution layers and build the MSMC decoder model
Hyperparameters include:
# epochs: 300
batch size: 8

# filters: 16
kernel stride: 1x1
conv kernel initialization: LeCun uniform
activation function: ReLU
SGD optimizer: Adam
"""
input_shapes = ((2, time_length, n_res), (2, time_length, n_res), (1, time_length, n_res))
nb_filter = 16
nb_row = (2, 3, 4) # Time scales
nb_col = n_res
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)

data_format = 'channels_first'
optimizer = 'adam'
batch_size = 8
nb_epoch = 300
verbose = 1

# Build the MSMC decoder model
inputs = []
features = []
for i in range(3):
    input = Input(shape=input_shapes[i])
    inputs.append(input)

    pools = []
    for j in range(len(nb_row)):
        conv = Conv2D(nb_filter, (nb_row[j], nb_col), 
                      kernel_initializer=kernel_initializer, activation=activation, 
                      padding=padding, strides=strides, data_format=data_format)(input)
        pool = GlobalMaxPooling2D(data_format=data_format)(conv)
        pools.append(pool)

    features.append(concatenate(pools))

    
"""
hands_features = features[0]
legs_features = features[1]
trunk_features = features[2]
body_features = Dense(nb_filter * len(nb_row), kernel_initializer = kernel_initializer, activation = activation)(concatenate([hands_features, legs_features, trunk_features]))
"""

# Initialize the Conv component of the model
body_features = Dense(nb_filter * len(nb_row), kernel_initializer=kernel_initializer, 
                      activation=activation)(concatenate(features))

# Initialize the output layer of the model with softmax for classification
outputs = Dense(num_classes, kernel_initializer=kernel_initializer, activation='softmax')(body_features)

# Initialize the ConvESN_MSMC model using Adam optimizer, Categorical Cross-Ent loss, and accuracy evaluation metric
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(echo_states_train, labels_train, batch_size=batch_size, epochs=nb_epoch, 
          verbose=verbose, validation_data=(echo_states_test, labels_test))
