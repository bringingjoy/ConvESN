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
# Code written by Lifeng Shen, Qianli Ma with modifications and 
# hyperparameter optimization by Jenny Hamer
####################################################################

# Imports and dependencies
import numpy as np
import pickle as cp
import sys
import random
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import reservoir
import utils

np.random.seed(1)

# Percent of test set to dedicate to validation
PERCENT_VALIDATION = 0.28 
def random_valid_test_split(x_test, y_test, num_val_samples):
    """ Randomly split the test set into validation and test. 
    """
    rand_state = np.random.get_state()
    sampler = np.random.rand(len(y_test)) < PERCENT_VALIDATION
    x_val = x_test[:, sampler, :, :]
    x_test = x_test[:, ~sampler, :, :]
    np.random.set_state(rand_state)
    sampler = np.random.rand(len(y_test)) < PERCENT_VALIDATION
    y_val = y_test[sampler]
    y_test = y_test[~sampler]
    return np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)


def get_valid_test_set(x_test, y_test, num_val_samples):
    """ Generate a validation set from the test set with an
    equal number of samples from each class.
    """
    num_per_class = int(len(y_test) / num_val_samples) 
    dimen_0, dimen_1, dimen_2, dimen_3 = x_test.shape
    val_dimen_1 = int(num_per_class * len(y_test[0]))
    x_val = np.zeros((dimen_0, val_dimen_1, dimen_2, dimen_3))
    
#     print("x_val dimens:", x_val.shape)
    y_val = []
    count = np.zeros((len(y_test[0])))
    total_count = 0
    val_indices = []
    print("Number of validation samples per class:", num_per_class)
    
    # Create the validation set
    for i in range(len(y_test)):
        
        curr_class = np.argmax(y_test[i])
        
        if count[curr_class] < num_per_class:
            x_val[:, total_count, :, :] = x_test[:, i, :, :]
            y_val.append(y_test[i])
            val_indices.append(i)
            count[curr_class] += 1
            total_count += 1
            
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    # Remove the validation samples from the test set
    num_test = dimen_1 - val_dimen_1
    tmp_x_test = np.zeros((dimen_0, num_test, dimen_2, dimen_3))
    tmp_y_test = []
    test_count = 0
    
    for i in range(len(y_test)):
        if i not in val_indices:
            tmp_x_test[:, test_count, :, :] = x_test[:, i, :, :]
            tmp_y_test.append(y_test[i])
            test_count += 1
    x_test = np.array(tmp_x_test)
    y_test = np.array(tmp_y_test)

    return x_val, y_val, x_test, y_test


# Early stopping thresholds
monitor = "val_categorical_accuracy" # metric to monitor
min_delta = 0        # minimum change in monitored quantity to qualify as an improvement
patience = 15        # the number of permissible epochs with no improvement after which training will be stopped
mode = "max"         # maximize the input metric
baseline = None      # baseline desired value for the monitored quantity to reach


print('Loading data...')
"""
Input: a p file composed of a list:
[left_hand_skeleton, right_hand_skeleton, left_leg_skeleton, right_leg_skeleton, central_trunk_skeleton, labels]

- The shape of the first five elements: (num_regions, num_samples, time_length, num_joints)
- The shape of the last element (labels): (num_samples,)
"""

# # Check that a dataset was specified: if not, throw an exception
# if len(sys.argv) < 3:
#     raise Exception("Please specify the name of the dataset and desired data split ('all', 'AS1', etc.) and run again")
# else: 
#     dataset_name = sys.argv[1]
#     split = sys.argv[2]

# Check that a dataset was specified
if len(sys.arg) < 2:
    raise Exception("Please specifiy the name of the dataset (e.g. 'MSRDailyActivity3D') and run again")
else:
    dataset_name = sys.argv[1]
    
splits = ["AS1", "AS2", "AS3"]

# dataset_name = "MSR_NewSplits_DailyActivity"
# dataset_name = "MSRDailyActivity3D"
    
for split in splits:
    # Load in the desired data split: combine all splits or use one of the 3
    if split == "all":
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
        filepath = "./data/" + dataset_name + "_P4_Split_" + split #+ sys.argv[2]
        data_train = cp.load(open(filepath + "_train.p", "rb"))
        skeletons_train = data_train[0:5]
        labels_train = data_train[5]

        data_test = cp.load(open(filepath + "_test.p", "rb"))
        skeletons_test = data_test[0:5]
        labels_test = data_test[5]


    print("Dimens of train labels", labels_train.shape, "dimens of test labels", labels_test.shape)
    print('Transfering labels...')
    labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

    # Determine the number of test set samples to use as a validation set
    num_samples_valid = int(PERCENT_VALIDATION * len(labels_test))

    # Shuffle the test samples and their labels
    skeletons_test = np.array(skeletons_test)
    skeletons_valid, labels_valid, skeletons_test, labels_test = get_valid_test_set(skeletons_test, labels_test, num_samples_valid)

    # ALTERNATIVELY:
    # Split the validation samples from the test set
    # skeletons_valid = skeletons_test[:, : num_samples_valid, :, :]
    # labels_valid = labels_test[: num_samples_valid]
    # Remove the validation samples from the test set
    # skeletons_test = skeletons_test[:, num_samples_valid :, :, :]
    # labels_test = labels_test[num_samples_valid :]

    """
    Set parameters of reservoirs, create five reservoirs 
    and get echo states of five skeleton parts
    """
    num_samples_train = labels_train.shape[0]
    num_samples_valid = labels_valid.shape[0]
    num_samples_test = labels_test.shape[0]

    print("Num training samples:", num_samples_train)
    print("Num validation samples:", num_samples_valid)
    print("Num testing samples:", num_samples_test)

    # Total number of training samples, max number of frames across all train/test videos
    num_samples, time_length, n_in = skeletons_train[0].shape
    print("From training set: num samples", num_samples, "time_length", time_length, "n_in", n_in)
    # print("From test set: num samples, time length, and n_in:", skeletons_test[0].shape)

    # Original hyperparameter settings
    n_res = n_in * 9
    IS = 0.1
    SR = 0.99 # 0.99 in the paper 
    sparsity = 0.1
    leakyrate = 0.9

    # Set hyperparameter optimization - random search - settings
    # n_res = n_in * 3
    # input_scaling = [0.001, 1]   # input scaling
    # spectral_radius = [0.1, 1] # spectral radius of reservoir weight matrix (W_res)
    # leaky_rate = [0.1, 1]       # leaky-integrated discrete-time continuous-value units
    # sparsity = [0.0001, 0.5]

    # input_scaling = np.linspace(input_scaling[0], input_scaling[1], 50)
    # spectral_radius = np.linspace(spectral_radius[0], spectral_radius[1], 15)
    # leaky_rate = np.linspace(leaky_rate[0], leaky_rate[1], 5)
    # sparsity = np.linspace(sparsity[0], sparsity[1], 20)

    print("Hyperparameters:")
    print("Reservoir size:", n_res)
    print("Input scale:", IS)
    print("Spectral radius:", SR)
    print("Sparsity:", sparsity)
    print("Leaking rate:", leakyrate)


    # Initialize five different reservoirs, one for each skeletal region
    reservoirs = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate) for i in range(5)]
    print('Getting echo states...')
    
    echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(5)]
    echo_states_valid = [np.empty((num_samples_valid, 1, time_length, n_res), np.float32) for i in range(5)]
    echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(5)]

    # Get the Echo States for the 5 skeletal regions for each dataset split
    for i in range(5):
        echo_states_train[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_train[i])
        echo_states_valid[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_valid[i])
        echo_states_test[i][:, 0, :, :] = reservoirs[i].get_echo_states(skeletons_test[i])

    echo_states_train = [np.concatenate(echo_states_train[0:2], axis=1), 
                         np.concatenate(echo_states_train[2:4], axis=1), echo_states_train[4]]
    echo_states_valid = [np.concatenate(echo_states_valid[0:2], axis=1), 
                         np.concatenate(echo_states_valid[2:4], axis=1), echo_states_valid[4]]
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
    activation = 'relu' # Conv2D activation function; reservoirs use tanh
    padding = 'valid'
    strides = (1, 1)

    data_format = 'channels_first'
    optimizer = 'adam'
    batch_size = 8
    nb_epoch = 150
    verbose = 2 # One line per epoch

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
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit(echo_states_train, labels_train, batch_size=batch_size, epochs=nb_epoch, 
    #           verbose=verbose, validation_data=(echo_states_test, labels_test))

    # Save important training data: trained model weights, plot of training loss, training history
    training_description = dataset_name + "_" + split + "_early_stopping"
    
    # Setup up the EarlyStopping callback criteria
    early_stopping = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, 
                                      mode=mode, baseline=baseline)
    
    # Save the best model checkpoint only, based on validation accuracy
    model_filepath = "./model_chkpts/model_params" + training_description
    model_checkpoint = ModelCheckpoint(model_filepath, monitor="val_categorical_accuracy", verbose=1, 
                                       save_best_only=True, mode="max")
    callbacks_list = [early_stopping, model_checkpoint] # [model_checkpoint] 
    
    # Train and evaluate the model on training and validation sets, using EARLY STOPPING
    history = model.fit(echo_states_train, labels_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, 
                        callbacks=callbacks_list, validation_data=(echo_states_valid, labels_valid))


#     model.save_weights("./model_chkpts/model_params" + training_description)
#     plt.plot(history.history["loss"])
#     plt.savefig(training_description + "_training_loss_plot.jpg")
#     plt.plot(history.history["val_loss"])
#     plt.savefig(training_description + "_val_loss_plot.jpg")
    
    # Save the training history as a pickled dictionary
    cp.dump(history.history, open(training_description + "_training_history_dict.pkl", "wb"))

    print("****************************************************************")
    print("Evaluating the model on the test set:")
    print("Loading in the saved model weights with best validation accuracy")
    print("****************************************************************")
    # Load in saved weights and recompile model
    model.load_weights(model_filepath)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Get the echo states on the test set and evaluate the model on the test set
    scores = model.evaluate(x=echo_states_test, y=labels_test, batch_size=batch_size, verbose=verbose)
    print("****************************************************************")
    print("Overall classification accuracy:")
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     print("Test set loss:")
#     print("%s: %.2f%" % (model.metrics_names[0], scores[0]))
    print("****************************************************************")
    
    # Compute the model predictions and print the accuracies by activity
    predictions = model.predict(x=echo_states_test)

    # The activities in each split:
    # Current splits:
    AS1 = [2, 3, 5, 6, 10, 13]
    AS2 = [1, 4, 7, 8, 9, 11]
    AS3 = [6, 12, 14, 15, 16]
    # Previous splits
#     AS1 = [2, 3, 5, 10, 13]
#     AS2 = [1, 4, 6, 7, 8, 9]
#     AS3 = [11, 12, 14, 15, 16]
    
    # Map the numerical activity key to its description
    activity_dict = {1: "drink", 2: "eat", 3: "read book", 4: "call cellphone", 
                     5: "write on a paper", 6: "use laptop", 7: "use vacuum cleaner", 
                     8: "cheer up", 9: "sit still", 10: "toss paper", 
                     11: "play game", 12: "lie down on sofa", 13: "walk", 
                     14: "play guitar", 15: "stand up", 16: "sit down"}

    # print("Shape of predictions:", predictions.shape, predictions[:5], "labels:", labels_test[:5])
    accuracies = np.zeros((num_classes))
    totals_per_class = np.zeros((num_classes))
    
    for i in range(len(predictions)):
        # FOR DEBUG
#         if i < 2:
#             print("FOR DEBUG")
#             print("labels_test[i]", labels_test[i], "max index:", np.argmax(labels_test[i]))
#             print("predictions[i]", predictions[i], "max pred ind:", np.argmax(predictions[i]))
        
        true_label = np.argmax(labels_test[i])
        if np.argmax(predictions[i]) == np.argmax(labels_test[i]):
            accuracies[true_label] += 1
        totals_per_class[true_label] += 1

    print("****************************************************************")
    print("Computing prediction accuracies by activity:")
    print("****************************************************************")
    if split == "AS1":
        activities = AS1
    elif split == "AS2":
        activities = AS2
    else:
        activities = AS3

    log_file = open(training_description + "_accuracies.txt", "a")
    print("Activity\tAccuracy (% correct)")
    log_file.write("Activity\tAccuracy (% correct)")
    for i in range(len(accuracies)):
        activity = activities[i]
        accuracy = str(round( ((accuracies[i] / totals_per_class[i]) * 100), 2))
        
        acc_report = activity_dict[activity] + ": " + accuracy + "% correct"
        print(acc_report)
#         print("Total # samples of this class in test set:", totals_per_class[i])
        log_file.write(acc_report)

    log_file.close()
##################################################################################