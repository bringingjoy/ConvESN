####################################################################
# Padding.py
# 
# Description: 
# Adds padding to the input data such that all samples in the set
# have "max_frame" size 
# (maximum number of frames of a recording in the given set)
# 
# Run with the specified dataset to pad,
# e.g. <python Padding.py MSRDailyActivity3D>
####################################################################

# import cPickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

def process(data_train, data_test):
    train_left_hand, train_right_hand, train_left_leg, train_right_leg, train_central_trunk, train_Labels, train_max_frames = data_train
    test_left_hand, test_right_hand, test_left_leg, test_right_leg, test_central_trunk, test_Labels, test_max_frames = data_test

    max_frames = max(train_max_frames, test_max_frames)

    print("Preprocessing training data ...")
    train_left_hand  	= padding(train_left_hand , max_frame = max_frames)
    train_right_hand 	= padding(train_right_hand, max_frame = max_frames)
    train_left_leg   	= padding(train_left_leg  , max_frame = max_frames)
    train_right_leg  	= padding(train_right_leg , max_frame = max_frames)
    train_central_trunk = padding(train_central_trunk, max_frame = max_frames)

    print("Preprocessing testing data ...")
    test_left_hand  	= padding(test_left_hand 	, max_frame = max_frames)
    test_right_hand 	= padding(test_right_hand   , max_frame = max_frames)
    test_left_leg   	= padding(test_left_leg  	, max_frame = max_frames)
    test_right_leg  	= padding(test_right_leg 	, max_frame = max_frames)
    test_central_trunk 	= padding(test_central_trunk, max_frame = max_frames)

    print("Preprocessing finished ...")

    data_train = [train_left_hand, train_right_hand, train_left_leg, train_right_leg, train_central_trunk, train_Labels]
    data_test = [test_left_hand, test_right_hand, test_left_leg, test_right_leg, test_central_trunk, test_Labels]

    return data_train, data_test


def process_singal(data):
    left_hand, right_hand, left_leg, right_leg, central_trunk, Labels, train_max_frames = data

    max_frames = train_max_frames

    print("Preprocessing training data ...")
    left_hand  	= padding(left_hand , max_frame = max_frames)
    right_hand 	= padding(right_hand, max_frame = max_frames)
    left_leg   	= padding(left_leg  , max_frame = max_frames)
    right_leg  	= padding(right_leg , max_frame = max_frames)
    central_trunk = padding(central_trunk, max_frame = max_frames)

    print("Preprocessing finished ...")

    data = [left_hand, right_hand, left_leg, right_leg, central_trunk, Labels]
    return data


def padding(data, max_frame=300):

    nums_sample = data.shape[0]
    nums_data_frame, nums_feature = np.shape(data[0])

    data_new = []

    for i in range(nums_sample):
        sample_i = np.array(data[i])

        num_frame_sample_i = np.shape(sample_i)[0]

        assert max_frame >= num_frame_sample_i

        zero = np.zeros((max_frame - num_frame_sample_i, nums_feature))
        sample_i_new = np.vstack((sample_i, zero))

        data_new.append(sample_i_new)

    data = np.array(data_new)
    return data


if __name__ == '__main__':

    train_path = "../data/MSRDailyActivity3D_train_split_s"
    test_path = "../data/MSRDailyActivity3D_test_split_s"
#     train_path = "../data/MSRDailyActivity3D_fromAction_noInterpolation_train_split_s"
#     test_path = "../data/MSRDailyActivity3D_fromAction_noInterpolation_test_split_s"
    
    subjects = np.arange(1, 11)
    for sub in subjects:
        
        train = pickle.load(open(train_path + str(sub) + ".pkl", "rb"))
        test = pickle.load(open(test_path + str(sub) + ".pkl", "rb"))
        train, test = process(train, test)
        
        train_left_hand, train_right_hand, train_left_leg, train_right_leg, train_central_trunk, train_labels = train
        test_left_hand, test_right_hand, test_left_leg, test_right_leg, test_central_trunk, test_labels = test
        
        pickle.dump([train_left_hand, train_right_hand, train_left_leg, 
         train_right_leg, train_central_trunk, train_labels], 
            open("../data/" + train_path + str(sub) + ".pkl", "wb"))
        
        pickle.dump([test_left_hand, test_right_hand, test_left_leg, 
                     test_right_leg, test_central_trunk, test_labels],
        open("../data/" + test_path + str(sub) + ".pkl", "wb"))
    
        print("Saved and padded split with holdout subject", sub)
     
    print("Padding complete")
