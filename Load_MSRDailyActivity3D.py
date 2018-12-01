"""
Dataset description: 
The format of the skeleton file is as follows: 
The first integer is the number of frames. 
The second integer is the number of joints which is always 20. 

For each frame, the first integer is the number of rows 
(=40 when there is exactly one skeleton detected in current frame; =zero when no skeleton is detected; =80 when two skeletons are detected [which is rare]).

For most of the frames, the number of rows is 40. Each joint corresponds to two rows: the first row is its real world coordinates (x,y,z) and the second row is its screen coordinates plus depth (u, v, depth) where u and v are normalized to be within [0,1]. For each row, the integer at the end is supposed to be the confidence value, but it is not useful.

"""


"""
Example - "a01_s01_e01_skeleton.txt"
>>> 20 action types
>>> 10 subjects
>>> each subject performs each action 2 or 3 times
"""
import os
import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import cPickle
import random


def load_MSRDailyActivity3D_P4(filepath, params):
    
    print("Loading skeleton data of MSR DailyActivity 3D...")
    flag_normalize = True
    Samples = []
    Labels = []
    action_type = params[0]
    subjects = params[1]
    times = params[2]

    # Track the maximum number of frames across the full dataset
    max_frames = 0

    for a in action_type:
        for s in subjects:
            for t in times:
                # Load each skeleton joint file
                fileName = filepath + "/" + 'a%02i_s%02i_e%02i_skeleton.txt' % (a, s, t)
                print("Processing file a%02i_s%02i_e%02i_skeleton.txt .........." % (a, s, t))

                # Verify the current skeleton file exists
                flag = os.path.exists(fileName)
                if not flag:
                    print("Specified file does not exist!")
                else:
                    fr = open(fileName)     # Open the current file
                    frame_real_world = []   # Real world coord. (x,y,z) for each frame
                    frame_screen_depth = [] # Screen coordinates + depth (u,v,depth) for each frame
                    
                    # Get the number of frames and joints for the current file
                    header_info = fr.readline().split(" ")
                    curr_num_frames = int(header_info[0])
                    curr_num_joints = int(header_info[1])
                    # For DEBUG
#                     print("Number of frames and joints in current video:", curr_num_frames, curr_num_joints)
                    
                    sd_frame_mtx = [] # Matrix of screen + depth coord. frames
                    line_count = 0
                  
                    # Iterate through the number of frames in this file
                    for frame_count in range(curr_num_frames):
                        
                        num_joint_rows = fr.readline()
                        num_joint_rows = int(num_joint_rows)

                        if num_joint_rows == 0:
                            for i in range(curr_num_joints):
                                frame_screen_depth.append([0, 0, 0])
                                
                        # Process this frame's skeleton data and store each set of screen/depth coordinates 
                        for joint_count in range(int(num_joint_rows / 2)):
                            
                            # Read in the line of real world coordinates (to ignore)
                            rw_coord = fr.readline()
                            
                            # Get this frame's screen coordinates & add to the list
                            sd_coord = fr.readline().split(" ") # Screen (u,v) and depth coordinates
                            frame_screen_depth.append([float(sd_coord[0]), float(sd_coord[1]), float(sd_coord[2])])

                            
                    # Create a new reshaped "matrix" of frames for the curr file:
                    for i in range(curr_num_frames):
                        
                        sd_mat_frame = []
                        for j in range(curr_num_joints):
                            
                            index = i * curr_num_joints + j
                            if index >= len(frame_screen_depth):
                                continue
#                             print(len(frame_screen_depth), len(frame_screen_depth[0]))
                            sd_mat_frame.extend(frame_screen_depth[index])
    
                        # Append all 3 coordinate values (for 20 joints) of current frame to sd_frame_mtx
                        sd_frame_mtx.append(np.array(sd_mat_frame)) # sd_mat_frame is has shape (60,)

                    # Convert into an array of lists & get its dimensions
                    sd_frame_mtx = np.array(sd_frame_mtx)
                    n_frames, n_features = sd_frame_mtx.shape
                    temp = sd_frame_mtx
                    frame_mtx_temp = np.zeros((curr_num_frames - 4, n_features))

                    # Apply the Savitzky-Golay smoothing filter
                    for i in range(2, n_frames - 2):
                        frame_mtx_temp[i-2,:] = (-3*temp[i-2,:]+12*temp[i-1,:]+17*temp[i,:]+12*temp[i+1,:]-3*temp[i+2,:]) / 35
                    sd_frame_mtx = frame_mtx_temp
                    n_frames, n_features = sd_frame_mtx.shape
#                     print("(line 112) Num frames:", n_frames, "num features:", n_features)
 
                    # Track the maximum number of frames for padding later
                    if n_frames > max_frames:
                        max_frames = n_frames

                    # Centralize origin coordinates to the average of the three hip joints w.r.t. each frame
                    temp = sd_frame_mtx
                    frame_mtx_temp = np.zeros((n_frames, n_features))
                    for i in range(n_frames):
                        
                        # Origin is the average of the left, right, and center hip joints
                        Origin = (temp[i, 12 : 15] + temp[i, 15 : 18] + temp[i, 18 : 21]) / 3

                        for j in range(curr_num_joints):
                            index = 3 * j
                            frame_mtx_temp[i, index] = temp[i, index] - Origin[0]
                            frame_mtx_temp[i, index + 1] = temp[i, index + 1] - Origin[1]
                            frame_mtx_temp[i, index + 2] = temp[i, index + 2] - Origin[2]

                    # Normalization
                    sd_frame_mtx = frame_mtx_temp
                    if flag_normalize:
                        
                        # Normalize by subtracting the mean of each feature
                        for j in range(n_features):
                            sd_frame_mtx[:, j] = sd_frame_mtx[:, j] - mean(sd_frame_mtx[:, j])

                    print("Dimens of sd_frame_mtx:", len(sd_frame_mtx[0]), len(sd_frame_mtx))
                    Samples.append(sd_frame_mtx)
                    Labels.extend([a-1])
                    print("Added video to Samples; dimen:", len(Samples), len(Samples[0]))
                    print("Added label to Labels; dimen:", len(Labels))
    
    Samples = np.array(Samples)
    print("Before splitting samples into 5 parts; shape of Samples:", Samples.shape, len(Samples[0]))

    # Split the skeleton data into 5 regions (from original data loader)
    flag_split5parts = True
    if flag_split5parts:

        Samples_left_hand = []
        Samples_right_hand = []
        Samples_left_leg = []
        Samples_right_leg = []
        Samples_central_trunk = []
        nums_samples = len(Samples)

        for i in range(nums_samples):
            sample = Samples[i]
            nums_frames = sample.shape[0]

            Frames_left_hand = []
            Frames_right_hand = []
            Frames_left_leg = []
            Frames_right_leg = []
            Frames_central_trunk = []

            for j in range(nums_frames):

                left_hand_joint = []
                for k in [1, 8, 10, 12]:
                    left_hand_joint.extend(list(Samples[i][j][(k-1)*3 : k*3]))

                right_hand_joint = []
                for k in [2, 9, 11, 13]:
                    right_hand_joint.extend(list(Samples[i][j][(k-1)*3 : k*3]))

                left_leg_joint = []
                for k in [5, 14, 16, 18]:
                    left_leg_joint.extend(list(Samples[i][j][(k-1)*3 : k*3]))

                right_leg_joint = []
                for k in [6, 15, 17, 19]:
                    right_leg_joint.extend(list(Samples[i][j][(k-1)*3 : k*3]))

                central_trunk_joint = []
                for k in [20, 3, 4, 7]:
                    central_trunk_joint.extend(list(Samples[i][j][(k-1)*3 : k*3]))

                Frames_left_hand.append(left_hand_joint)
                Frames_right_hand.append(right_hand_joint)
                Frames_left_leg.append(left_leg_joint)
                Frames_right_leg.append(right_leg_joint)
                Frames_central_trunk.append(central_trunk_joint)

            Samples_left_hand.append(Frames_left_hand)
            Samples_right_hand.append(Frames_right_hand)
            Samples_left_leg.append(Frames_left_leg)
            Samples_right_leg.append(Frames_right_leg)
            Samples_central_trunk.append(Frames_central_trunk)

    results = np.array(Samples_left_hand), np.array(Samples_right_hand), np.array(Samples_left_leg), \
              np.array(Samples_right_leg), np.array(Samples_central_trunk), np.array(Labels), max_frames

    return results

if __name__ == '__main__':
    """
    In Protocol 4, the most widely adopted, cross-subject validation 
    with subjects 1,3,5,7, and 9 for training, the others for testing is adopted,
    following the HS-V protocol: Half subjects to test model, the rest for training.
    """
    
    filepath = "../data/MSRDailyActivity3D"

    # There are 16 activities in MSR DailyActivity 
    AS1_a = [2, 3, 5, 6, 10, 13]
    AS2_a = [1, 4, 7, 8, 9, 11]
    AS3_a = [6, 14, 15, 16, 11, 12]
    # Original activity splits used for MSR Action 3D
#     AS1_a = [2, 3, 5, 6, 10, 13, 18, 20]
#     AS2_a = [1, 4, 7, 8, 9, 11, 12, 14]
#     AS3_a = [6, 14, 15, 16, 17, 18, 19, 20]

    # There are 10 subjects in the dataset
    subject_train = [1, 3, 5, 7, 9]
    subject_test = [2, 4, 6, 8, 10]

    # Each subject in the dataset performs an activity twice
    times = [1, 2] 
    AS1_train_params = [AS1_a, subject_train, times]
    AS2_train_params = [AS2_a, subject_train, times]
    AS3_train_params = [AS3_a, subject_train, times]
    AS1_test_params = [AS1_a, subject_test, times]
    AS2_test_params = [AS2_a, subject_test, times]
    AS3_test_params = [AS3_a, subject_test, times]

    # Store the maximum # frames in each split
    max_frames = [0 for i in range(6)]

    # Load in the skeleton data for all activities, subjects,...
    AS1_train_left_hand, AS1_train_right_hand, AS1_train_left_leg, AS1_train_right_leg, AS1_train_central_trunk, AS1_train_Labels, max_frames[0] = load_MSRDailyActivity3D_P4(filepath, AS1_train_params)
    
    AS2_train_left_hand, AS2_train_right_hand, AS2_train_left_leg, AS2_train_right_leg, AS2_train_central_trunk, AS2_train_Labels, max_frames[1] = load_MSRDailyActivity3D_P4(filepath, AS2_train_params)
    
    AS3_train_left_hand, AS3_train_right_hand, AS3_train_left_leg, AS3_train_right_leg, AS3_train_central_trunk, AS3_train_Labels, max_frames[2] = load_MSRDailyActivity3D_P4(filepath, AS3_train_params)

    AS1_test_left_hand, AS1_test_right_hand, AS1_test_left_leg, AS1_test_right_leg, AS1_test_central_trunk, AS1_test_Labels, max_frames[3] = load_MSRDailyActivity3D_P4(filepath, AS1_test_params)
    
    AS2_test_left_hand, AS2_test_right_hand, AS2_test_left_leg, AS2_test_right_leg, AS2_test_central_trunk, AS2_test_Labels, max_frames[4] = load_MSRDailyActivity3D_P4(filepath, AS2_test_params)
    
    AS3_test_left_hand, AS3_test_right_hand, AS3_test_left_leg, AS3_test_right_leg, AS3_test_central_trunk, AS3_test_Labels, max_frames[5] = load_MSRDailyActivity3D_P4(filepath, AS3_test_params)

    # Get the maximum number of frames across the 6 splits
    total_max_frames = np.max(max_frames)    
    
    flag_created = True # From original code: not sure why this is here...
    if flag_created:
        
        # Use the max frames across all 6 splits as the largest frame size to pad the entire dataset
        pickle.dump([AS1_train_left_hand, AS1_train_right_hand, AS1_train_left_leg, 
                     AS1_train_right_leg, AS1_train_central_trunk, AS1_train_Labels, total_max_frames], 
                    open("../data/MSRDailyActivity3D_P4_Split_AS1_train.p", "wb"))
        
        pickle.dump([AS2_train_left_hand, AS2_train_right_hand, AS2_train_left_leg, 
                     AS2_train_right_leg, AS2_train_central_trunk, AS2_train_Labels, total_max_frames], 
                    open("../data/MSRDailyActivity3D_P4_Split_AS2_train.p", "wb"))
        
        pickle.dump([AS3_train_left_hand, AS3_train_right_hand, AS3_train_left_leg, 
                     AS3_train_right_leg, AS3_train_central_trunk, AS3_train_Labels, total_max_frames], 
                    open("../data/MSRDailyActivity3D_P4_Split_AS3_train.p", "wb"))
        
        pickle.dump([AS1_test_left_hand, AS1_test_right_hand, AS1_test_left_leg, 
                     AS1_test_right_leg, AS1_test_central_trunk, AS1_test_Labels, total_max_frames], 
                    open("../data/MSRDailyActivity3D_P4_Split_AS1_test.p", "wb"))
        
        pickle.dump([AS2_test_left_hand, AS2_test_right_hand, AS2_test_left_leg, 
                     AS2_test_right_leg, AS2_test_central_trunk, AS2_test_Labels, total_max_frames], 
                    open("../data/MSRDailyActivity3D_P4_Split_AS2_test.p", "wb"))
        
        pickle.dump([AS3_test_left_hand, AS3_test_right_hand, AS3_test_left_leg, 
                     AS3_test_right_leg, AS3_test_central_trunk, AS3_test_Labels, total_max_frames], 
                    open("../data/MSRDailyActivity3D_P4_Split_AS3_test.p", "wb"))
        
        print("Dataset created!")







