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

def interpolate_coords(start_frame, end_frame, num_frames, num_joints):
    """ Interpolate coordinates between two frames by finding the 
    vector which spans the start and end frame.  

    Args:
        start_frame: (list of floats)
        end_frame:  (list of floats)
        num_frames: (int)
        num_joints: (int)
    Returns:
        approx_frames: (A list of lists of floats) The coordinates of each joint for the "num_frames" 
            missing frames.
    """
    
    # Track the distance between start and end frames for each joint
    u_component = []
    v_component = []
    d_component = []
    
    approx_frames = [] # Store the approximated frame skeleton data
    factor = 1 / num_frames 
    
    for i in range(num_joints):
        u_coord_diff = end_frame[i][0] - start_frame[i][0]
        v_coord_diff = end_frame[i][1] - start_frame[i][1]
        d_coord_diff = end_frame[i][2] - start_frame[i][2]
        
#         u_dists.append(sqrt(u_coord_diff ** 2))
#         v_dists.append(sqrt(v_coord_diff ** 2))
#         d_dists.append(sqrt(d_coord_diff ** 2))
        
        # Compute the vectors which span each coordinate
        u_component.append(u_coord_diff)
        v_component.append(v_coord_diff)
        d_component.append(d_coord_diff)
    
    # Approximate the coordinates across the missing frames for each joint
    prev_frame = start_frame
    for i in range(num_frames):
        
        print("In interpolate_coords: prev_frame", len(prev_frame), len(prev_frame[0]))
        for j in range(num_joints):
            
#             print("Types:", type(u_component), type(prev_frame), type(prev_frame[0]))
            u_coord = u_component[j] * factor + prev_frame[j][0]
            v_coord = v_component[j] * factor + prev_frame[j][1]
            d_coord = d_component[j] * factor + prev_frame[j][2]
            approx_frames.append([u_coord, v_coord, d_coord])

        # Update the previous frame with the last twenty joint coordinates
        prev_frame = approx_frames[-20:][:]

    return approx_frames


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
                file_name = filepath + "/" + 'a%02i_s%02i_e%02i_skeleton.txt' % (a, s, t)
                print("Processing file a%02i_s%02i_e%02i_skeleton.txt .........." % (a, s, t))

                # Verify the current skeleton file exists
                flag = os.path.exists(file_name)
                if not flag:
                    print("Specified file does not exist!")
                else:
                    fr = open(file_name)     # Open the current file
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
                  
                    # In case of missing data: rack the number of frames without skeleton coordinates
                    missing_data_start_frame = []
                    missing_data_end_frame = []
                    num_missing_frames = 0
                    total_missing_frames = 0
                    start_frame_position = 0
                    end_frame_position = 0
                    missing_frames = False
                    
                    
                    # Remember the previous frame in case interpolation is needed
#                     prev_frame = [0, 0, 0] 
                    
                    # Iterate through the number of frames in this file
                    for frame_count in range(curr_num_frames):
                        
                        num_joint_rows = int(fr.readline())

                        # Count the number of consecutive frames missing skeleton coordinates
                        if num_joint_rows == 0:
                            
                            # Remember the previous frame as a starting point for interpolation
                            if num_missing_frames == 0:
                                missing_frames = True
                                print("Shape of frame_screen_depth:", len(frame_screen_depth), len(frame_screen_depth[0]))
                                missing_data_start_frame = frame_screen_depth[-20:][:]
                                print("Missing_data_start_frame shape:", len(missing_data_start_frame)) #, missing_data_start_frame)
                                start_frame_position = len(frame_screen_depth)
                                print("Found missing data: setting flag to True.")
                                print("Start index of missing frame:", start_frame_position)
                            
                            num_missing_frames += 1
                            
                            # Corner case: the video ends with missed frames
                            if (frame_count + 1) == curr_num_frames:
                                curr_num_frames -= num_missing_frames
                                break

                        else:         
                            # Process this frame's skeleton data and store each set of screen/depth coordinates 
                            for joint_count in range(int(num_joint_rows / 2)):

                                # Remember the first frame following the missing data if it occurred in current video
                                if missing_frames and len(missing_data_end_frame) != curr_num_joints:

                                    # Get the index in "frame_screen_depth" of the last missed frame
                                    if end_frame_position == 0:
                                        end_frame_position = len(frame_screen_depth[-2])

                                    # Add the current joints coordinates to the end frame
                                    missing_data_end_frame.append(frame_screen_depth[-1][:])

                                # Combine the original and approximate data into an array and reset flags
                                elif missing_frames:
                                    print(num_missing_frames, "frames were missing in current video:", file_name)
                                    approx_frames = interpolate_coords(missing_data_start_frame, missing_data_end_frame, 
                                                       num_missing_frames, curr_num_joints)  

                                    # Combine the original & approximate data into an array
                                    num_frames = start_frame_position + len(approx_frames) + curr_num_joints #(num_missing_frames * 20)
                                    temp = np.zeros((num_frames, 3))
                                    print("Shape of temp:", temp.shape, "len of frame_screen_depth", len(frame_screen_depth), len(frame_screen_depth[0]), start_frame_position, end_frame_position, num_frames)
                                    temp[: start_frame_position, :] = frame_screen_depth[: start_frame_position]

                                    # Add the approximated frames and update frame_screen_depth
                                    end_ind = start_frame_position + len(approx_frames)
                                    temp[start_frame_position : end_ind, :] = approx_frames
                                    temp[end_ind :, :] = frame_screen_depth[-20:]
                                    frame_screen_depth = temp.tolist()
                                        
                                    # Update and reset counters to account for additional gaps of missing frames
                                    total_missing_frames += num_missing_frames
                                    missing_data_start_frame = []
                                    missing_data_end_frame = []
                                    num_missing_frames = 0
                                    start_frame_position = 0
                                    end_frame_position = 0
                                    missing_frames = False
                                
                        
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
                    print("Type before cast:", type(sd_frame_mtx))
                    sd_frame_mtx = np.array(sd_frame_mtx)
                    print("Type after cast:", type(sd_frame_mtx), type(sd_frame_mtx[0]), sd_frame_mtx.shape)
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
    Perform a 10-fold holdout validation:
    - 9/10 of the subjects are used to train with 1 held out for testing.
    """
    
    filepath = "../data/MSRDailyActivity3D/"

    # Define the training and holdout sets, using one subject as the holdout set
    actions = np.arange(1, 17)
    subjects = np.arange(1, 11)
    times = [1, 2]
    train_params = []
    test_params = []
    for i in range(len(subjects)):
        
        # Reserve the ith subject for holdout and use the remaining as training
        train_subjects = [subjects[j] for j in range(len(subjects)) if i != j]
        train_params.append([actions, train_subjects, times])
        test_params.append([actions, [i+1], times])
    
    for i in range(len(train_params)):
        
        # Load in the skeleton data for the current train/holdout split
        train_left_hand, train_right_hand, train_left_leg, train_right_leg, train_central_trunk, train_labels, train_max_frames = load_MSRDailyActivity3D_P4(filepath, train_params[i])
        
        test_left_hand, test_right_hand, test_left_leg, test_right_leg, test_central_trunk, test_labels, test_max_frames = load_MSRDailyActivity3D_P4(filepath, test_params[i])
                
        # Track the maximum number of frames across all videos
        max_frames_padding = max(train_max_frames, test_max_frames)
        
        # Save the current train and test splits as pickles
        train_filename = "../data/MSRDailyActivity3D_train_split_s" + str(test_params[i][1][0])
        test_filename = "../data/MSRDailyActivity3D_test_split_s" + str(test_params[i][1][0])
        
        pickle.dump([train_left_hand, train_right_hand, train_left_leg, train_right_leg, 
                     train_central_trunk, train_labels, max_frames_padding], open(train_filename + ".pkl", "wb"))
        
        pickle.dump([test_left_hand, test_right_hand, test_left_leg, test_right_leg, 
                     test_central_trunk, test_labels, max_frames_padding], open(test_filename + ".pkl", "wb"))
        print("Saved train set using:", train_params[i][1], "subjects")
        print("Saved test set using:", test_params[i][1][0], "as holdout subject")
        print("*****************************************")
        
    print("Dataset created!")
    





