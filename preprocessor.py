import pdb
import pandas as pd
import numpy as np
from test_pre_trained import split_to_frames
from pc_filter import filter_x_y_z_axis, dbscan_filter_frame
import functools
from test_pre_trained import show_plot

def sort_xyz(point1, point2):
    if point1[0] != point2[0]:
        return point1[0]-point2[0]
    if point1[1] != point2[1]:
        return point1[2]-point2[2]
    if point1[3] != point2[3]:
        return point1[2]-point2[2]
    return point1[0]-0

def set_frames_64(data, trail):
    np_frames = split_to_frames(data)
    all_frames = np.empty((0, 64, 5))
    seen_frames = []
    old_frames = []
    all_valid_frames = set()
    for i,frame in enumerate(np_frames):
        frame_id = frame[0][0]
        frame = filter_x_y_z_axis(frame, [-1,2], [1, 6], [-1.5, 0.5])[:,1:] # configed for mindspace pufim room
        if frame.shape[0] == 0:
            continue
        seen_frames.append(frame_id)
        old_frames.append(frame)
        old_frames = old_frames[-trail:]
        if len(seen_frames) >= trail and valid_trail(seen_frames[-trail:], trail):
            # if frame_id == 9961:
            #     pdb.set_trace()
            # frame = dbscan_filter_frame(frame)
            if len(frame) > 0:
                frame = np.array(sorted(frame, key = functools.cmp_to_key(sort_xyz)))
                if frame.shape[0] < 64:
                    frame = np.concatenate((frame, np.zeros((64-frame.shape[0], frame.shape[1]))), axis= 0)
                if frame.shape[0] > 64:
                    frame = frame[:64, :]
                all_frames = np.append(all_frames, np.array([frame]), axis = 0)
                if trail != 0:
                    frame = np.concatenate(old_frames[-trail:], axis = 0)
                    all_valid_frames.add(seen_frames[-trail//2])
                else:
                    all_valid_frames.add(seen_frames[-1])
            else:
                print(f"dbscan filtered frame num: {frame_id}")
    print(f"data set size: {all_frames.shape[0]}")
    return all_frames, all_valid_frames

def valid_trail(last_frames, trail):
    if trail == 0:
        return True
    for i in range(trail - 1):
        if last_frames[trail-1-i] - last_frames[trail-2-i] != 1:
            return False
    return True

def frame_to_np(frame):
    rows = frame.shape[0]
    frame = np.moveaxis(frame, [0, 1, 2], [0, 2, 1])
    frame = frame.reshape((rows, 5, 8, 8))
    frame = np.moveaxis(frame, [0, 1, 2, 3], [0, 3, 2, 1])
    return frame

def run_preprocess(data_csv_file, trail = 0):
    df = pd.read_csv(data_csv_file)
    data, valid_frames = set_frames_64(df, trail)
    np_data = frame_to_np(data)
    return np_data, valid_frames

def pick_closesd_one(annotations):
    if len(annotations) == 1:
        return annotations[0]
    #get best one by x axis distance
    cur_array = sorted(annotations, key = lambda x: np.nanmean(np.array(x["pose3d"])[:,0]), reverse= True)
    return cur_array[0]

def make_row(pose_array, data):
    x_axis = pose_array[:,0]
    y_axis = pose_array[:,1]
    z_axis = pose_array[:,2] * -1
    pose = np.concatenate((x_axis, y_axis, z_axis), axis = 0)
    nan_vals = np.argwhere(pd.isnull(pose))
    pose[nan_vals] = data[-1][nan_vals]
    return pose



def to_npy( jsons, valid_map, feature_size = 14):
    data = np.zeros((len(valid_map), feature_size * 3))
    labels = {}
    for f in jsons:
        labels.update(f)
    for i, index in enumerate(valid_map):
        key = str(int(index)+1)
        frame_annotation = labels[key].copy()
        if frame_annotation["annotations"] and frame_annotation["annotations"][0]["pose3d"]:
            best_annotation = pick_closesd_one(frame_annotation["annotations"])
            if i != 0 :
                pose = make_row(np.array(best_annotation["pose3d"])[:feature_size], data)
            else:
                pose = np.nan_to_num(np.array(best_annotation["pose3d"])[:feature_size])
                x_axis = pose[:,0]
                y_axis = pose[:,1]
                z_axis = pose[:,2] * -1
                pose = np.concatenate((x_axis, y_axis, z_axis), axis = 0)
            data[i] = pose
        else:
            if i != 0:
                data[i] = data[i-1]
    return data

