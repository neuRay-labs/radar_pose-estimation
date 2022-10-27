import pdb
import pandas as pd
import numpy as np
from test_pre_trained import split_to_frames, filter_x_y_z_axis
import functools

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
        frame_id = frame[0][1]
        frame = filter_x_y_z_axis(frame, [-2,2], [1, 10], [-1.6, 0.7])[:,2:]
        if frame.shape[0] == 0:
            continue
        seen_frames.append(frame_id)
        old_frames.append(frame)
        old_frames = old_frames[-trail:]
        if len(seen_frames) >= trail and valid_trail(seen_frames[-trail:], trail):
            frame = np.concatenate(old_frames[-trail:], axis = 0)
            all_valid_frames.add(seen_frames[-trail//2])
            cur_frame = np.zeros((64, frame.shape[1]))
            cur_frame[:min(64, frame.shape[0]),:] = frame[-64:,:]
            frame = np.array(sorted(frame, key = functools.cmp_to_key(sort_xyz)))
            all_frames = np.append(all_frames, np.array([cur_frame]), axis = 0)
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
    return sorted(annotations, key = lambda x: np.average(np.array(x["pose3d"])[:,0]))[0]

def to_npy( jsons, valid_map):
    feature_size = 54
    data = np.zeros((len(valid_map), feature_size))
    i = 0
    for f in jsons:
        for j in f.values():
            if j["frame_id"] in valid_map:
                if j["annotations"] and j["annotations"][0]["pose3d"]:
                    best_annotation = pick_closesd_one(j["annotations"])
                    pose = np.nan_to_num(np.array(best_annotation["pose3d"])[:18])
                    x_axis = pose[:,0]
                    y_axis = pose[:,1]
                    z_axis = pose[:,2] * -1
                    pose = np.concatenate((x_axis, y_axis, z_axis), axis = 0)
                    data[i] = pose
                i +=1 
    return data

