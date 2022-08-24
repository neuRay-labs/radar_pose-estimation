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

def set_frames_64(data):
    np_frames = split_to_frames(data)
    all_frames = np.empty((0, 64, 5))

    for i,frame in enumerate(np_frames):
        frame = filter_x_y_z_axis(frame, [-1,1], [2.5, 5], [-1.2, 0.7])[:,1:]
        rows_count = frame.shape[0]
        if rows_count == 0:
            continue
        if rows_count < 64:
            frame = np.concatenate((frame, np.zeros((64-rows_count, frame.shape[1]))), axis=0)
        else:
            frame = frame[:64, :]
        frame = np.array(sorted(frame, key = functools.cmp_to_key(sort_xyz)))
        all_frames = np.append(all_frames, np.array([frame]), axis = 0)
    
    return all_frames

def frame_to_np(frame):
    rows = frame.shape[0]
    frame = np.moveaxis(frame, [0, 1, 2], [0, 2, 1])
    frame = frame.reshape((rows, 5, 8, 8))
    frame = np.moveaxis(frame, [0, 1, 2, 3], [0, 3, 2, 1])
    return frame

def run_preprocess(data_csv_file):
    df = pd.read_csv(data_csv_file)
    data = set_frames_64(df)
    np_data = frame_to_np(data)
    return np_data


# data = np.load(r"D:\multi_env\parking_1\csv_pc\split_removed_close_frames\elad_16.npy")

final = run_preprocess(r"D:\multi_env\garden\csv_pc\Elad.csv")
print(final.shape)
np.save("MARS_Elad.npy", final)