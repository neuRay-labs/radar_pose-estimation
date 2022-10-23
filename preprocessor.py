import pandas as pd
import numpy as np
from test_pre_trained import split_to_frames, filter_x_y_z_axis
import functools



        # self.pose_estimation_map = pd.DataFrame(columns=["idx", "head_x", "spine_base_x", "shoulder_right_x", "elbow_right_x",
        #                                                  "wrist_right_x", "shoulder_left_x", "elbow_left_x", "wrist_left_x", 
        #                                                 "hip_right_x", "knee_right_x", "foot_right_x", "hip_left_x", "knee_left_x",
        #                                                  "foot_left_x", "head_y", "spine_base_y", "shoulder_right_y", "elbow_right_y",
        #                                                  "wrist_right_y", "shoulder_left_y", "elbow_left_y", "wrist_left_y", 
        #                                                 "hip_right_y", "knee_right_y", "foot_right_y", "hip_left_y", "knee_left_y",
        #                                                  "foot_left_y",  "head_z", "spine_base_z", "shoulder_right_z", "elbow_right_z",
        #                                                  "wrist_right_z", "shoulder_left_z", "elbow_left_z", "wrist_left_z", 
        #                                                 "hip_right_z", "knee_right_z", "foot_right_z", "hip_left_z", "knee_left_z",
        #                                                  "foot_left_z", "point_cloud_idx" ])
        
def sort_xyz(self, point1, point2):
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
        frame = filter_x_y_z_axis(frame, [-2,2], [1, 10], [-1.6, 0.7])[:,1:]
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

    def to_npy( jsons):
        def calculate_size():
            size = 1000 * (len(jsons)-1)
            return size + len(jsons[-1])
        size = calculate_size()
        data = np.zeros((size, 42))
        i = 0
        for f in jsons:
            for j in f.values():
                if j["annotations"] and j["annotations"][0]["pose3d"]:
                    pose = np.array(j["annotations"][0]["pose3d"])[:14].reshape((1,42))
                    data[i] = pose
                i +=1 
        return data

