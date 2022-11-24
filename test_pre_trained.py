from tkinter.messagebox import showerror
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pose_estimation_model import PoseEstimation
import torch


# FRAME_ID = "Frame #"
FRAME_ID = "frame_id"

def split_to_frames(data_set):
    gb = data_set.groupby(FRAME_ID)
    return [gb.get_group(x).to_numpy() for x in gb.groups]

def pose_to_np(pose, feature_size):
    x_array = pose[:feature_size].reshape((feature_size, 1))[:feature_size]
    y_array = pose[feature_size:2*feature_size].reshape((feature_size, 1))[:feature_size]
    z_array = pose[2*feature_size:3*feature_size].reshape((feature_size, 1))[:feature_size]
    return np.concatenate((x_array, y_array, z_array), axis=1)
    
def frame_to_np(frame):
    frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
    frame = frame.reshape(5, -1)
    frame = np.moveaxis(frame, [0, 1], [1, 0])
    return frame

def visualize_results(mmwave_csv, pose_csv):
    
    mmwave = pd.read_csv(mmwave_csv)
    pose = pd.read_csv(pose_csv).to_numpy()
    frames = split_to_frames(mmwave)
    assert len(frames) == pose.shape[0], "prediction must have same amount as samples"
    show_plot(frames, pose, 25)

def visualize_label(pose_npy, pose_model_out = None):
    show_plot(None, pose_npy,  18)
    if pose_model_out is not None:
        show_plot(None, pose_model_out,18)

def show_plot(frames, pose, feature_size):
    fig = plt.figure(figsize=(8, 8))
    nice = Axes3D(fig)
    if pose is not None:
        existing_data = pose
    elif frames is not None:
        existing_data = frames
    else:
        existing_data = np.empty((0))
    for i in range(len(existing_data)):
        # if f.shape[0] != 0 and p.shape[0]!= 0:
        if pose is not None:
            p = pose_to_np(pose[i], feature_size)
            px, py, pz = p[:,0].T, p[:,1].T, p[:,2].T
        if frames is not None:
            f = frames[i]
            if len(f.shape) > 2:
                f = frame_to_np(frames[i])
            else:
                f = f[:,2:7] 
            fx, fy, fz = f[:,0].T, f[:,1].T, f[:,2].T
        nice.set_zlim3d(bottom=-1.6, top=1)
        nice.set_ylim(bottom=0, top=12)
        nice.set_xlim(left=-2, right=2)
        nice.set_xlabel('X Label')
        nice.set_ylabel('Y Label')
        nice.set_zlabel('Z Label')
        if frames is not None:
            nice.scatter(fx, fy, fz, color="red", marker='o')
        if pose is not None:
            nice.scatter(px, py, pz, color="blue", marker='o')
        plt.pause(0.1)
        nice.clear()
        nice.grid(False)


def get_prediction(pre_trained_path, featuremap_test_path, feature_size):
    featuremap_test = torch.tensor(np.load(featuremap_test_path)).double()
    model = PoseEstimation(feature_size).double()
    model.load_state_dict(torch.load(pre_trained_path))
    model.eval()
    result_test = model(featuremap_test)
    # result_test = np.load(r"E:\Radar\22_11\single_person\6_meter\labelmap_test.npy")
    print(result_test.shape)
    return result_test.detach().numpy()   , featuremap_test.detach().numpy() #

# visualize_label(np.load(r"E:\Radar\pose_estimation_rec\30_10\labelmap_test.npy"))