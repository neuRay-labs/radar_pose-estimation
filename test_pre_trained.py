from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf



# FRAME_ID = "Frame #"
FRAME_ID = "frame_id"

def split_to_frames(data_set):
    gb = data_set.groupby(FRAME_ID)
    return [gb.get_group(x).to_numpy() for x in gb.groups]

def pose_to_np(pose, feature_size):
    # pose = pose.array
    x_array = pose[:feature_size].reshape((feature_size, 1))
    y_array = pose[feature_size:2*feature_size].reshape((feature_size, 1))
    z_array = pose[2*feature_size:3*feature_size].reshape((feature_size, 1))
    return np.concatenate((x_array, y_array, z_array), axis=1)
    
def frame_to_np(frame):
    frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
    frame = frame.reshape(5, -1)
    frame = np.moveaxis(frame, [0, 1], [1, 0])
    return frame

def filter_x_y_z_axis(data,  x_filter=[-np.inf, np.inf], y_filter=[-np.inf, np.inf], z_filter=[-np.inf, np.inf]):
        filtered_data = data.copy()        
        filtered_data = filtered_data[(filtered_data[:,1] > x_filter[0]) & (filtered_data[:,1] < x_filter[1])]
        filtered_data = filtered_data[(filtered_data[:,2] > y_filter[0]) & (filtered_data[:,2] < y_filter[1])]
        filtered_data = filtered_data[(filtered_data[:,3] > z_filter[0]) & (filtered_data[:,3] < z_filter[1])]
        return filtered_data

def visualize_results(mmwave_csv, pose_csv):
    
    mmwave = pd.read_csv(mmwave_csv)
    pose = pd.read_csv(pose_csv).to_numpy()
    frames = split_to_frames(mmwave)
    assert len(frames) == pose.shape[0], "prediction must have same amount as samples"
    show_plot(frames, pose, 25)


def show_plot(frames, pose, feature_size):
    fig = plt.figure(figsize=(8, 8))
    nice = Axes3D(fig)
    for f, p in zip(frames, pose):
        p = pose_to_np(p, feature_size)
        if len(f.shape) > 2:
            f = frame_to_np(f)
        else:
            f = f[:,2:7]
        fx, fy, fz = f[:,0].T, f[:,1].T, f[:,2].T
        px, py, pz = p[:,0].T, p[:,1].T, p[:,2].T
        # import pdb; pdb.set_trace()
        nice.set_zlim3d(bottom=-3, top=0.5)
        nice.set_ylim(bottom=-2, top=2)
        nice.set_xlim(left=-2, right=2)
        nice.set_xlabel('X Label')
        nice.set_ylabel('Y Label')
        nice.set_zlabel('Z Label')
        nice.scatter(fx, fy, fz, color="red", marker='o')
        nice.scatter(px, py, pz, color="blue", marker='o')
        plt.pause(0.1)
        nice.clear()
        nice.grid(False)


def get_prediction(pre_trained_path, featuremap_test_path):
    model = tf.keras.models.load_model(pre_trained_path)
    featuremap_test = np.load(featuremap_test_path)
    result_test = model.predict(featuremap_test)
    return result_test, featuremap_test  


if __name__ == "__main__":
    # visualize_results(r"synced_data\woutlier\subject1\radar_data_all.csv", r"synced_data\woutlier\subject1\kinect_data_all.csv")
    result_test, featuremap_test = get_prediction("model/MARS_test.h5", "MARS_Elad.npy")
    show_plot(featuremap_test, result_test, 19)
