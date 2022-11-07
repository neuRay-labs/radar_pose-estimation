import sklearn.cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dbscan_filter_frame(frame, p=3, min_samples=2, eps=0.5):
    if len(frame) < 1:
        return frame
    dbscan = sklearn.cluster.DBSCAN(p=p, min_samples=min_samples, eps=eps)
    clustering = dbscan.fit(frame[:,:3])
    labels = clustering.labels_
    # max_label = max(labels)
    # best_group_x = np.inf
    # best_group = None
    # largest_group = None
    # groups = []
    filtered = frame[labels != -1]
    if len(filtered) > len(frame):
        return filtered
    # labels = labels[labels != -1]
    # for group in range(max_label + 1):
    #     group_points = frame[labels == group]
    #     if largest_group is None:
    #         largest_group = group_points
    #     groups.append(group_points)
    #     if abs(group_points[:,0].mean()) < abs(best_group_x):
    #         best_group = group
    #         best_group_x = abs(group_points[:,0].mean())
    #     if largest_group is not None and group_points.shape[0] > largest_group.shape[0]:
    #         largest_group = group_points
    # if best_group is not None:
    #     labels -= best_group
    #     labels = abs(labels)
    return frame
    
    
def filter_x_y_z_axis(data,  x_filter=[-np.inf, np.inf], y_filter=[-np.inf, np.inf], z_filter=[-np.inf, np.inf]):
        filtered_data = data.copy()        
        filtered_data = filtered_data[(filtered_data[:,1] > x_filter[0]) & (filtered_data[:,1] < x_filter[1])]
        filtered_data = filtered_data[(filtered_data[:,2] > y_filter[0]) & (filtered_data[:,2] < y_filter[1])]
        filtered_data = filtered_data[(filtered_data[:,3] > z_filter[0]) & (filtered_data[:,3] < z_filter[1])]
        return filtered_data
