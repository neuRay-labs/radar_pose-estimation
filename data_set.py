import os
import torch
import pandas as pd
import numpy as np
import torch
import json
import glob
import preprocessor
from torch.utils.data import Dataset

ANNOTATIONS_FILE_SIZE = 1000

class PoseDataSet(Dataset):
    def __init__(self, body_points_path, pc_path) -> None:
        """
        the PoseDataSet is expecting to find files in the following hierarchy:
        body_poinst_path:
        --subject_name(directory)
        ----file_0
        ----file_1
        
        pc_path:
        --subjec_name_dir
        ----pc_file.csv
        
        """
        super().__init__()
        self.main_body_path = body_points_path
        self.main_pc_path = pc_path
        self.all_subjects_body = {}
        self.all_subjects_pc = {}
        self.map = {}
        self.extract_files(self.main_body_path, self.all_subjects_body, "json")
        self.extract_files(self.main_pc_path, self.all_subjects_pc, "csv")
        self.data_size = None

    def extract_files(self, path, dictionary, ext):
        sub_folders = glob.glob(os.path.join(path,"*"))
        for folder in sub_folders:
            head_path, name = os.path.split(folder)
            sub_name = name.split("_")[0]
            dictionary[sub_name] = glob.glob(os.path.join(folder,f"*.{ext}"))
            if ext == "csv" and len(dictionary[sub_name]) > 0:
                for file in dictionary[sub_name]:
                    if not ".ts." in file:
                        dictionary[sub_name] = preprocessor.run_preprocess(file)
            if ext == 'json':
                dictionary[sub_name] = [self.read_json(f) for f in dictionary[sub_name]]
                dictionary[sub_name] = preprocessor.to_npy(dictionary[sub_name])
                self.map[sub_name] = [self.map[sub_name-1],self.map[sub_name-1] + dictionary[sub_name].shape[0]]
            if ext == 'npy':
                dictionary[sub_name] = np.load(dictionary[sub_name][0])
                self.map[sub_name] = [self.map[sub_name-1],self.map[sub_name-1] + dictionary[sub_name].shape[0]]


    def read_json(self, file):
        with open(file) as j_file:
            data = json.load(j_file)
        return data
    
    def find_subject(self, idx):
        for s in self.map.keys():
            if self.map[s][0] <= idx <= self.map[s][1]: 
                return s

    def __len__(self):
        if self.data_size is not None:
            return self.data_size
        sum = 0
        for subject in self.all_subjects_body.keys():
            last_obj = self.all_subjects_body[subject][-1]
            sum += max(last_obj.keys())
        self.data_size = sum
        return sum
    
    def __getitem__(self, idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject = self.find_subject(idx)
        new_idx = idx - self.map[subject][0]
        return {"data" : torch.tensor(self.all_subjects_pc[subject][new_idx]),
               "label" : torch.tensor(self.all_subjects_body[subject][new_idx])}

            
# a = PoseDataSet(r"C:\Users\NitzanKarby\Desktop\neuRay\experiment_data\Radar_data\pose_estimation\annotations", r"C:\Users\NitzanKarby\Desktop\neuRay\experiment_data\Radar_data\pose_estimation\recordings")