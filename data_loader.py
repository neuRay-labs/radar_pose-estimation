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
        self.extract_files(self.main_body_path, self.all_subjects_body, "json")
        self.extract_files(self.main_pc_path, self.all_subjects_pc, "csv")
        self.map = pd.DataFrame(columns=["id", "frame", "label"])
        self.data_size = None

    
    def map_label_sample(self):
        n = len(self.map)


    def extract_files(self, path, dictionary, ext):
        sub_folders = glob.glob(os.path.join(path,"*"))
        for folder in sub_folders:
            head_path, name = os.path.split(folder)
            # sub_name = name.split(".")[0]
            sub_name = name.split("_")[0]
            dictionary[sub_name] = glob.glob(os.path.join(folder,f"*.{ext}"))
            if ext == "csv" and len(dictionary[sub_name]) > 0:
                for file in dictionary[sub_name]:
                    if not ".ts." in file:
                        dictionary[sub_name] = preprocessor.run_preprocess(file)
                        import pdb;pdb.set_trace()
                        break
            elif ext == 'json':
                dictionary[sub_name] = [self.read_json(f) for f in dictionary[sub_name]]


    def read_json(self, file):
        with open(file) as j_file:
            data = json.load(j_file)
        return data
    
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
        file_from_annotations = idx // ANNOTATIONS_FILE_SIZE
        inside_file = idx % ANNOTATIONS_FILE_SIZE


            
a = PoseDataSet(r"D:\pose_estimation\18_10", r"E:\Radar\pose_estimation_rec")