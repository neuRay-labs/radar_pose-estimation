import os
import torch
import torch
import json
import glob
import preprocessor
from torch.utils.data import Dataset


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
        self.valid_frames = {}
        self.extract_files(self.main_pc_path, self.all_subjects_pc, "csv")
        self.extract_files(self.main_body_path, self.all_subjects_body, "json")
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
                        dictionary[sub_name], self.valid_frames[sub_name] = preprocessor.run_preprocess(file)
            if ext == 'json':
                dictionary[sub_name] = [self.read_json(f) for f in dictionary[sub_name]]
                dictionary[sub_name] = preprocessor.to_npy(dictionary[sub_name], self.valid_frames[sub_name])
                self.update_map(sub_name, dictionary)


    def update_map(self, name, dictionary):
        if self.map.keys():
            self.map[name] = [self.map[str(int(name)-1)][1] + 1,self.map[str(int(name)-1)][1] + dictionary[name].shape[0] - 1]
            return
        else:
            self.map[name] = [0, dictionary[name].shape[0] -1]
            return


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
            sum += len(self.all_subjects_body[subject])
        self.data_size = sum
        return sum
    
    def __getitem__(self, idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject = self.find_subject(idx)
        new_idx = idx - self.map[subject][0]
        return {"data" : torch.tensor(self.all_subjects_pc[subject][new_idx]),
               "label" : torch.tensor(self.all_subjects_body[subject][new_idx])}

            
