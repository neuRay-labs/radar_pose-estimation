import os
import torch
import json
import glob
import numpy as np
import preprocessor


class PreparePoseDataSet():
    def __init__(self, body_points_path, pc_path, trail, feature_size) -> None:
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
        self.all_subjects_pc = None
        self.all_subjects_body = None
        self.valid_frames = {}
        self.trail = trail
        self.extract_files(self.main_pc_path, "csv")
        self.extract_files(self.main_body_path, "json")
        self.feature_size = feature_size
        self.data_size = None

    def extract_files(self, path, ext):
        sub_folders = glob.glob(os.path.join(path,"*"))
        for folder in sub_folders:
            head_path, name = os.path.split(folder)
            sub_name = name.split("_")[0]
            dataset_array = glob.glob(os.path.join(folder,f"*.{ext}"))
            if ext == "csv" and len(dataset_array) > 0:
                dataset_target = [d for d in dataset_array if ".ts" not in d][0]
                if self.all_subjects_pc is None:
                    self.all_subjects_pc, self.valid_frames[sub_name] = preprocessor.run_preprocess(dataset_target, self.trail)
                else:
                    new_dataset, self.valid_frames[sub_name] = preprocessor.run_preprocess(dataset_target, self.trail)
                    self.all_subjects_pc = np.concatenate((self.all_subjects_pc, new_dataset), axis = 0)
            if ext == 'json':
                dataset_array = [self.read_json(f) for f in dataset_array]
                if self.all_subjects_body is None:
                    self.all_subjects_body = preprocessor.to_npy(dataset_array, self.valid_frames[sub_name])
                else:
                    new_dataset  = preprocessor.to_npy(dataset_array, self.valid_frames[sub_name], self.feature_size)
                    self.all_subjects_body = np.concatenate((self.all_subjects_body, new_dataset), axis = 0)


    def read_json(self, file):
        with open(file) as j_file:
            data = json.load(j_file)
        return data

    def save_data(self, train_percent, output_path):
        size = (len(self.all_subjects_pc) * train_percent )//100
        if self.all_subjects_pc is not None:
            featuremap_train, featuremap_test = self.all_subjects_pc[:size,:], self.all_subjects_pc[size:,:]
            np.save(os.path.join(output_path, "featuremap_train.npy"), featuremap_train)
            np.save(os.path.join(output_path, "featuremap_test.npy"), featuremap_test)
        if self.all_subjects_body is not None:
            label_train , label_test = self.all_subjects_body[:size,:], self.all_subjects_body[size:,:]
            np.save(os.path.join(output_path, "labelmap_train.npy"), label_train)
            np.save(os.path.join(output_path, "labelmap_test.npy"), label_test)

        print (f"Successfuly saves processed dataset to {output_path}")


    def __len__(self):
        return len(self.all_subjects_body)
    
    