import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
from pose_estimation_model import PoseEstimation
from clearml.utilities.seed import make_deterministic
import clearml
import json
from collections import defaultdict
import random
from Mars_dataset import PoseDataSetMars

now = datetime.datetime.now()

ACCURACY = "accuracy"
LOSS = "loss"
CONFUSION = "confusion"
RUNNING_LOSS = "Running Loss"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
MODEL_SNAPSHOT_FILE_NAME_TEMPLATE = "model_snapshot_{epoch}.pth"

class ModelTrainer():
    def __init__(self, args) -> None:
        self.clearml_task_name = args.clearml_task_name
        self.feature_size = args.feature_size
        self.train_path_radar = args.point_cloud_repo_train
        self.train_path_label = args.pose_label_repo_train
        self.test_path_radar = args.point_cloud_repo_test
        self.test_path_label = args.pose_label_repo_test
        self.output_path = args.output_path
        self.no_validation = args.no_validation
        self.train_percentage = args.train_percentage
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.metrics_report_interval = args.metrics_report_interval
        self.learning_rate = args.learning_rate
        self.no_clearml = args.no_clearml
        self.seed = args.seed
        self.cuda_device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.test_dataset = None
        self.train_dataset = None
        self.args = args
        print(vars(args))

        # if not self.no_clearml:
        #     self.init_clearml()
        self.datasets = []
        
        self.datasets.append(TEST)

        if self.train_path_label:
            self.datasets.append(TRAIN)
            if self.test_path_radar and not self.no_validation:
                self.datasets.append(VALIDATION)
        self.model = PoseEstimation(self.feature_size).double()
        print(f'number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
    
        self.model.to(self.cuda_device)
        
        self.iteration = 0
        self.epoch = 0

        trained_params = self.model.parameters()
        self.optimizer = optim.Adam(trained_params, lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.metrics = {
            TRAIN: {
                ACCURACY: None,
                LOSS: None,
                CONFUSION: None,
            },
            VALIDATION: {
                ACCURACY: None,
                LOSS: None,
                CONFUSION: None,
            },
            TEST: {
                ACCURACY: None,
                LOSS: None,
                CONFUSION: None,
            },
        }
        self.data_loaders = {}
        self.load_dataset(PoseDataSetMars)
        self.data_loaders[TRAIN] = self.get_dataloader(self.train_dataset)
        self.data_loaders[TEST] = self.get_dataloader(self.test_dataset)
        
        self.current_folder = os.path.join(self.output_path, f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{self.clearml_task_name}")
        os.makedirs(self.current_folder)


    def load_dataset(self, dataLoader):
        if self.train_path_label:
            pose_data_set = dataLoader(body_points_path = self.train_path_label, pc_path=self.train_path_radar)
            if not self.test_path_radar:
                train_size = int((self.train_percentage / 100) * len(pose_data_set))
                test_size = len(pose_data_set) - train_size
                self.train_dataset, self.test_dataset = torch.utils.data.random_split(pose_data_set, [train_size, test_size])
            else:
                self.train_dataset = pose_data_set
                self.test_dataset = dataLoader(body_points_path = self.test_path_label, pc_path=self.test_path_radar)

    def get_dataloader(self, database):
        return DataLoader(database, batch_size=self.batch_size, shuffle=True)
    
    def init_clearml(self):
        config = {'preprocess': defaultdict(dict)}
        all_paths = list(set(self.train_path_label + self.test_path_radar))
        for path in all_paths:
            config_path = os.path.join(path, "preprocess.json")
            with open(config_path, "r") as f:
                key = ''
                if path in self.train_path_label and path in self.test_path_radar:
                    key = 'train&test'
                elif path in self.train_path_label:
                    key = 'train'
                elif path in self.test_path_radar:
                    key = 'test'
                config['preprocess'][key][path] = json.load(f)

        # super ugly hack, since clearml uses the same seed by default. horrible!
        if self.seed == -1:
            self.seed = random.randint(0, 1e10)
        self.clearml_task = clearml.Task.init(project_name='pose estimation', task_name=f"{self.clearml_task_name}, {str(now)}")
        print(f"current seed: {self.seed}")
        make_deterministic(seed=self.seed)

        current_config = vars(self.args)
        if self.args.command_line:
            config.pop('func', None)
            config.pop('subcommand', None)
        else:
            self.args.clearml_task = self.clearml_task
        config.update(current_config)
        self.clearml_task.connect(config)


    def calculate_metrics(self):
        for dataset in self.datasets:
            loss= self.get_accuracy_and_loss(dataset)
            if dataset == TEST:
                print (f"CURRENT TEST LOSS: {loss}")
            self.metrics[dataset][LOSS] = loss
    
    def get_accuracy_and_loss(self, dataset):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0
            for data in self.data_loaders[dataset]:
                inputs, labels = data['data'], data['label']
                inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
                outputs = self.model(inputs)
                running_loss += self.criterion(outputs, labels).cpu().item()
            running_loss /= len(self.data_loaders[dataset])
            print(f"data_set size: {len(self.data_loaders[dataset])}")
        self.model.train()
        return  running_loss 
    
    def report_metrics(self):
        for dataset in self.datasets:
            self.clearml_task.logger.report_scalar(
                ACCURACY,
                dataset,
                iteration=self.iteration,
                value=self.metrics[dataset][ACCURACY],
            )
            self.clearml_task.logger.report_scalar(
                LOSS,
                dataset,
                iteration=self.iteration,
                value=self.metrics[dataset][LOSS],
            )
            

    def report_loss(self, loss):
        self.clearml_task.logger.report_scalar(
            RUNNING_LOSS,
            LOSS,
            iteration=self.iteration,
            value=loss,
        )

    def snapshot_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.current_folder, MODEL_SNAPSHOT_FILE_NAME_TEMPLATE.format(epoch=self.epoch)))
    
    def test(self):
        accuracy, loss = self.get_accuracy_and_loss(TEST)
        if self.no_clearml:
            return

        self.clearml_task.logger.report_scalar(
            ACCURACY,
            TEST,
            iteration=0,
            value=accuracy,
        )
        self.clearml_task.logger.report_scalar(
            LOSS,
            TEST,
            iteration=0,
            value=loss,
        )

    
    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            if epoch % self.metrics_report_interval == 0:
                self.calculate_metrics()
                if not self.no_clearml:
                    self.report_metrics()
                self.snapshot_model()

            running_loss = 0.0
            for i, data in enumerate(self.data_loaders[TRAIN]):
                self.iteration += 1
                inputs, labels = data['data'], data['label']
                inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 50 == 49:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.5f}")
                    running_loss = 0

