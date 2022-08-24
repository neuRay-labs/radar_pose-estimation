import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class torch_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(8, 16, stride=1, padding='same')
        self.conv2 = nn.Conv2d(16, 32, padding='same')
        self.drop = nn.Dropout(0.3)
        self.flatten = nn.Flatten(start_dim=1)
        self.batch_norm1 = nn.BatchNorm2d(32, momentum=0.95)
        self.batch_norm2 = nn.BatchNorm2d(512, momentum=0.95)
        self.drop2 = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(_, 512)
        self.fc2 = nn.Linear(512, 75)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.drop(x)))
        x = self.batch_norm1(self.drop(x))
        x = self.flatten(x)
        import pdb;pdb.set_trace()
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.drop2(x)
        x = self.fc2(x)
