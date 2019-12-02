import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class ClassifierC(nn.Module):
    def __init__(self):
        super(ClassifierC, self).__init__()

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 128, kernel_size=3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels = 64, kernel_size=3, stride = 2, padding=2)
        self.dropout1 = nn.Dropout(0.25)
        self.FC1 = nn.Linear(64*15*15, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.FC2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        in_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(-1, 64*15*15)
        x = self.relu(self.FC1(x))
        x = self.dropout2(x)
        x = self.FC2(x)
        x = self.softmax(x)

        return x