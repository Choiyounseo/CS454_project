import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class ClassifierB(nn.Module):
    def __init__(self):
        super(ClassifierB, self).__init__()

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 64, kernel_size=8, stride = 2, padding = 3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size=6, stride = 2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels = 128, kernel_size=5, stride = 1)
        self.dropout2 = nn.Dropout(0.5)
        self.FC1 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()


    def forward(self, x):
        in_size = x.size(0)
        x = self.dropout1(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(in_size, -1)
        x = self.FC1(x)
        x = self.softmax(x)

        return x