import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(28*28, 50)
        self.l2 = nn.Linear(50, 25)
        self.l3 = nn.Linear(25, 10)

    def forward(self, x):
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        return self.l3(h2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.l1 = nn.Linear(12*12*64, 50)
        self.l2 = nn.Linear(50, 25)
        self.l3 = nn.Linear(25, 10)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = self.pool(F.relu(self.conv2(h1)))
        h2 = h2.view(-1, 12*12*64)
        h3 = torch.tanh(self.l1(h2))
        h4 = torch.tanh(self.l2(h3))
        return self.l3(h4)
