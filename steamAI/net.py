import torch
import numpy as np
import torch.nn as nn

class net(nn.Module):
    """网络"""
    def __init__(self):
        super(net,self).__init__()
        self.fc1 = nn.Linear(7,7)
        self.fc2 = nn.Linear(3,3)
        self.MergeVertex = nn.Linear(38,50)
        self.fc3 = nn.Linear(50,30)
        self.fc4 = nn.Linear(30,20)
        self.fc5 = nn.Linear(20,1)

    def forward(self,x):
        """前向传播"""
        for i in range(6):
            

    def DecentralizedParameter(self,x):
        """分散参数"""
        