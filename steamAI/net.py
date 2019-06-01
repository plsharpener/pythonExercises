import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data.dataset as dataset
import torch.optim as optim

batch_size = 100
epochs = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
    """网络"""
    def __init__(self):
        super(net,self).__init__()
        self.fc = []
        self.fc11 = nn.Linear(7,7)
        self.fc.append(self.fc11)
        self.fc12 = nn.Linear(7,7)
        self.fc.append(self.fc12)
        self.fc13 = nn.Linear(7,7)
        self.fc.append(self.fc13)
        self.fc14 = nn.Linear(7,7)
        self.fc.append(self.fc14)
        self.fc15 = nn.Linear(7,7)
        self.fc.append(self.fc15)
        self.fc16 = nn.Linear(3,3)
        self.fc.append(self.fc16)
        self.MergeVertex = nn.Linear(38,50)
        self.dropout = nn.Dropout(p=0.96)
        self.fc3 = nn.Linear(50,30)
        self.fc4 = nn.Linear(30,20)
        self.fc5 = nn.Linear(20,1)

    def forward(self,x):
        """前向传播"""
        iput = []
        oput = []
        for i in range(5):
            iput.append(x[i*7:(i+1)*7])
        iput.append(x[35:38])
        for i,j in enumerate(iput):
            a = self.fc[i](j)
            oput.append(F.relu(a))
        x = torch.cat([oput[0],oput[1],oput[2],oput[3],oput[4],oput[5]],dim=0)
        # x = Variable(torch.tensor(x,dtype=torch.float))
        x = F.relu(self.MergeVertex(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train(model,device,train_loader,optimizer,epochs):
    """训练"""
    # model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        print(data,target) 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1) %30 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))




if __name__ == "__main__":
    model = net().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    datas = dataset.traindataset("./data/zhengqi_train.txt")
    trainloader = torch.utils.data.DataLoader(datas,batch_size=batch_size,shuffle=True,num_workers=0)
    for epoch in range(1,epochs+1):
        train(net,DEVICE,trainloader,optimizer,epoch)
    # traindatas,terget = datas[0] 
    # NET = net()
    # print(NET(traindatas))
    # data = iter(trainloader)
    # print(next(data))