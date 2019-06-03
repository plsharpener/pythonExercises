import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data.dataset as dataset
import torch.optim as optim
import time
# import pysnooper

batch_size = 5
epochs = 1500
batch_size_test = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
    """网络"""
    def __init__(self):
        super(net,self).__init__()
        self.fc1 = nn.Linear(7,7)
        self.fc2 = nn.Linear(3,3)
        self.MergeVertex = nn.Linear(38,50)
        self.dropout = nn.Dropout(p=0.96)
        self.fc3 = nn.Linear(50,30)
        self.fc4 = nn.Linear(30,20)
        self.fc5 = nn.Linear(20,1)
    def forward(self,x):
        """前向传播"""
        output = []
        for i in x.chunk(6,dim=1)[:-1]:
            output.append(self.fc1(i))
        output.append(self.fc2(x.chunk(6,dim=1)[-1]))
        x = torch.cat(output,dim=1)
        # x = Variable(torch.tensor(x,dtype=torch.float))
        x = F.relu(self.MergeVertex(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class net2(nn.Module):
    def __init__(self):
        super(net2,self).__init__()
        self.fc1 = nn.Linear(38,38)
        # self.fc2 = nn.Linear(bitch_size*38,batch_size*38)
        self.fc2 = nn.Linear(38,50)
        self.dropout = nn.Dropout(p=0.96)
        self.fc3 = nn.Linear(50,30)
        self.fc4 = nn.Linear(30,20)
        self.fc5 = nn.Linear(20,1)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# @pysnooper.snoop(output="./log/log.log")
def train(model,device,train_loader,optimizer,epochs,lossfuc):
    """训练"""
    # model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        # print("data:",data.shape) 
        # print("target:",target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        # output = torch.unsqueeze(output,0)
        # print(output)
        # print("output.shape:{},target.shape:{}".format(output.shape,target.shape))
        loss = lossfuc(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1) %30 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model,device,test_loader):
    """输出结果"""
    outputs = []
    for batch_idx,data in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        # outputs.append(list(output.numpy()))
        outputs.append(output)
        # if batch_idx == 10:
        #     break
    return outputs

    



if __name__ == "__main__":
    model = net().to(DEVICE)
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    datas = dataset.traindataset("./data/zhengqi_train.txt",train=True)
    trainloader = torch.utils.data.DataLoader(datas,batch_size=batch_size,shuffle=True,num_workers=0)
    lossfuc = nn.MSELoss()
    for epoch in range(1,epochs+1):
        train(model,DEVICE,trainloader,optimizer,epoch,lossfuc)
    torch.save({"net":model.state_dict()},"./model_train{}".format(epochs))


    # model = net().to(DEVICE)
    # obj = torch.load("./model_0603-18")
    # model.load_state_dict(obj['net'])
    # datas = dataset.traindataset("./data/zhengqi_test.txt",train=False)
    # testloader = torch.utils.data.DataLoader(datas,batch_size=batch_size_test,shuffle=False)
    # outputs = test(model,DEVICE,testloader)
    # with open("./data/result.txt","w") as file:
    #     for output in outputs:
    #         file.write(str(output.item())+"\n")


    # traindatas,terget = datas[0] 
    # NET = net()
    # print(NET(traindatas))
    # data = iter(trainloader)
    # print(next(data))