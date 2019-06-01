import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.fc1 = nn.Linear(1350,10)

    def forward(self,x):
        print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        print(x.shape)
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        print(x.shape)
        x = x.view(x.size()[0],-1)
        print(x.shape)
        x = self.fc1(x)
        return x

net = Net()
for name,parameters in net.named_parameters():
    print(name,":",parameters.shape)

ip = torch.randn(1,1,32,32)
out = net(ip)
print("out.shape:",out.shape)
# net.zero_grad()
# out.backward(torch.ones(1,10))

target = torch.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(out,target)
print(loss.item())

optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
optimizer.zero_grad()
loss.backward()

optimizer.step()

out = net(ip)
loss = criterion(out,target)
print(loss.item())
