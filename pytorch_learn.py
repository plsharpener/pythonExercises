import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    """网络部分"""

    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x:torch.tensor) -> torch.tensor:
        """forward"""
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x) -> torch.tensor:
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=s
        return num_features
net = Net()
print(net)

params = list(net.parameters())  #net.parameters()返回的是可被学习的参数列表和值
#print(params)
print(len(params))
print(params[0].size())

input = torch.randn(1,1,32,32)
out = net(input)
print(out)

# net.zero_grad()
# out.backward(torch.randn(1,10))


#计算损失函数

target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()
loss = criterion(out,target)
loss.backward()
optimzer = optim.SGD(net.parameters(),lr=0.01)
optimzer.zero_grad()
optimzer.step()
# print(loss)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

#反向传播
