import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.image as image
import matplotlib.pyplot as plt
import cv2
import numpy as np

batch_size = 256
epochs = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=batch_size,shuffle=True)


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,3)
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)

    def forward(self,x):
        in_size = x.size(0)
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(in_size,-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model,device,train_loader, optimizer, epochs):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1) %30 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# def test(model,device,train_loader, optimizer, epochs):
#     model.train()
#     for batch_idx,(data,target) in enumerate(train_loader):
#         data, target = data.to(DEVICE), target.to(DEVICE)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         print("batch ", batch_idx , ": ", output, "loss: ", loss)
#         #loss.backward()
#         #optimizer.step()
#         #if(batch_idx+1) %30 ==0:
#         #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))


def test(model,device,test_loader):
   model.eval()
   test_loss = 0
   correct = 0
   with torch.no_grad():
       for data, target in test_loader:
           data, target = data.to(DEVICE) , target.to(DEVICE)
        #    print(target.shape)
        #    print(data.shape)
        #    print(type(data))
           output = model(data)
        #    print("output:  ",output)
        #    print("target:  ",target)
           test_loss += F.nll_loss(output, target, reduction='sum').item()
           pred = output.max(1,keepdim=True)[1]
           correct += pred.eq(target.view_as(pred)).sum().item()
   test_loss /= len(test_loader.dataset)
   print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))


def mydataset(imgpath):
    # img = cv2.imread("./three.png")  #用openCV读取图片
    img = cv2.imread(imgpath)
    # cv2.imshow("img",img)
    # cv2.waitKey()
    # print(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转化成灰度图像
    img = cv2.resize(img,(28,28))  #变换成28*28
    # print(img.shape)
    img = 255-img #图片反向
    # print(img.shape)
    # img.save(imgpath.split(".")[0]+".jpg")
    # print(imgpath.split("/")[1].split(".")[0]+".jpg")
    cv2.imwrite(imgpath.split("/")[1].split(".")[0]+".jpg",img)
    img = torch.from_numpy(img) #numpy转换成torch.Tensor
    img = img.float() #数据类型转化成float型
    img = torch.unsqueeze(img,0) #增加一个维度
    img = torch.unsqueeze(img,0) #增加一个维度
    img = torch.autograd.Variable(img,requires_grad=False)  #转换成Variable类型才可以输入到网络
    
    img = img.to(DEVICE) #判断是否可以使用gpu
    return img

if __name__ == "__main__":
    # for epoch in range(1,epochs+1):
    #     train(model,DEVICE,train_loader,optimizer,epoch)
    #     test(model,DEVICE,test_loader)


    # torch.save({'net': model.state_dict()},"./net")


    obj = torch.load("./net")
    model.load_state_dict(obj['net'])

    for i in range(8,12):
        img = mydataset("./zero{}.png".format(i))
    
        out = model(img) #输出
        print(torch.max(out,dim=1)[1]) #输出最后结果




