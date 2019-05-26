import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.image as image
import matplotlib.pyplot as plt
import cv2

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
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1) %30 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE) , target.to(DEVICE)
            print(target.shape)
            print(data.shape)
            print(type(data))
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    # for epoch in range(1,epochs+1):
    #     train(model,DEVICE,train_loader,optimizer,epoch)
    #     test(model,DEVICE,test_loader)

    # torch.save(model,"./net.pt")

    net2 = torch.load("./net.pt")

    img = image.imread("two.png")
    img = cv2.imread("two.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    # img = torch.unsqueeze(img,0)
    test1_loader = torch.utils.data.DataLoader(img,batch_size=1,shuffle=True)
    # print(img.shape)

    # net2.eval()
    # with torch.no_grad():
    #         data = img.to(DEVICE)
    #         print(data.shape)
    #         output = net2(data)
    #         print(output)

    # test(net2,DEVICE,test1_loader)
            
    test(net2,DEVICE,test_loader)
    # out  = net2(img)
    # print(out)




