import torch
from torch import nn
from torch import optim
from torch.utils import data as Data
from torch.nn import functional as F
import torchvision

train_loader = Data.DataLoader(
    torchvision.datasets.CIFAR10('./DataSet', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Resize([32, 32]),
                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                 ])),
    batch_size=512, shuffle=True)

test_loader = Data.DataLoader(
    torchvision.datasets.CIFAR10('./DataSet', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Resize([32, 32]),
                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                 ])),
    batch_size=512, shuffle=True)


# class Lenet5(nn.Module):
#
#     def __init__(self):
#         super(Lenet5, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         # b,3,32,32 => b,32,4,4
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2, 2)
#
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2, 2)
#
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2, 2)
#
#         x = x.view(x.size(0), -1)
#
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#
#         return x


class Lenet5_2(nn.Module):

    def __init__(self):
        super(Lenet5_2, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # b,3,32,32 => b,32,4,4
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


net = Lenet5_2().cuda()

optimizer = optim.Adam(net.parameters(), lr=0.005)
loss_func = nn.CrossEntropyLoss().cuda()

for epoch in range(8):
    for idx, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()

        out = net(x)

        # print(out.size(), y.size())
        loss = loss_func(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    num_acc = 0
    for idx, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        out = net(x)
        num_acc += torch.eq(torch.argmax(out, dim=1), y).sum().item()
    print(num_acc / len(test_loader.dataset))

torch.save(net.state_dict(), './Model/test2.mdl')
