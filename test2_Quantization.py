'''
pytorch 自带静态量化实现
只能量化到int8
'''

import torch
import torchvision
from visdom import Visdom
from module import *
from torch import nn
import torch.utils.data as Data
from torch.nn import functional as F

test_loader = Data.DataLoader(
    torchvision.datasets.CIFAR10('./DataSet', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Resize([32, 32]),
                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                 ])),
    batch_size=64, shuffle=True)


# 创建网络
class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # b,3,32,32 => b,32,4,4
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.dequant(x)

        return x


net = Lenet5()

try:
    net.load_state_dict(torch.load('./model/CNN.mdl'))
    print("load model success")
except:
    print("load model failed")

torch.quantization.fuse_modules(net, [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3']])

net.qconfig = torch.quantization.get_default_qconfig('fbgemm')

net_prepared = torch.quantization.prepare(net)

net.eval()
for x, y in test_loader:
    net_prepared(x)

net_prepared_int8 = torch.quantization.convert(net_prepared)


net.eval()
num_acc = 0
for idx, (x, y) in enumerate(test_loader):
    out = net_prepared_int8(x)
    num_acc += torch.eq(torch.argmax(out, dim=1), y).sum().item()
print(num_acc / len(test_loader.dataset))