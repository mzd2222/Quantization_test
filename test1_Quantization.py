'''
pytorch 使用公式实现静态量化
可以量化各个位宽
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

    def forward(self, x):
        # b,3,32,32 => b,32,4,4
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

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv3 = QConvBNReLU(self.conv3, self.bn3, qi=False, qo=True, num_bits=num_bits)
        self.qmaxpool2d_3 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)

        self.qfc1 = QLinear(self.fc1, qi=False, qo=True, num_bits=num_bits)
        self.qfc2 = QLinear(self.fc2, qi=False, qo=True, num_bits=num_bits)
        self.qfc3 = QLinear(self.fc3, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qmaxpool2d_2(x)
        x = self.qconv3(x)
        x = self.qmaxpool2d_3(x)

        x = x.view(x.size(0), -1)

        x = self.qfc1(x)
        x = self.qfc2(x)
        x = self.qfc3(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qconv3.freeze(qi=self.qconv2.qo)
        self.qmaxpool2d_3.freeze(self.qconv3.qo)

        self.qfc1.freeze(qi=self.qconv3.qo)
        self.qfc2.freeze(qi=self.qfc1.qo)
        self.qfc3.freeze(qi=self.qfc2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = self.qconv3.quantize_inference(qx)
        qx = self.qmaxpool2d_3.quantize_inference(qx)

        qx = qx.view(qx.size(0), -1)

        qx = self.qfc1.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        qx = self.qfc3.quantize_inference(qx)

        out = self.qfc3.qo.dequantize_tensor(qx)

        return out


net = Lenet5()

try:
    net.load_state_dict(torch.load('./model/CNN.mdl'))
    print("load model success")
except:
    print("load model failed")

net.eval()
num_acc = 0
for idx, (x, y) in enumerate(test_loader):
    out = net(x)
    num_acc += torch.eq(torch.argmax(out, dim=1), y).sum().item()
print(num_acc / len(test_loader.dataset))

net.quantize(num_bits=8)

for idx, (x, y) in enumerate(test_loader):
    out = net.quantize_forward(x)

    if idx > 500:
        break

net.freeze()

num_acc_q = 0
for idx, (x, y) in enumerate(test_loader):
    out_q = net.quantize_inference(x)
    num_acc_q += torch.eq(torch.argmax(out_q, dim=1), y).sum().item()
print(num_acc_q / len(test_loader.dataset))
