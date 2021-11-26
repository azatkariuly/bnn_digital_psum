import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import BinarizeConv2d

__all__ = ['resnet18_binary', 'resnet20_binary']

def Binaryconv3x3(in_planes, out_planes, stride=1, nbits_acc=8, T=64, k=2, s=2):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=False, nbits_acc=nbits_acc, T=T, k=k, s=s)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, do_bntan=True,
                 nbits_acc=8, T=64, k=2, s=2):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride, nbits_acc=nbits_acc,
                                   T=T, k=k, s=s)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes, nbits_acc=nbits_acc,
                                   T=T, k=k, s=s)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, x):
        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, do_bntan=True,
                    nbits_acc=8, T=64, k=2, s=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=stride, bias=False,
                               nbits_acc=nbits_acc, T=T, k=k, s=s),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_acc=nbits_acc, T=T, k=k, s=s))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, nbits_acc=nbits_acc, T=T, k=k, s=s))
        layers.append(block(self.inplanes, planes, do_bntan=do_bntan,
                            nbits_acc=nbits_acc, T=T, k=k, s=s))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18, nbits_acc=8, T=64, k=2, s=2):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False) #first layer
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n,
                                       nbits_acc=nbits_acc, T=T, k=k, s=s)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2,
                                       nbits_acc=nbits_acc, T=T, k=k, s=s)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2, do_bntan=False,
                                       nbits_acc=nbits_acc, T=T, k=k, s=s)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = nn.Linear(64*self.inflate, num_classes) #last layer

        init_model(self)


def resnet18_binary(**kwargs):
    num_classes = 10
    depth = 18
    return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth,
                          T=kwargs['T'], nbits_acc=kwargs['nbits_acc'],
                          k=kwargs['k'], s=kwargs['s'])

def resnet20_binary(**kwargs):
    num_classes = 10
    depth = 20
    return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth,
                          T=kwargs['T'], nbits_acc=kwargs['nbits_acc'],
                          k=kwargs['k'], s=kwargs['s'])
