import torch.nn as nn
import torchvision.transforms as transforms
import math

from .lsq import Conv2dLSQ

__all__ = ['resnet20_lsq']

def conv3x3(in_planes, out_planes, stride=1, T=64, nbits=4, nbits_SA=8, nbits_psum=8):
    "3x3 convolution with padding"
    return Conv2dLSQ(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, T=T, nbits=nbits,
                     nbits_SA=nbits_SA, nbits_psum=nbits_psum)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, T=64, nbits=4, nbits_SA=8, nbits_psum=8):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, T=T, nbits=nbits, nbits_SA=nbits_SA, nbits_psum=nbits_psum)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, T=T, nbits=nbits, nbits_SA=nbits_SA, nbits_psum=nbits_psum)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, T=64, nbits=4, nbits_SA=8, nbits_psum=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dLSQ(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          T=T, nbits=nbits, nbits_SA=nbits_SA, nbits_psum=nbits_psum),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, T=T, nbits=nbits, nbits_SA=nbits_SA, nbits_psum=nbits_psum))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18, T=64, nbits=4, nbits_SA=8, nbits_psum=8):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False) #first layer
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n, T=T, nbits=nbits,
                                       nbits_SA=nbits_SA, nbits_psum=nbits_psum)
        self.layer2 = self._make_layer(block, 32, n, stride=2, T=T, nbits=nbits,
                                       nbits_SA=nbits_SA, nbits_psum=nbits_psum)
        self.layer3 = self._make_layer(block, 64, n, stride=2, T=T, nbits=nbits,
                                       nbits_SA=nbits_SA, nbits_psum=nbits_psum)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes) #last layer

        init_model(self)

def resnet20_lsq(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])

    num_classes = num_classes or 10
    depth = depth or 20 #56
    return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth,
                          T=kwargs['T'], nbits=kwargs['nbits'],
                          nbits_SA=kwargs['nbits_SA'], nbits_psum=kwargs['nbits_psum'])