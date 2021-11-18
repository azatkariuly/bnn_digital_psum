import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from functools import reduce

import numpy as np

def satmm(A, X, b=8, signed=True):
    width=2**b # 256
    max = (width >> signed) - 1 #127 or 255
    min = max - width + 1
    N, M = A.shape[-2], A.shape[-1]
    _, K = X.shape
    return reduce(lambda x,y : (x+y).clip(min,max) ,torch.multiply(X.flatten(), A.reshape(-1,N,M,1).expand(-1,-1,-1,K).reshape(-1,N,M*K)).reshape(-1,N,M,K).transpose(-2,0)).transpose(0,-2).squeeze()

def satconv2D(image, kernel, padding=0, stride=1, b=8, signed=True):
    #B,Cin,H,W
    #Cout, Cin, H,W
    #B,Cout,H,W
    # Gather Shapes of Kernel + Image + Padding
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding[0]) // stride[0] + 1
    OW = (W - CW + 2 * padding[1]) // stride[0] + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    return satmm(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(), b=b,signed=signed).transpose(1, 2).reshape(B,Cout,OH,OW)

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(BinarizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.nbits_OA = kwargs['nbits_OA']


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        #print(out.max(), out.min())
        #out = out.int().float()

        #out = satconv2D(input, self.weight, self.padding, self.stride, b=self.nbits_OA, signed=True)

        out = OA(out.int(), b=self.nbits_OA).float() + out - out.int()

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def OA(x, b=4):
    mask = (1 << b) - 1

    Qn = -2**(b-1)
    Qp = 2**(b-1)-1

    upper = (x > Qp).float()
    lower = (x < Qn).float()
    middle = 1.0 - upper - lower

    out = x*middle
    out += (x*(upper+lower)).int()&mask

    return out

