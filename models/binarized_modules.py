import satmm_cuda
import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import reduce

import numpy as np

class satmm_psum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X, t):
        ctx.t = t
        out = satmm_cuda.forward_psum(A, X, t)
        ctx.save_for_backward(A, X)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.sum(axis=-1) / grad_output.shape[-1]
        A, X = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, X.T)
        grad_weight = torch.matmul(A.transpose(1,2), grad_output)
        return grad_input, grad_weight, None

'''
def satmm(A, X, b=8, signed=True):
    width=2**b # 256
    max = (width >> signed) - 1 #127 or 255
    min = max - width + 1
    N, M = A.shape[-2], A.shape[-1]
    _, K = X.shape
    return reduce(lambda x,y : (x+y).clip(min,max) ,torch.multiply(X.flatten(), A.reshape(-1,N,M,1).expand(-1,-1,-1,K).reshape(-1,N,M*K)).reshape(-1,N,M,K).transpose(-2,0)).transpose(0,-2).squeeze()
'''

def satmm(A, X, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    width=2**b # 256
    max = (width >> signed) - 1 #127 or 255
    min = max - width + 1
    N, M = A.shape[-2], A.shape[-1]
    _, K = X.shape

    mult = torch.multiply(X.flatten(), A.reshape(-1,N,M,1).expand(-1,-1,-1,K).reshape(-1,N,M*K)).reshape(-1,N,M,K).transpose(-2,0)

    rem = T - M%T
    psum_num = (M+rem)//T

    mult_reshaping = F.pad(input=mult, pad=(0, 0, 0, 0, 0, 0, 0, rem), mode='constant', value=0).reshape(T, psum_num, N, -1, K)

    psum = torch.sum(mult_reshaping, axis=0)
    # B N K T
    #print(torch.max(torch.abs(satmm_cuda.forward_psum(A.contiguous(),X.contiguous(),T).permute(3,1,0,2) - psum)))
    if step_size_psum is not None:
        psum, s = quantizeLSQ_psum(psum, step_size_psum, nbits_psum, psum.shape[1])
        return (OA(torch.sum(psum, axis=0), b=b)*s).transpose(1,2)

    #return reduce(lambda x,y: (x+y).clip(min, max), psum).transpose(0,-2).squeeze()
    return reduce(lambda x,y: OA((x+y), b=b), psum).transpose(0,-2).squeeze()

def satmm_cuda_temp(A, X, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    satmm_cuda_psum = satmm_psum.apply
    psum = satmm_cuda_psum(A.contiguous(),X.contiguous(), T)
    if step_size_psum is not None:
        psum, s = quantizeLSQ_psum(psum, step_size_psum, nbits_psum, psum.shape[1])
        return OA(torch.sum(psum, axis=-1), b=b)*s

def satconv2D(image, kernel, padding=0, stride=1, T=64, b=8, signed=True,
              nbits_psum=8, step_size_psum=None):
    #B,Cin,H,W
    #Cout, Cin, H,W
    #B,Cout,H,W
    # Gather Shapes of Kernel + Image + Padding
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding[0]) // stride[0] + 1
    OW = (W - CW + 2 * padding[1]) // stride[0] + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    return satmm_cuda_temp(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(), T=T, b=b, signed=signed,
                           nbits_psum=nbits_psum, step_size_psum=step_size_psum).reshape(B,Cout,OH,OW)

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ_psum(v, s, p, numl):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    gradScaleFactor = 1.0 / math.sqrt(numl*Qp)
    s = round_pass(grad_scale(s, gradScaleFactor))

    vbar = round_pass((v/s).clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s

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
        self.T = kwargs['T']
        self.nbits_psum = kwargs['nbits_psum']    #psum bit size

        #psum step sizes
        self.step_size_psum = Parameter(torch.ones(1)*4.0)

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        #out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        out = satconv2D(input, self.weight, self.padding, self.stride,
                        T=self.T, b=self.nbits_OA, signed=True,
                        nbits_psum=self.nbits_psum, step_size_psum=self.step_size_psum)

        #out = OA(out.int(), b=self.nbits_OA).float() + out - out.int()

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def OA(x, b=4):
    mask = (1 << b) - 1
    mask2 = 2**(b-1)

    Qn = -2**(b-1)
    Qp = 2**(b-1)-1

    upper = (x > Qp).float()
    lower = (x < Qn).float()
    middle = 1.0 - upper - lower

    out = x*middle

    out2 = (x*(upper+lower)).int()&mask

    upper2 = (out2 > Qp).float()
    lower2 = (out2 < Qn).float()
    middle2 = 1.0 - upper2 - lower2

    out3 = out2*middle2 + (out2-2*mask2)*upper2 + (out2+2*mask2)*lower2

    return out+out3
