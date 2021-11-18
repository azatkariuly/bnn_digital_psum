import satmm_cuda
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch import Tensor
import math
import numpy as np
from functools import reduce

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

def satmm_cuda_temp(A, X, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    satmm_cuda_psum = satmm_psum.apply
    psum = satmm_cuda_psum(A.contiguous(),X.contiguous(), T)

    #print(psum.max(), psum.min())
    if step_size_psum is not None:
        psum, s = quantizeLSQ_psum(psum, step_size_psum, nbits_psum)
        return OA(torch.sum(psum, axis=-1), b=b)*s

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

    #if step_size_psum is not None:
    #    #print(step_size_psum, nbits_psum, psum.shape[1])
    #    #psum, s = psum //2 , 1
    #    psum, s = quantizeLSQ_psum(psum, step_size_psum, nbits_psum, psum.shape[1])
    #    return reduce(lambda x,y: (x+y).clip(min, max), psum).transpose(0,-2).squeeze()*(2**s)
    return reduce(lambda x,y: (x+y), psum).transpose(0,-2).squeeze().transpose(1,2)

def satconv2D(image, kernel, padding=0, stride=1, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    #B,Cin,H,W
    #Cout, Cin, H,W
    #B,Cout,H,W
    # Gather Shapes of Kernel + Image + Padding
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding[0]) // stride[0] + 1
    OW = (W - CW + 2 * padding[1]) // stride[0] + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    return satmm(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(), T=T, b=b, signed=signed, nbits_psum=nbits_psum, step_size_psum=step_size_psum).reshape(B,Cout,OH,OW)

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

def quantizeLSQ(v, s, p, numl, isActivation=False):
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
    else:
        if p==-1 or p==1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2**(p-1)
            Qp = 2**(p-1) - 1

    gradScaleFactor = 1.0 / math.sqrt(numl*Qp)
    s = grad_scale(s, gradScaleFactor)

    if p==1:
        vbar = round_pass((v/s).sign())
    else:
        vbar = round_pass((v/s).clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s

def quantizeLSQ_psum(v, s, p, numl, isActivation=False):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    gradScaleFactor = 1.0 / math.sqrt(numl*Qp)
    s = round_pass(grad_scale(s, gradScaleFactor))

    vbar = round_pass((v/(2**s)).clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s

class Conv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(Conv2dLSQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.T = kwargs['T']

        self.nbits = kwargs['nbits']
        self.nbits_SA = kwargs['nbits_SA']        #SA bit size
        self.nbits_psum = kwargs['nbits_psum']    #psum bit size

        self.step_size_w = Parameter(torch.Tensor(1))
        self.step_size_a = Parameter(torch.Tensor(1))

        #psum step sizes
        self.step_size_psum = Parameter(torch.ones(1)*3.0)

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.step_size_a.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))

            self.init_state.fill_(1)

        x_q, s_a = quantizeLSQ(x, self.step_size_a, self.nbits, x.shape[1], isActivation=True)
        w_q, s_w = quantizeLSQ(self.weight, self.step_size_w, self.nbits, self.weight.data.numel())

        #OA = F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)*s_a*s_w
        SA = satconv2D(x_q, w_q, self.padding, self.stride, T=self.T, b=self.nbits_SA, signed=True, nbits_psum=self.nbits_psum, step_size_psum=self.step_size_psum)*s_a*s_w

        return SA

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
