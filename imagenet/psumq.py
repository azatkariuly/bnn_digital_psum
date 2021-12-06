import satmm_cuda
import torch
import torch.nn as nn

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


def quantizeLSQ_psum(v, s, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    #gradScaleFactor = 1.0 / math.sqrt(v.numel()*Qp)
    #s = round_pass(grad_scale(s, gradScaleFactor))

    vbar = round_pass((v/s).clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s


def satmm_cuda_temp(A, X, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    width=2**b # 256
    max = (width >> signed) - 1 #127 or 255
    min = max - width + 1

    satmm_cuda_psum = satmm_psum.apply
    psum = satmm_cuda_psum(A.contiguous(),X.contiguous(), T)

    if step_size_psum is not None:
        psum, s = quantizeLSQ_psum(psum, step_size_psum, nbits_psum)
        #out = reduce(lambda x,y: (x+y).clip(min, max), psum.transpose(0,3)).squeeze().transpose(0,-1)
        out = OA(torch.sum(psum, axis=3).squeeze().transpose(1,-1), b=b)
        #out = cyclic_activation(out, k=2, b=b)
        return out*step_size_psum

    #out = reduce(lambda x,y: (x+y).clip(min, max), psum.transpose(0,3)).squeeze().transpose(0,-1)
    out = OA(torch.sum(psum, axis=3).squeeze().transpose(1,-1), b=b)
    return out


def satconv2D(image, kernel, padding=0, stride=1, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding[0]) // stride[0] + 1
    OW = (W - CW + 2 * padding[1]) // stride[0] + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    return satmm_cuda_temp(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(), T=T, b=b, signed=signed,
                           nbits_psum=nbits_psum, step_size_psum=step_size_psum).reshape(B,Cout,OH,OW)


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
