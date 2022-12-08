"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019
https://github.com/huangleiBuaa/IterNorm-pytorch
"""

import torch
import torch.nn as nn

class iterative_normalization_py(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        
        # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)

            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(
                beta=eps,
                input=P[0],
                alpha=1.0 / m,
                batch1=xc,
                batch2=xc.transpose(1, 2)
            )
    
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(
                    beta=1.5,
                    input=P[k],
                    alpha=-0.5,
                    batch1=torch.matrix_power(P[k], 3),
                    batch2=Sigma_N
                )
            saved.extend(P)

            # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
            wm = P[ctx.T].mul_(rTr.sqrt())

            running_mean.copy_(momentum * mean + (1.0 - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1.0 - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(beta=1.5, alpha=-0.5, batch1=g_tmp, batch2=P2)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P2, batch2=g_tmp)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P[k - 1].matmul(g_tmp), batch2=P[k - 1])
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0 * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):

    def __init__(
        self,
        nb_features,
        nb_groups=64,
        nb_channels=None,
        T=5,
        dim=2,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    ):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm does not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.nb_features = nb_features
        self.affine = affine
        self.dim = dim
        if nb_channels is None:
            nb_channels = (nb_features - 1) // nb_groups + 1
        nb_groups = nb_features // nb_channels
        while nb_features % nb_channels != 0:
            nb_channels //= 2
            nb_groups = nb_features // nb_channels
        assert (
            nb_groups > 0 and nb_features % nb_groups == 0
        ), f'nb_features={nb_features}, nb_groups={nb_groups}'
        self.nb_groups = nb_groups
        self.nb_channels = nb_channels
        shape = [1] * dim
        shape[1] = self.nb_features
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*shape))
            self.bias = nn.Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer(
            'running_mean',
            torch.zeros(nb_groups, nb_channels, 1)
        )
        # running whiten matrix
        self.register_buffer(
            'running_wm',
            torch.eye(nb_channels).expand(nb_groups, nb_channels, nb_channels).clone()
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(
            X,
            self.running_mean,
            self.running_wm,
            self.nb_channels,
            self.T,
            self.eps,
            self.momentum,
            self.training
        )
        if self.affine:
            return X_hat * self.weight + self.bias

        return X_hat

    def extra_repr(self):
        return (
            f'{self.nb_features}, '
            f'nb_channels={self.nb_channels}, '
            f'T={self.T}, '
            f'eps={self.eps}, '
            f'momentum={self.momentum}, '
            f'affine={self.affine}'
        )