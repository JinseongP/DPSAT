#https://github.com/TheSunWillRise/DPNAS

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n_group = 4
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(n_group, C_out, affine=False)
        )

    def forward(self, x):
        return self.op(x)
class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(n_group,C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)
class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(n_group,C_in, affine=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(n_group,C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)
class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.GroupNorm(n_group,C_out, affine=False)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

def GN(plane):
    return nn.GroupNorm(8, plane, affine=False)
def relu():
    return nn.ReLU()
def elu():
    return nn.ELU()
def selu():
    return nn.SELU()
def tanh():
    return nn.Tanh()
def htanh():
    return nn.Hardtanh()
def sigmoid():
    return nn.Sigmoid()
def lrelu():
    return nn.LeakyReLU()

class PrivReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(PrivReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            GN(C_out),
            # relu(),
        )

    def forward(self, x):
        return self.op(x)
class PrivDilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
        super(PrivDilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            GN(C_out),
            Act,
        )

    def forward(self, x):
        return self.op(x)
class PrivSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False, groups=C_out),
            GN(C_out),
            Act,
        )

    def forward(self, x):
        x = self.op(x)
        return x
class PrivResSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivResSepConv, self).__init__()
        self.conv = PrivSepConv(C_in, C_out, kernel_size, stride, padding, Act)
        self.res = Identity() if stride == 1 \
                else PrivFactorizedReduce(C_in, C_out, Act)
        self.res = (self.res)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])
class PrivConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False,),
            GN(C_out),
            Act,
        )

    def forward(self, x):
        x = self.op(x)
        return x
class PrivResConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivResConv, self).__init__()
        self.conv = PrivConv(C_in, C_out, kernel_size, stride, padding, Act)
        self.res = Identity() if stride == 1 \
                else PrivFactorizedReduce(C_in, C_out, Act)
        self.res = (self.res)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])
class PrivFactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, Act):
        super(PrivFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = Act
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = GN(C_out)

    def forward(self, x):
        if x.size(2)%2!=0:
            x = F.pad(x, (1,0,1,0), "constant", 0)

        out1 = self.conv_1(x)
        out2 = self.conv_2(x[:, :, 1:, 1:])

        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        if self.relu is not None:
            out = self.relu(out)
        return out

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),

    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),

    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),

    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 1, 1, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 2, 1, affine=affine),

    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.GroupNorm(n_group,C,affine=False, )
    ),

    ######################## For DP ########################
    'priv_avg_pool_3x3': lambda C, stride, affine:
         nn.AvgPool2d(3, stride=stride, padding=1),
    'priv_max_pool_3x3': lambda C, stride, affine:
         nn.MaxPool2d(3, stride=stride, padding=1),

    'priv_skip_connect': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, None),
    'priv_skip_connect_relu': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, relu()),
    'priv_skip_connect_elu': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, elu()),
    'priv_skip_connect_tanh': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, tanh()),
    'priv_skip_connect_selu': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, selu()),
    'priv_skip_connect_sigmoid': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, sigmoid()),

    'priv_sep_conv_3x3_relu': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, relu()),
    'priv_sep_conv_3x3_elu': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, elu()),
    'priv_sep_conv_3x3_tanh': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, tanh()),
    'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, sigmoid()),
    'priv_sep_conv_3x3_selu': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, selu()),
    'priv_sep_conv_3x3_htanh': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, htanh()),
    'priv_sep_conv_3x3_linear': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, Identity()),

    'priv_resep_conv_3x3_relu': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, relu()),
    'priv_resep_conv_3x3_elu': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, elu()),
    'priv_resep_conv_3x3_tanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, tanh()),
    'priv_resep_conv_3x3_sigmoid': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, sigmoid()),
    'priv_resep_conv_3x3_selu': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, selu()),
    'priv_resep_conv_3x3_htanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, htanh()),
    'priv_resep_conv_3x3_linear': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, Identity()),

    'priv_sep_conv_5x5_relu': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, relu()),
    'priv_sep_conv_5x5_elu': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, elu()),
    'priv_sep_conv_5x5_tanh': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, tanh()),
    'priv_sep_conv_5x5_sigmoid': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, sigmoid()),
    'priv_sep_conv_5x5_selu': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, selu()),
    'priv_sep_conv_5x5_htanh': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, htanh()),

    'priv_resep_conv_5x5_relu': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, relu()),
    'priv_resep_conv_5x5_elu': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, elu()),
    'priv_resep_conv_5x5_tanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, tanh()),
    'priv_resep_conv_5x5_sigmoid': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, sigmoid()),
    'priv_resep_conv_5x5_selu': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, selu()),
    'priv_resep_conv_5x5_htanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, htanh()),
}

DP_PRIMITIVES = [
    'none',
    'priv_max_pool_3x3',
    'priv_avg_pool_3x3',
    'priv_skip_connect',

    'priv_sep_conv_3x3_relu',
    'priv_sep_conv_3x3_selu',
    'priv_sep_conv_3x3_tanh',
    'priv_sep_conv_3x3_linear',
    'priv_sep_conv_3x3_htanh',
    'priv_sep_conv_3x3_sigmoid',
]

class DPBlock(nn.Module):
    def __init__(self, arch, C_prev, C, step):
        super(DPBlock, self).__init__()
        self._arch = arch
        self._steps = step
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(C_prev, C, 1, bias=False),
            GN(C),
        )
        self._ops = nn.ModuleDict()
        for item in arch:
            op_idx, fn, tn = item
            op_name = DP_PRIMITIVES[op_idx]
            op = OPS[op_name](C, 1, False)
            self._ops[str(fn)+'_'+str(tn)] = op

    def forward(self, s1):
        s1 = self.preprocess1.forward(s1)
        states = {0: s1}
        for op, f, t in self._arch:
            edge = str(f) + '_' + str(t)
            if t in states.keys():
                states[t] = states[t] + self._ops[edge](states[f])
            else:
                states[t] = self._ops[edge](states[f])

        return torch.cat([states[i] for i in range(1, self._steps + 1)], dim=1)


class DPNASNet_CIFAR(nn.Module):
    def __init__(self,):
        super(DPNASNet_CIFAR, self).__init__()

        ### cifar
        arch = [(5, 0, 1), (1, 0, 2), (5, 1, 2), (6, 0, 3), (1, 1, 3),
                (0, 2, 3), (5, 0, 4), (0, 1, 4), (0, 2, 4), (3, 3, 4),
                (5, 0, 5), (5, 1, 5), (6, 2, 5), (1, 3, 5), (6, 4, 5)]

        print(arch)
        layers = 6
        multiplier = 5
        C = 16
        stem_multiplier = 3
        C_prev = C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            GN(C_curr)
        )

        self.cells = []
        for i in range(layers):
            if i in [1,3,5]:
                cell = nn.MaxPool2d(2, 2)
            else:
                if i in [2,4]:
                    C_curr *= 2
                cell = DPBlock(arch, C_prev, C_curr, multiplier)
                C_prev = multiplier * C_curr
            self.cells += [cell]
        self.cells = nn.Sequential(*self.cells)

        self.adapt = nn.Sequential(
            nn.Conv2d(C_prev, 128, 1, bias=False),
            GN(128),)
        _shape = 4
        self.cls = nn.Sequential(
            nn.Linear(128 *_shape*_shape, 128),
            nn.SELU(),
            nn.Linear(128, 10),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x
    
class DPNASNet_CIFAR100(nn.Module):
    def __init__(self,):
        super(DPNASNet_CIFAR100, self).__init__()

        ### cifar
        arch = [(5, 0, 1), (1, 0, 2), (5, 1, 2), (6, 0, 3), (1, 1, 3),
                (0, 2, 3), (5, 0, 4), (0, 1, 4), (0, 2, 4), (3, 3, 4),
                (5, 0, 5), (5, 1, 5), (6, 2, 5), (1, 3, 5), (6, 4, 5)]

        print(arch)
        layers = 6
        multiplier = 5
        C = 16
        stem_multiplier = 3
        C_prev = C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            GN(C_curr)
        )

        self.cells = []
        for i in range(layers):
            if i in [1,3,5]:
                cell = nn.MaxPool2d(2, 2)
            else:
                if i in [2,4]:
                    C_curr *= 2
                cell = DPBlock(arch, C_prev, C_curr, multiplier)
                C_prev = multiplier * C_curr
            self.cells += [cell]
        self.cells = nn.Sequential(*self.cells)

        self.adapt = nn.Sequential(
            nn.Conv2d(C_prev, 128, 1, bias=False),
            GN(128),)
        _shape = 4
        self.cls = nn.Sequential(
            nn.Linear(128 *_shape*_shape, 128),
            nn.SELU(),
            nn.Linear(128, 100),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x    

class DPNASNet_MNIST(nn.Module):
    def __init__(self,):
        super(DPNASNet_MNIST, self).__init__()

        ### mnist
        arch = [(5, 0, 1), (6, 0, 2), (3, 1, 2), (0, 0, 3), (5, 1, 3),
                (4, 2, 3), (0, 0, 4), (6, 1, 4), (0, 2, 4), (3, 3, 4),
                (1, 0, 5), (0, 1, 5), (1, 2, 5), (1, 3, 5), (4, 4, 5)]

        print(arch)
        layers = 6
        multiplier = 5
        C = 16
        stem_multiplier = 2
        C_prev = C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            GN(C_curr)
        )

        self.cells = []
        for i in range(layers):
            if i in [1,3,5]:
                cell = nn.MaxPool2d(2, 2)
            else:
                if i in [4,]:
                    C_curr *= 2
                cell = DPBlock(arch, C_prev, C_curr, multiplier)
                C_prev = multiplier * C_curr
            self.cells += [cell]
        self.cells = nn.Sequential(*self.cells)

        _c = 128
        self.adapt = nn.Sequential(
            nn.Conv2d(C_prev, _c, 1, bias=False),
            GN(_c),)
        _shape = 3
        self.cls = nn.Sequential(
            nn.Linear(_c *_shape*_shape, _c),
            nn.Tanh(),
            nn.Linear(_c, 10),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x

class DPNASNet_FMNIST(nn.Module):
    def __init__(self, ):
        super(DPNASNet_FMNIST, self).__init__()

        ### fmnist
        arch = [(3, 0, 1), (0, 0, 2), (5, 1, 2), (4, 0, 3), (5, 1, 3),
                (3, 2, 3), (0, 0, 4), (0, 1, 4), (6, 2, 4), (3, 3, 4),
                (5, 0, 5), (6, 1, 5), (1, 2, 5), (4, 3, 5), (1, 4, 5)]

        print(arch)
        layers = 6
        multiplier = 5
        C = 16
        stem_multiplier = 2
        C_prev = C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            GN(C_curr)
        )

        self.cells = []
        for i in range(layers):
            if i in [1,3,5]:
                cell = nn.MaxPool2d(2, 2)
            else:
                if i in [4,]:
                    C_curr *= 2
                cell = DPBlock(arch, C_prev, C_curr, multiplier)
                C_prev = multiplier * C_curr
            self.cells += [cell]
        self.cells = nn.Sequential(*self.cells)

        _c = 128
        self.adapt = nn.Sequential(
            nn.Conv2d(C_prev, _c, 1, bias=False),
            GN(_c),)
        _shape = 3
        self.cls = nn.Sequential(
            nn.Linear(_c *_shape*_shape, _c),
            nn.Tanh(),
            nn.Linear(_c, 10),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x


MODELS = {
    "cifar10": DPNASNet_CIFAR,
    "cifar100": DPNASNet_CIFAR100,
    "fmnist": DPNASNet_FMNIST,
    "mnist": DPNASNet_MNIST,
}




