import itertools

import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------- operation dict -------------------------------------------------------- #
OPS_enc = {
    'avg_pool': lambda in_ch, out_ch, stride, spec_norm: Pool(in_ch, out_ch, mode='Avg'),
    'max_pool': lambda in_ch, out_ch, stride, spec_norm: Pool(in_ch, out_ch, mode='Max'),
    'conv_3x3': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 3, stride, 1, spec_norm),
    'conv_4x4': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 4, stride, 1, spec_norm),
    'conv_5x5': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 5, stride, 2, spec_norm),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, spec_norm: DilConv(in_ch, out_ch, 3, stride, 2, 2, spec_norm),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, spec_norm: DilConv(in_ch, out_ch, 5, stride, 4, 2, spec_norm)
}

OPS_res = {
    'conv_3x3': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 3, stride, 0, spec_norm),
    'conv_5x5': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 5, stride, 1, spec_norm),
    'conv_7x7': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 7, stride, 2, spec_norm),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, spec_norm: DilConv(in_ch, out_ch, 3, stride, 1, 2, spec_norm),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, spec_norm: DilConv(in_ch, out_ch, 5, stride, 3, 2, spec_norm),
}

OPS_dec = {
    'nearest': lambda in_ch, out_ch, stride, spec_norm: Up(in_ch, out_ch, mode='nearest'),
    'bilinear': lambda in_ch, out_ch, stride, spec_norm: Up(in_ch, out_ch, mode='bilinear'),
    'ConvT_3x3': lambda in_ch, out_ch, stride, spec_norm: Up(in_ch, out_ch, 3, mode='ConvT'),
}

OPS_skip = {
    'none': lambda in_ch, out_ch, stride, spec_norm: Zero(),
    'skip': lambda in_ch, out_ch, stride, spec_norm: Identity(),
    'conv_1x1': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 1, stride, 0, spec_norm),
    'conv_3x3': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 3, stride, 1, spec_norm),
    'conv_5x5': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 5, stride, 2, spec_norm),
    'conv_7x7': lambda in_ch, out_ch, stride, spec_norm: Conv(in_ch, out_ch, 7, stride, 3, spec_norm),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, spec_norm: DilConv(in_ch, out_ch, 3, stride, 2, 2, spec_norm),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, spec_norm: DilConv(in_ch, out_ch, 5, stride, 4, 2, spec_norm),
}

ops_enc = list(OPS_enc.keys()) # 8
ops_res = list(OPS_res.keys()) # 5
ops_dec = list(OPS_dec.keys()) # 3
ops_skip = list(OPS_skip.keys()) # 8

# -------------------------------------------------- Operation Modules ---------------------------------------------------------------- #
class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)
    
class Identity(nn.Module):
    def forward(self, x):
        return x
    
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, spec_norm):
        super(Conv, self).__init__()
        if spec_norm:
            self.op = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        else:
            self.op = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
            
    def forward(self, x):
        return self.op(x)
    
    
class DilConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, spec_norm):
        super(DilConv, self).__init__()
        if spec_norm:
            self.op = nn.utils.spectral_norm(
              nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        else:
            self.op = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.op(x)
    
    
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=None, mode=None):
        super(Up, self).__init__()
        self.up_mode = mode
        if self.up_mode == 'ConvT':
            self.convT = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1, output_padding=kernel_size // 2),
            )
        else:
            self.c = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
            )

    def forward(self, x):
        if self.up_mode == 'ConvT':
            return self.convT(x)
        else:
            if self.up_mode == 'bilinear':
                return self.c(F.interpolate(x, scale_factor=2, mode=self.up_mode, align_corners=False))
            else:
                return self.c(F.interpolate(x, scale_factor=2, mode=self.up_mode))


class Pool(nn.Module):
    def __init__(self, in_ch, out_ch, mode=None):
        super(Pool, self).__init__()
        self.pool_mode = mode
        self.c = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        )
        
        if self.pool_mode == 'Avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif self.pool_mode == 'Max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

    def forward(self, x):
        return self.c(self.pool(x))

# -------------------------------------------------------- Mixed Operation ----------------------------------------------------- #
class Mixed(nn.Module):
    def __init__(self, in_ch, out_ch, stride, spec_norm, op_dict):
        super(Mixed, self).__init__()
        self.ops = nn.ModuleList()
        for k, o in op_dict.items():
            op = o(in_ch, out_ch, stride, spec_norm)
            self.ops.append(op)
        
    def forward(self, x, ws=None):
        # discrete
        if ws == None:
            return self.ops[0](x)
        # supernetwork
        else:
            return sum(w * op(x) for w, op in zip(ws, self.ops))
        
    
    
    
    