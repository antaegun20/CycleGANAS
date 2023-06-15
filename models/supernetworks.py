import os
import functools

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from .search_space import *
    
###############################################################################
# CycleGAN Supernetworks
###############################################################################
class SuperResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(SuperResnetGenerator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
                
        model = [nn.ReflectionPad2d(1),
                 Mixed(input_nc, ngf, 1, False, OPS_res),
                 #nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()
                ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                Mixed(ngf * mult, ngf * mult * 2, 2, False, OPS_enc),
                #nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [SuperResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                Mixed(ngf * mult, int(ngf * mult / 2), 2, False, OPS_dec),
                #nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ]
            
        model += [nn.ReflectionPad2d(1)]
        model += [
            Mixed(ngf, output_nc, 1, False, OPS_res),
            #nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        ]
        model += [nn.Tanh()]
            
        self.model = nn.Sequential(*model)
        self.reset_arch()
        
    def set_tau(self, tau):
        self.tau = tau
        
    def set_gumbel(self, gumbel):
        self.gumbel = gumbel
        
    def arch_softmax(self):
        if self.gumbel:
            return [F.gumbel_softmax(F.softmax(w, dim=-1), tau=self.tau, hard=False, dim=-1) for w in self.arch_parameters]
        else:
            return [F.softmax(w, dim=-1) for w in self.arch_parameters]
        
    def reset_arch(self):
        self.arch_parameters = nn.ParameterList()
        for m in self.model:
            if isinstance(m, Mixed):
                self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(m.ops)))]
            elif isinstance(m, SuperResnetBlock):
                for mm in m.conv_block:
                    if isinstance(mm, Mixed):
                        self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(mm.ops)))]
        
    def forward(self, x):
        ws = self.arch_softmax()
        
        h = x
        count = 0
        for m in self.model:
            if isinstance(m, Mixed):
                h = m(h, ws[count])
                count += 1
            elif isinstance(m, SuperResnetBlock):
                h = m(h, ws[count:count + 2])
                count += 2
            else:
                h = m(h)
            
        return h
    
class SuperResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(SuperResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            Mixed(dim, dim, 1, False, OPS_res),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [
            Mixed(dim, dim, 1, False, OPS_res),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)
    
    def forward(self, x, ws):
        h = x
        count = 0
        for m in self.conv_block:
            if isinstance(m, Mixed):
                h = m(h, ws[count])
                count += 1
            else:
                h = m(h)
        out = x + h  # add skip connections
        return out


class SuperResnetGenerator_skip(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(SuperResnetGenerator_skip, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
                
        model = [nn.ReflectionPad2d(1),
                 Mixed(input_nc, ngf, 1, False, OPS_res),
                 #nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()
                ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                Mixed(ngf * mult, ngf * mult * 2, 2, False, OPS_enc),
                #nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [SuperResnetBlock_skip(
                ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_skips=i
            )]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                Mixed(ngf * mult, int(ngf * mult / 2), 2, False, OPS_dec),
                #nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ]
            
        model += [nn.ReflectionPad2d(1)]
        model += [
            Mixed(ngf, output_nc, 1, False, OPS_res),
            #nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        ]
        model += [nn.Tanh()]
            
        self.model = nn.Sequential(*model)
        self.reset_arch()
        print('Total {} Mixed Operations Created.'.format(len(self.arch_parameters)))
        
    def set_tau(self, tau):
        self.tau = tau
        
    def set_gumbel(self, gumbel):
        self.gumbel = gumbel
        
    def arch_softmax(self):
        if self.gumbel:
            return [F.gumbel_softmax(F.softmax(w, dim=-1), tau=self.tau, hard=False, dim=-1) for w in self.arch_parameters]
        else:
            return [F.softmax(w, dim=-1) for w in self.arch_parameters]
        
    def reset_arch(self):
        self.arch_parameters = nn.ParameterList()
        skip_count = 0
        for m in self.model:
            if isinstance(m, Mixed):
                self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(m.ops)))]
            elif isinstance(m, SuperResnetBlock_skip):
                # main ops
                for mm in m.conv_block:
                    if isinstance(mm, Mixed):
                        self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(mm.ops)))]
                # res op
                self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(m.res_op.ops)))]
                
                # skip_ops
                for sm in m.skip_ops:
                    self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(sm.ops)))]
                
    def forward(self, x):
        ws = self.arch_softmax()
        stack, skip_count = 0, 0
        skip_hs = []
        
        h = x
        for m in self.model:
            if isinstance(m, Mixed):
                h = m(h, ws[stack])
                stack += 1
            elif isinstance(m, SuperResnetBlock_skip):
                # forward 
                h, skip_h = m(h, ws[stack:stack + 3 + skip_count], skip_hs)
                skip_hs += [skip_h]
                
                # stack pointer
                stack += 3
                stack += skip_count
                skip_count += 1
            else:
                h = m(h)
            
        return h
    
class SuperResnetBlock_skip(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, num_skips):
        super(SuperResnetBlock_skip, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, num_skips)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, num_skips):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            Mixed(dim, dim, 1, False, OPS_res),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [
            Mixed(dim, dim, 1, False, OPS_res),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        
        self.res_op = Mixed(dim, dim, 1, False, OPS_skip)
        self.skip_ops = nn.ModuleList()
        for _ in range(num_skips):
            self.skip_ops.append(Mixed(dim, dim, 1, False, OPS_skip))
            
        return nn.Sequential(*conv_block)
    
    def forward(self, x, ws, skip_ins=[]):
        # ws = [1st Mixed, 2nd Mixed, res_op, skip_1, ...]
        assert len(ws) == len(skip_ins) + 3, "{} vs {}".format(len(ws), len(skip_ins) + 3)
        
        h = x
        count = 0 # 0 or 1
        for m in self.conv_block:
            # Mixed operations
            if isinstance(m, Mixed):
                # skip_out: output of 1st conv
                if count == 0:
                    h = m(h, ws[0])
                    skip_out = h
                # skip_in: added before 2nd conv
                else:
                    for i, (skip_op, skip_in) in enumerate(zip(self.skip_ops, skip_ins)):
                        h += skip_op(nn.ReflectionPad2d(1)(skip_in), ws[i + 3])
                        
                    h = m(h, ws[1])
                    
                count += 1
            # operations without weights (activation, norm, ...)
            else:
                h = m(h)
        out = h + self.res_op(x, ws[2])  # add residual operations
        return out, skip_out
    
class SuperUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SuperUnetGenerator, self).__init__()
        
        self.arch_parameters = nn.ParameterList()
        def get_arch(block):
            for m in block.model:
                if isinstance(m, Mixed):
                    self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(m.ops)))]
        
        # construct unet structure
        # add the innermost layer
        unet_block = SuperUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        get_arch(unet_block)
        
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = SuperUnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )
            get_arch(unet_block)
            
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = SuperUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        get_arch(unet_block)
        
        unet_block = SuperUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        get_arch(unet_block)
        
        unet_block = SuperUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        get_arch(unet_block)
        
        # add the outermost layer
        self.model = SuperUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        get_arch(self.model)
        
    def set_tau(self, tau):
        self.tau = tau
        
    def arch_softmax(self, gumbel=True):
        if gumbel:
            return [F.gumbel_softmax(F.softmax(w, dim=-1), tau=self.tau, hard=False, dim=-1) for w in self.arch_parameters]
        else:
            return [F.softmax(w, dim=-1) for w in self.arch_parameters]
        
    def reset_arch(self):
        new_params = []
        for p in self.arch_parameters:
            new_params += [nn.Parameter(1e-3 * torch.rand_like(p))]
            
        self.arch_parameters = nn.ParameterList(new_params)
        
    def forward(self, x):
        ws = self.arch_softmax()
        
        return self.model(x, ws)


class SuperUnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SuperUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = Mixed(input_nc, inner_nc, 2, False, OPS_enc)
        #nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = Mixed(inner_nc * 2, outer_nc, 2, False, OPS_dec)
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = Mixed(inner_nc, outer_nc, 2, False, OPS_dec)
            #upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = Mixed(inner_nc * 2, outer_nc, 2, False, OPS_dec)
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
                
    def forward(self, x, ws):
        h = x
        count = 0
        for m in self.model:
            # downconv -> submodule -> upconv
            if isinstance(m, Mixed):
                h = m(h, ws[count])
                count += 1
            # submodule
            elif isinstance(m, SuperUnetSkipConnectionBlock):
                h = m(h, ws[2:])
            else:
                h = m(h)
        
        if self.outermost:
            return h
        else:   # add skip connections
            return torch.cat([x, h], 1)


class SuperNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(SuperNLayerDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        sequence = [
            Mixed(input_nc, ndf, 2, True, OPS_enc),
            #nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            nn.LeakyReLU(0.2)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Mixed(ndf * nf_mult_prev, ndf * nf_mult, 2, True, OPS_enc),
                #nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Mixed(ndf * nf_mult_prev, ndf * nf_mult, 1, True, OPS_res),
            #nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]
        
        # output 1 channel prediction map
        sequence += [
            Mixed(ndf * nf_mult, 1, 1, True, OPS_res),
            #nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        self.model = nn.Sequential(*sequence)
        self.reset_arch()
        
    def set_tau(self, tau):
        self.tau = tau
        
    def set_gumbel(self, gumbel):
        self.gumbel = gumbel
        
    def arch_softmax(self):
        if self.gumbel:
            return [F.gumbel_softmax(F.softmax(w, dim=-1), tau=self.tau, hard=False, dim=-1) for w in self.arch_parameters]
        else:
            return [F.softmax(w, dim=-1) for w in self.arch_parameters]
    
    def reset_arch(self):
        self.arch_parameters = nn.ParameterList()
        
        # set arch parameters
        for m in self.model:
            if isinstance(m, Mixed):
                self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(len(m.ops)))]
                
    def forward(self, x):
        ws = self.arch_softmax()
        
        h = x
        count = 0
        for m in self.model:
            if isinstance(m, Mixed):
                h = m(h, ws[count])
                count += 1
            else:
                h = m(h)
        
        return h


class SuperPixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(SuperPixelDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            Mixed(input_nc, ndf, 1, True, OPS_res),
            #nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            Mixed(ndf, ndf * 2, 1, True, OPS_res),
            #nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2),
            Mixed(ndf * 2, 1, 1, True, OPS_res),
            #nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]
        
        self.net = nn.Sequential(*self.net)
                
    def set_tau(self, tau):
        self.tau = tau
        
    def arch_softmax(self, gumbel=True):
        if gumbel:
            return [F.gumbel_softmax(F.softmax(w, dim=-1), tau=self.tau, hard=False, dim=-1) for w in self.arch_parameters]
        else:
            return [F.softmax(w, dim=-1) for w in self.arch_parameters]
        
    def forward(self, input):
        """Standard forward."""
        return self.net(input)
    

class NetWeightGenerator(nn.Module):
    def __init__(self, net):
        super(NetWeightGenerator, self).__init__()
        
        self.lstm = nn.LSTM(64, 64, 1)
        self.heads = nn.ModuleList()
        for p in net.arch_parameters:
            self.heads.append(nn.Linear(64, len(p)))
        
        self.init_hc = (torch.randn(1, 1, 64), torch.randn(1, 1, 64))
        self.x = torch.randn(1, 1, 64)
        
    def forward(self):
        (h, c) = self.init_hc
        
        ws = []
        for head in self.heads:
            o, (h, c) = self.lstm(self.x, (h, c))
            ws.append(head(o.reshape(-1)))
            
        return ws
    
    
class HyperNetwork(nn.Module):
    def __init__(self, net):
        super(HyperNetwork, self).__init__()
        self.z = torch.randn(64)
        self.heads = []
        for p in net.arch_parameters:
            self.heads += [nn.Linear(64, len(p))]
        
    def forward(self):
        ws = []
        for head in self.heads:
            ws.append(head(self.z))
            
        return ws
    
class HyperNetwork_w(nn.Module):
    def __init__(self, net):
        super(HyperNetwork, self).__init__()
        self.z = torch.randn(64)
        self.heads = []
        for p in net.arch_parameters:
            self.heads += [nn.Linear(64, len(p))]
        
    def forward(self):
        ws = []
        for head in self.heads:
            ws.append(head(self.z))
            
        return ws

###############################################################################
# discrete networks
###############################################################################
class DiscreteResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, netG_arch, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(DiscreteResnetGenerator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        model = [nn.ReflectionPad2d(1),
                 Mixed(input_nc, ngf, 1, False, {ops_res[netG_arch[0]]: OPS_res[ops_res[netG_arch[0]]]}),
                 #nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()
                ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                Mixed(ngf * mult, ngf * mult * 2, 2, False, {ops_enc[netG_arch[i + 1]]: OPS_enc[ops_enc[netG_arch[i + 1]]]}),
                #nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [
                DiscreteResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    arch=netG_arch[3 + i * 2: 3 + (i + 1) * 2]
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                Mixed(ngf * mult, int(ngf * mult / 2), 2, False, {ops_dec[netG_arch[-i - 2]]: OPS_dec[ops_dec[netG_arch[-i - 2]]]}),
                #nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ]
            
        model += [nn.ReflectionPad2d(1)]
        model += [
            Mixed(ngf, output_nc, 1, False, {ops_res[netG_arch[-1]]: OPS_res[ops_res[netG_arch[-1]]]}),
            #nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        ]
        model += [nn.Tanh()]
            
        self.model = nn.Sequential(*model)
                
    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class DiscreteResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, arch):
        super(DiscreteResnetBlock, self).__init__()
        
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, arch)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, arch):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            Mixed(dim, dim, 1, False, {ops_res[arch[0]]: OPS_res[ops_res[arch[0]]]}),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [
            Mixed(dim, dim, 1, False, {ops_res[arch[1]]: OPS_res[ops_res[arch[1]]]}),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class DiscreteResnetGenerator_skip(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, netG_arch, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(DiscreteResnetGenerator_skip, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
                
        model = [nn.ReflectionPad2d(1),
                 Mixed(input_nc, ngf, 1, False, {ops_res[netG_arch[0]]: OPS_res[ops_res[netG_arch[0]]]}),
                 #nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()
                ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                Mixed(ngf * mult, ngf * mult * 2, 2, False, {ops_enc[netG_arch[i + 1]]: OPS_enc[ops_enc[netG_arch[i + 1]]]}),
                #nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        mult = 2 ** n_downsampling
        stack_pointer = 3
        for i in range(n_blocks):       # add ResNet blocks
            model += [DiscreteResnetBlock_skip(
                ngf * mult,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                use_bias=use_bias,
                num_skips=i,
                arch=netG_arch[stack_pointer:stack_pointer + i + 3],
            )]
            
            stack_pointer += (i + 3)

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                Mixed(ngf * mult, int(ngf * mult / 2), 2, False, {ops_dec[netG_arch[-i-2]]: OPS_dec[ops_dec[netG_arch[-i-2]]]}),
                #nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ]
            
        model += [nn.ReflectionPad2d(1)]
        model += [
            Mixed(ngf, output_nc, 1, False, {ops_res[netG_arch[-1]]: OPS_res[ops_res[netG_arch[-1]]]}),
            #nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        ]
        model += [nn.Tanh()]
            
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        skip_hs = []
        
        h = x
        for m in self.model:
            if isinstance(m, Mixed):
                h = m(h)
            elif isinstance(m, DiscreteResnetBlock_skip):
                # forward 
                h, skip_h = m(h, skip_hs)
                skip_hs += [skip_h]
            else:
                h = m(h)
            
        return h
    
class DiscreteResnetBlock_skip(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, num_skips, arch):
        super(DiscreteResnetBlock_skip, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, num_skips, arch)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, num_skips, arch):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            Mixed(dim, dim, 1, False, {ops_res[arch[0]]: OPS_res[ops_res[arch[0]]]}),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [
            Mixed(dim, dim, 1, False, {ops_res[arch[1]]: OPS_res[ops_res[arch[1]]]}),
            #nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        
        self.res_op = Mixed(dim, dim, 1, False, {ops_skip[arch[2]]: OPS_skip[ops_skip[arch[2]]]})
        self.skip_ops = nn.ModuleList()
        for ns in range(num_skips):
            self.skip_ops.append(Mixed(dim, dim, 1, False, {ops_skip[arch[3 + ns]]: OPS_skip[ops_skip[arch[3 + ns]]]}))
            
        return nn.Sequential(*conv_block)
    
    def forward(self, x, skip_ins=[]):
        h = x
        count = 0 # 0 or 1
        for m in self.conv_block:
            # Mixed operations
            if isinstance(m, Mixed):
                # skip_out: output of 1st conv
                if count == 0:
                    h = m(h)
                    skip_out = h
                # skip_in: added before 2nd conv
                else:
                    for i, (skip_op, skip_in) in enumerate(zip(self.skip_ops, skip_ins)):
                        h += skip_op(nn.ReflectionPad2d(1)(skip_in))
                        
                    h = m(h)
                    
                count += 1
            # operations without weights (activation, norm, ...)
            else:
                h = m(h)
        out = h + self.res_op(x)  # add residual operations
        return out, skip_out
    
class DiscreteUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, netG_arch, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DiscreteUnetGenerator, self).__init__()
        
        # construct unet structure
        # add the innermost layer
        unet_block = DiscreteUnetSkipConnectionBlock(
            ngf * 8, ngf * 8, netG_arch[0:2], input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = DiscreteUnetSkipConnectionBlock(
                ngf * 8, ngf * 8, netG_arch[(i + 1) * 2:(i + 2) * 2], input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )
            
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = DiscreteUnetSkipConnectionBlock(
            ngf * 4, ngf * 8, netG_arch[-8:-6], input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = DiscreteUnetSkipConnectionBlock(
            ngf * 2, ngf * 4, netG_arch[-6:-4], input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = DiscreteUnetSkipConnectionBlock(
            ngf, ngf * 2, netG_arch[-4:-2], input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        
        # add the outermost layer
        self.model = DiscreteUnetSkipConnectionBlock(
            output_nc, ngf, netG_arch[-2:], input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )
        
    def forward(self, input):
        return self.model(input)


class DiscreteUnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, arch, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DiscreteUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = Mixed(input_nc, inner_nc, 2, False, {ops_enc[arch[0]]: OPS_enc[ops_enc[arch[0]]]})
        #nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2,)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = Mixed(inner_nc * 2, outer_nc, 2, False, {ops_dec[arch[1]]: OPS_dec[ops_dec[arch[1]]]})
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = Mixed(inner_nc, outer_nc, 2, False, {ops_dec[arch[1]]: OPS_dec[ops_dec[arch[1]]]})
            #upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = Mixed(inner_nc * 2, outer_nc, 2, False, {ops_dec[arch[1]]: OPS_dec[ops_dec[arch[1]]]})
            #upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
                
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class DiscreteNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, netD_arch, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(DiscreteNLayerDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            Mixed(input_nc, ndf, 2, True, {ops_enc[netD_arch[0]]: OPS_enc[ops_enc[netD_arch[0]]]}),
            #nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            nn.LeakyReLU(0.2)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Mixed(ndf * nf_mult_prev, ndf * nf_mult, 2, True, {ops_enc[netD_arch[n]]: OPS_enc[ops_enc[netD_arch[n]]]}),
                #nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Mixed(ndf * nf_mult_prev, ndf * nf_mult, 1, True, {ops_res[netD_arch[-2]]: OPS_res[ops_res[netD_arch[-2]]]}),
            #nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]
        
        # output 1 channel prediction map
        sequence += [
            Mixed(ndf * nf_mult, 1, 1, True, {ops_res[netD_arch[-1]]: OPS_res[ops_res[netD_arch[-1]]]}),
            #nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        self.model = nn.Sequential(*sequence)
                
    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class DiscretePixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, netD_arch, norm_layer=nn.BatchNorm2d):
        super(DiscretePixelDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            Mixed(input_nc, ndf, 1, True, OPS_res),
            #nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            Mixed(ndf, ndf * 2, 1, True, OPS_res),
            #nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2),
            Mixed(ndf * 2, 1, 1, True, OPS_res),
            #nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]
        
        self.net = nn.Sequential(*self.net)
        
        for m in self.net:
            if isinstance(m, Mixed):
                self.arch_parameters += [m.ws]
                
    def forward(self, input):
        """Standard forward."""
        return self.net(input)
    
###############################################################################
# original networks
###############################################################################
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
