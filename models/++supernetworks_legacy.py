import os
import functools

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from .search_space import *
    
###############################################################################
# Supernetworks
###############################################################################
class SuperResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, gumbel, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(SuperResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.use_gumbel = gumbel
        
        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf // 4, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf // 4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(),
        )
        self.cells = []
        self.cells += [Cell(ngf // 4, ngf, 'enc', [OPS_enc, OPS_enc])]
        
        # Resnet part
        for i in range(n_blocks):
            self.cells += [Cell(ngf, ngf, 'res', [OPS_res, OPS_res])]
        
        # Decoder part
        self.cells += [Cell(ngf, ngf // 4, 'dec', [OPS_dec, OPS_dec])]
        
        # to image
        self.dec2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf // 4, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        )
        self.cell_nets = nn.Sequential(*self.cells)
        self.initialize_alphas()
        
    def initialize_alphas(self):
        # cells
        num_enc_cells = 1
        num_res_cells = 9
        num_dec_cells = 1
        # edges in cell
        ## edge with size change
        num_edges = 2
        # operations -- from primitives
        num_enc_ops = len(ops_enc)
        num_res_ops = len(ops_res)
        num_dec_ops = len(ops_dec)
        
        # enc & res & dec
        self.enc_alphas = nn.Parameter(1e-3 * torch.randn(num_enc_cells, num_edges, num_enc_ops))
        self.res_alphas = nn.Parameter(1e-3 * torch.randn(num_res_cells, num_edges, num_res_ops))
        self.dec_alphas = nn.Parameter(1e-3 * torch.randn(num_dec_cells, num_edges, num_dec_ops))
        
        # arch parameters
        self.arch_parameters = [
            self.enc_alphas,
            self.res_alphas,
            self.dec_alphas,
        ]
        
    def set_tau(self, tau):
        self.tau = tau

    def softmax_weights(self):
        # preprocessing weights
        if self.use_gumbel:
            # encoder weights
            enc_alphas_pi = F.softmax(self.enc_alphas, dim=-1)
            enc_weights = F.gumbel_softmax(enc_alphas_pi, tau=self.tau, hard=False, dim=-1)
            
            # resnet
            res_alphas_pi = F.softmax(self.res_alphas, dim=-1)
            res_weights = F.gumbel_softmax(res_alphas_pi, tau=self.tau, hard=False, dim=-1)
            
            # decoder weights
            dec_alphas_pi = F.softmax(self.dec_alphas, dim=-1)
            dec_weights = F.gumbel_softmax(dec_alphas_pi, tau=self.tau, hard=False, dim=-1)
        else:
            enc_weights = F.softmax(self.enc_alphas, dim=-1)
            res_weights = F.softmax(self.res_alphas, dim=-1)
            dec_weights = F.softmax(self.dec_alphas, dim=-1)
            
        return [enc_weights[0]] + [w for w in res_weights] + [dec_weights[0]]
    
    def forward(self, x):
        ws = self.softmax_weights()
        h = x
        
        # handcrafted encoder part
        h = self.enc0(h)
        
        # forward cells
        for cell, w in zip(self.cell_nets, ws):
            h = cell(h, w) # skip_out is next skip_in
        
        # handcrafted decoder part
        output = self.dec2(h)
        
        return output

    
class SuperUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, gumbel, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SuperUnetGenerator, self).__init__()
        # construct unet structure
        ## add the innermost layer
        self.d1 = SuperUnetSkipConnectionBlock_D(input_nc, ngf * 1)
        self.d2 = SuperUnetSkipConnectionBlock_D(ngf * 1, ngf * 2)
        self.d3 = SuperUnetSkipConnectionBlock_D(ngf * 2, ngf* 4)
        self.d4 = SuperUnetSkipConnectionBlock_D(ngf * 4, ngf * 8)
        self.d5 = SuperUnetSkipConnectionBlock_D(ngf * 8, ngf * 8)
        self.d6 = SuperUnetSkipConnectionBlock_D(ngf * 8, ngf * 8)
        self.d7 = SuperUnetSkipConnectionBlock_D(ngf * 8, ngf * 8)
        self.d8 = SuperUnetSkipConnectionBlock_D(ngf * 8, ngf * 8, 1)
        
        self.u7 = SuperUnetSkipConnectionBlock_U(ngf * 8 * 2, ngf * 8)
        self.u6 = SuperUnetSkipConnectionBlock_U(ngf * 8 * 2, ngf * 8)
        self.u5 = SuperUnetSkipConnectionBlock_U(ngf * 8 * 2, ngf * 8)
        self.u4 = SuperUnetSkipConnectionBlock_U(ngf * 8 * 2, ngf * 8)
        self.u3 = SuperUnetSkipConnectionBlock_U(ngf * 8 + ngf * 4, ngf * 4)
        self.u2 = SuperUnetSkipConnectionBlock_U(ngf * 4 + ngf * 2, ngf * 2)
        self.u1 = SuperUnetSkipConnectionBlock_U(ngf * 2, output_nc)
        
        # arch_parameters
        num_enc_ops = len(ops_enc)
        num_res_ops = len(ops_res)
        num_dec_ops = len(ops_dec)
        
        self.arch_parameters = []
        for _ in range(8):
            self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(num_enc_ops))]
        for _ in range(7):
            self.arch_parameters += [nn.Parameter(1e-3 * torch.randn(num_dec_ops))]
    
    def load_arch_weights(self, device):
        for i in range(len(self.arch_parameters)):
            self.arch_parameters[i] = self.arch_parameters[i].to(device)
        
    def set_tau(self, tau):
        self.tau = tau
    
    def softmax_weights(self):
        # preprocessing weights
        sm_ws = []
        for w in self.arch_parameters:
            w_pi = F.softmax(w, dim=-1)
            sm_ws.append(F.gumbel_softmax(w_pi, tau=self.tau, hard=False, dim=-1))
                
        return sm_ws
    
    def forward(self, x):
        ws = self.softmax_weights()
        h0 = x
        
        h1 = self.d1(h0, ws[0])
        h2 = self.d2(h1, ws[1])
        h3 = self.d3(h2, ws[2])
        h4 = self.d4(h3, ws[3])
        h5 = self.d5(h4, ws[4])
        h6 = self.d6(h5, ws[5])
        h7 = self.d7(h6, ws[6])
        
        # innermost
        h8 = self.d8(h7, ws[7])
        h9 = self.u7(h8, h7, ws[8])
        
        h10 = self.u6(h9, h6, ws[9])
        h11 = self.u5(h10, h5, ws[10])
        h12 = self.u4(h11, h4, ws[11])
        h13 = self.u3(h12, h3, ws[12])
        h14 = self.u2(h13, h2, ws[13])
        h15 = self.u1(h14, w=ws[14])
        
        return h15


class SuperUnetSkipConnectionBlock_D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, gumbel=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SuperUnetSkipConnectionBlock_D, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        self.op = Mixed(in_ch, out_ch, stride, False, OPS_enc)
        self.act = nn.LeakyReLU(0.2)
        self.norm = norm_layer(out_ch)
        
    def forward(self, x, w):
        h = x
        h = self.act(h)
        h = self.op(h, w)
        h = self.norm(h)
        return h


class SuperUnetSkipConnectionBlock_U(nn.Module):
    def __init__(self, in_ch, out_ch, gumbel=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SuperUnetSkipConnectionBlock_U, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.op = Mixed(in_ch, out_ch, 2, False, OPS_dec)
        self.act = nn.ReLU()
        self.norm = norm_layer(out_ch)
    
    def forward(self, x, skip_x=None, w=None):
        if skip_x != None:
            h = torch.cat([x, skip_x], 1)
        else:
            h = x
            
        h = self.act(h)
        h = self.op(h, w)
        h = self.norm(h)
        return h
        
class SuperNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, gumbel, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(SuperNLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.use_gumbel = gumbel
        
        self.cells = []
        self.cells += [Cell(3, ndf // 2, 'enc', [OPS_enc, OPS_enc], act=nn.LeakyReLU(negative_slope=0.2))]
        self.cells += [Cell(ndf // 2, ndf * 2, 'enc', [OPS_enc, OPS_enc], act=nn.LeakyReLU(negative_slope=0.2))]
        self.enc3 = nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=1, padding=1)
        self.cell_nets = nn.Sequential(*self.cells)
        
        self.initialize_alphas()
        
    def initialize_alphas(self):
        num_enc_cells = 2
        num_edges = 2
        num_enc_ops = len(ops_enc)
        
        # arch parameters        
        self.arch_parameters = [
            nn.Parameter(1e-3 * torch.randn(num_enc_cells, num_edges, num_enc_ops))
        ]
        
    def load_arch_weights(self, device):
        for i in range(len(self.arch_parameters)):
            self.arch_parameters[i] = self.arch_parameters[i].to(device)
    
    def set_tau(self, tau):
        self.tau = tau

    def softmax_weights(self):
        # preprocessing weights
        enc_alphas_pi = F.softmax(self.arch_parameters[0], dim=-1)
        enc_weights = F.gumbel_softmax(enc_alphas_pi, tau=self.tau, hard=False, dim=-1)
            
        return enc_weights
    
    def forward(self, x):
        ws = self.softmax_weights()
        
        # input to hidden
        h = x
        
        for cell, w in zip(self.cell_nets, ws):
            h = cell(h, w)
        output = self.enc3(h)
        
        return output
    
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
