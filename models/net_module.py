import torch
import torch.nn as nn
import functools
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayer_Discriminator(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False,
                 gpu_ids=[],
                 use_parallel=True):
        super(NLayer_Discriminator, self).__init__()
        self.gpu_ids        = gpu_ids
        self.use_parallel   = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult_prev    = 1
        nf_mult         = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev,
                                   ndf * nf_mult,
                                   kernel_size=kw,
                                   stride=2,
                                   padding=padw,
                                   bias=use_bias),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev,
                               ndf * nf_mult,
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        #if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
        if len(self.gpu_ids) > 0 and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
###############################################################################
# resnet
###############################################################################
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        padAndConv = {  'reflect':  [nn.ReflectionPad2d(1),
                                     nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],

                        'replicate':[nn.ReplicationPad2d(1),
                                     nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],

                        'zero':     [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]}

        '''try:
            blocks = padAndConv[padding_type] \
                     + \
                     [norm_layer(dim), nn.ReLU(True)] \
                     + \
                     [nn.Dropout(0.5)] if use_dropout else [] \
                     + \
                     padAndConv[padding_type] \
                     + \
                     [norm_layer(dim)]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)'''

        if use_dropout:
            blocks = padAndConv[padding_type] + [norm_layer(dim), nn.ReLU(True)] + [nn.Dropout(0.5)]
        else:
            blocks = padAndConv[padding_type] + [norm_layer(dim)]

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

#ConvBlock + ResidualBlock + DeconvBlock
#https://github.com/rrmina/fast-neural-style-pytorch/blob/master/transformer.py
class Resnet_Generator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf            = 64,
                 norm_layer     = nn.InstanceNorm2d,
                 use_dropout    = False,
                 n_layers       = 2,
                 n_blocks       = 9,
                 gpu_ids        = [],
                 use_parallel   = True,
                 learn_residual = False,
                 padding_type   ='reflect'):
        assert (n_blocks >= 0)
        super(Resnet_Generator, self).__init__()
        self.input_nc       = input_nc
        self.output_nc      = output_nc
        self.ngf            = ngf
        self.gpu_ids        = gpu_ids
        self.use_parallel   = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        '''sequence += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(128),
                     nn.ReLU(True),

                     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(256),
                     nn.ReLU(True)]

        for i in range(n_blocks):
            sequence += [ResnetBlock(256,
                                     padding_type=padding_type,
                                     norm_layer=norm_layer,
                                     use_dropout=use_dropout,
                                     use_bias=use_bias)]

        sequence += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                     norm_layer(128),
                     nn.ReLU(True),

                     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                     norm_layer(64),
                     nn.ReLU(True)]'''

        mult_prev = 1
        mult_curr = 1
        for n in range(1, n_layers + 1):
            mult_prev = mult_curr
            mult_curr = 2 ** n
            sequence += [nn.Conv2d(ngf * mult_prev, ngf * mult_curr, kernel_size=3, stride=2, padding=1, bias=use_bias),
                         norm_layer(ngf * mult_curr),
                         nn.ReLU(True)]

        for i in range(n_blocks):
            sequence += [ResnetBlock(ngf * mult_curr,
                                     padding_type=padding_type,
                                     norm_layer=norm_layer,
                                     use_dropout=use_dropout,
                                     use_bias=use_bias)]

        for n in range(n_layers, 0, -1):
            mult_prev = mult_curr
            mult_curr = 2 ** (n - 1)
            sequence += [nn.ConvTranspose2d(ngf * mult_prev, ngf * mult_curr, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                         norm_layer(ngf * mult_curr),
                         nn.ReLU(True)]

        sequence += [nn.ReflectionPad2d(3),
                     nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
                     nn.Tanh()]

        self.model = nn.Sequential(*sequence) #it's a deconvnet model

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)

        if self.learn_residual:
            output = torch.clamp(input + output, min=-1, max=1)

        return output

'''class UnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 gpu_ids=[],
                 use_parallel=False,
                 learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids        = gpu_ids
        self.use_parallel   = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8,
                                                 ngf * 8,
                                                 unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)

        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)

        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv   = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel  = [dConv]
            uModel  = [uRelu, uConv, nn.Tanh()]
            model   = [dModel, submodule, uModel]
        elif innermost:
            uConv   = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel  = [dRelu, dConv]
            uModel  = [uRelu, uConv, uNorm]
            model   = [dModel, uModel]
        else:
            uConv   = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel  = [dRelu, dConv, dNorm]
            uModel  = [uRelu, uConv, uNorm]

            model   = [dModel, submodule,uModel]
            model  += [nn.Dropout(0.5)] if use_dropout else []

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)'''

###############################################################################
# Functions
###############################################################################
def define_D(input_nc,
             ndf,
             #which_model_netD,
             n_layers_D=3,
             norm='batch',
             use_sigmoid=False,
             gpu_ids=[],
             use_parallel=False):
    #netD        = None
    use_gpu     = len(gpu_ids) > 0
    norm_layer  = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    '''if which_model_netD == 'basic':
        netD = NLayer_Discriminator(input_nc,
                                   ndf,
                                   n_layers     = 3,
                                   norm_layer   = norm_layer,
                                   use_sigmoid  = use_sigmoid,
                                   gpu_ids      = gpu_ids,
                                   use_parallel = use_parallel)

    elif which_model_netD == 'n_layers':
        netD = NLayer_Discriminator(input_nc,
                                   ndf,
                                   n_layers     = n_layers_D,
                                   norm_layer   = norm_layer,
                                   use_sigmoid  = use_sigmoid,
                                   gpu_ids      = gpu_ids,
                                   use_parallel = use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)'''

    netD = NLayer_Discriminator(input_nc,
                                ndf,
                                n_layers=n_layers_D,
                                norm_layer=norm_layer,
                                use_sigmoid=use_sigmoid,
                                gpu_ids=gpu_ids,
                                use_parallel=use_parallel)

    if use_gpu:
        netD.cuda(gpu_ids[0])

    netD.apply(weights_init)

    return netD

def define_G(input_nc,
             output_nc,
             ngf,
             #which_model_netG,
             n_layers_G,
             n_blocks_G,
             norm='batch',
             use_dropout=False,
             gpu_ids=[],
             use_parallel=False,
             learn_residual=False):
    #netG        = None
    use_gpu     = len(gpu_ids) > 0
    norm_layer  = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    '''if which_model_netG == 'resnet_9blocks':
        netG = Resnet_Generator(input_nc,
                               output_nc,
                               ngf,
                               norm_layer=norm_layer,
                               use_dropout=use_dropout,
                               n_blocks=9,
                               gpu_ids=gpu_ids,
                               use_parallel=use_parallel,
                               learn_residual=learn_residual)
    elif which_model_netG == 'resnet_6blocks':
        netG = Resnet_Generator(input_nc,
                               output_nc,
                               ngf,
                               norm_layer=norm_layer,
                               use_dropout=use_dropout,
                               n_blocks=6,
                               gpu_ids=gpu_ids,
                               use_parallel=use_parallel,
                               learn_residual=learn_residual)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc,
                             output_nc,
                             7,
                             ngf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout,
                             gpu_ids=gpu_ids,
                             use_parallel=use_parallel,
                             learn_residual=learn_residual)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc,
                             output_nc,
                             8,
                             ngf,
                             norm_layer=norm_layer,
                             use_dropout=use_dropout,
                             gpu_ids=gpu_ids,
                             use_parallel=use_parallel,
                             learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)'''

    netG = Resnet_Generator(input_nc,
                            output_nc,
                            ngf,
                            norm_layer      = norm_layer,
                            use_dropout     = use_dropout,
                            n_layers        = n_layers_G,
                            n_blocks        = n_blocks_G,
                            gpu_ids         = gpu_ids,
                            use_parallel    = use_parallel,
                            learn_residual  = learn_residual)

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])

    netG.apply(weights_init)

    return netG

