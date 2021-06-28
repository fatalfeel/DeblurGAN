import torch
import torch.nn as nn
import torchvision
from torch import autograd
from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################
'''class ContentLoss: #for pix2pix using
    def __init__(self):
        self.criterion 	= nn.L1Loss()

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)'''

class PerceptualLoss():
    def __init__(self, opt):
        device              = torch.device("cuda" if opt.cuda else "cpu")
        self.vgg_features   = self.getFeatures().to(device)
        self.mse_loss       = nn.MSELoss()

    #get vgg19 layer 0~16, end = Conv2d(256, 256, kernel_size=(3), stride=(1), padding=(1))
    def getFeatures(self):
        end_layer = 16
        vgg_net   = torchvision.models.vgg19(pretrained=True).features

        '''sequence = nn.Sequential()
        for i, layer in enumerate(list(vgg_net)):
            sequence.add_module(str(i), layer)
            if i == end_layer:
                break'''

        children = list(vgg_net.children())
        features = children[:end_layer+1]
        features = nn.Sequential(*features)

        return features

    def get_loss(self, fake_B, real_B):
        #f_fake = self.contentFunc.forward(fakeIm)
        f_fake  = self.vgg_features.forward(fake_B)

        #f_real = self.contentFunc.forward(realIm)
        f_real  = self.vgg_features.forward(real_B)
        #f_real_no_grad = f_real.detach()

        #loss 	= self.mse_loss(f_fake, f_real_no_grad)
        loss    = self.mse_loss(f_fake, f_real.detach())

        return loss

'''class GANLoss(nn.Module):
    def __init__(self,
                 enable_bceloss=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label 	= target_real_label
        self.fake_label 	= target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        
        if enable_bceloss:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.L1Loss()

    def get_target_tensor(self, input, istarget_real):
        target_tensor = None
    
        if istarget_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
    
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
    
            target_tensor = self.fake_label_var
    
        return target_tensor
    
    def __call__(self, input, istarget_real):
        target_tensor = self.get_target_tensor(input, istarget_real)
    
        return self.loss(input, target_tensor)'''

class DiscriminatorLoss:
    def __init__(self, opt, tensor):
        #self.loss_gan 		= GANLoss(enable_bceloss=True, tensor=tensor) #GANLoss use nn.BCELoss()
        self.real_label 	= 1.0
        self.fake_label 	= 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor 		= tensor
        self.loss_bce		= nn.BCELoss()

    def get_target_label(self, input, istarget_real):
        target_label = None
        if istarget_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)

            target_label = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)

            target_label = self.fake_label_var

        return target_label

    def	CriterionLoss(self, input, istarget_real):
        target_label 	= self.get_target_label(input, istarget_real)
        loss 			= self.loss_bce(input, target_label)

        return loss

    def get_lossG(self, net, fakeB):
        # First, G(A) should fake the discriminator
        fake_feature    = net.forward(fakeB)

        #return self.loss_gan(fake_feature, 1)
        loss            = self.CriterionLoss(fake_feature, True) #special

        return loss

    #def get_loss(self, net, realA, fakeB, realB):
    def get_lossD(self, net, fakeB, realB):
        # Fake
        fake_feature 	= net.forward(fakeB.detach())
        #self.loss_D_fake 	= self.loss_gan(self.pred_fake, target_is_real=0)
        loss_D_fake 	= self.CriterionLoss(fake_feature, False)

        # Real
        real_feature	= net.forward(realB)
        #self.loss_D_real 	= self.loss_gan(self.pred_real, 1)
        loss_D_real 	= self.CriterionLoss(real_feature, True)

        # Combined loss
        loss 			= (loss_D_fake + loss_D_real) * 0.5

        return loss

'''class DiscLossLS(DiscriminatorLoss):
    def __init__(self, opt, tensor):
        super(DiscriminatorLoss, self).__init__(opt, tensor)
        #self.loss_gan = GANLoss(use_l1=True, tensor=tensor)
        self.loss_gan = GANLoss(enable_bceloss=False, tensor=tensor)
    
    def get_lossG(self,net, fakeB):
        return DiscriminatorLoss.get_lossG(self,net, fakeB)
    
    def get_lossD(self, net, fakeB, realB):
        return DiscriminatorLoss.get_lossD(self, net, fakeB, realB)'''

class DiscLossWGANGP():
    def __init__(self, opt, tensor):
        self.LAMBDA = 10
        self.device	= torch.device("cuda" if opt.cuda else "cpu")

    def calc_gradient_penalty(self, net, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = net.forward(interpolates)

        gradient_list = autograd.grad(outputs=disc_interpolates,
                                      inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)

        gradients           = gradient_list[0]
        gradient_penalty    = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA

        return gradient_penalty

    #def get_g_loss(self, net, realA, fakeB):
    def get_lossG(self, net, fakeB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)

        return -self.D_fake.mean()

    #def get_loss(self, net, realA, fakeB, realB):
    def get_lossD(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D         = self.D_fake - self.D_real
        gradient_penalty    = self.calc_gradient_penalty(net, realB.data, fakeB.data)

        return self.loss_D + gradient_penalty

def init_loss(opt, tensor):
    disc_loss = None

    if opt.gan_type == 'wgan-gp':
        disc_loss = DiscLossWGANGP(opt, tensor)
    else:
        disc_loss = DiscriminatorLoss(opt, tensor)

    #if opt.model == 'content_gan':
        #content_loss = PerceptualLoss(nn.MSELoss())
    content_loss = PerceptualLoss(opt)

    '''elif opt.model == 'pix2pix':
        content_loss = ContentLoss()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)'''

    return disc_loss, content_loss