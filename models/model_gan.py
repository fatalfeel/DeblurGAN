import os
import torch
import util.util as util
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable
from models.net_module import define_G, define_D, print_network
from models.net_loss import init_loss

class ConditionalGAN():
    def __init__(self, opt):
        super(ConditionalGAN, self).__init__()
        
        self.s_epoch 	= 1
        self.opt         = opt
        self.gpu_cuda 	= opt.cuda
        self.isTrain 	= opt.isTrain
        self.Tensor 	= torch.cuda.FloatTensor if self.gpu_cuda else torch.Tensor
        self.save_dir 	= opt.checkpoints_dir
        
        # define tensors
        self.input_A         = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
        self.input_B         = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        
        self.real_A        	= None
        self.real_B         = None
        self.fake_A        	= None
        self.fake_B         = None
        
        self.loss_D         = None
        self.lossG        	= None
        self.lossC        	= None
        self.loss_GC        = None
        
        self.optimizer_D	= None
        self.optimizer_G 	= None
        
        # load/define networks
        # Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
        #use_parallel = not opt.gan_type == 'wgan-gp'
        self.criticUpdates 	= 1
        if opt.gan_type == 'wgan-gp':
            self.criticUpdates = 5
        
        #if self.isTrain:
        use_sigmoid = (opt.gan_type == 'gan')
        self.netD = define_D(opt.output_nc,
                        	 opt.ndf,
                        	 opt.n_layers_D,
                        	 opt.norm,
                        	 use_sigmoid,
                        	 self.gpu_cuda)
        
        self.netG = define_G(opt.input_nc,
                        	 opt.output_nc,
                        	 opt.ngf,
                        	 opt.which_model_netG,
                        	 opt.n_layers_G,
                        	 opt.n_blocks_G,
                        	 opt.norm,
                        	 not opt.no_dropout,
                        	 self.gpu_cuda,
                        	 opt.learn_residual)
        
        if self.isTrain:
            #self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
        
            # initialize optimizers
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
        
            # define loss functions
            self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)
        
        if opt.resume:
            self.load()

        print('---------- Networks netG -------------')
        print_network(self.netG)
        if self.isTrain:
            print('---------- Networks netD -------------')
            print_network(self.netD)
            print('---------- Networks contentLoss -------------')
            print_network(self.contentLoss.vgg_features)

    '''
    def NormalizeImg(self, img):
        nimg = (img - img.min()) / (img.max() - img.min())
        return nimg

    def show_MNIST(self, img):
        grid = torchvision.utils.make_grid(img)
        trimg = grid.numpy().transpose(1, 2, 0)
        plt.imshow(trimg)
        plt.title('Batch from dataloader')
        plt.axis('off')
        plt.show()
    '''

    def name(self):
        return 'ConditionalGANModel'

    def save(self, epoch):
        checkpoint = {'state_dict_D': 	self.netD.state_dict(),
                	  'state_dict_G':	self.netG.state_dict(),
                	  'optimizer_D': 	self.optimizer_D.state_dict(),
                	  'optimizer_G': 	self.optimizer_G.state_dict(),
                	  'epoch':         	epoch}
        save_path = os.path.join(self.save_dir, f'net_{epoch}.pth')
        torch.save(checkpoint, save_path)

        last_path = os.path.join(self.save_dir, 'net_last.pth')
        torch.save(checkpoint, last_path)

    def load(self):
        last_path 	= os.path.join(self.save_dir, 'net_last.pth')
        checkpoint	= torch.load(last_path)
        self.netD.load_state_dict(checkpoint['state_dict_D'])
        self.netG.load_state_dict(checkpoint['state_dict_G'])

        if self.optimizer_D is not None:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        if self.optimizer_G is not None:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

        self.s_epoch = checkpoint['epoch']

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputA = input['A' if AtoB else 'B'] #in Train is blurred
        inputB = input['B' if AtoB else 'A'] #in Train is sharp
        self.input_A.resize_(inputA.size()).copy_(inputA)
        self.input_B.resize_(inputB.size()).copy_(inputB)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A 	= Variable(self.input_A)         	#self.real_A is original blurred
        #In dcgan: fake_feture = netG(noise_images)
        #The noise_image here is self.real_A original blurred
        self.fake_B 	= self.netG.forward(self.real_A) 	#self.fake_B is fake blurred
        self.real_B 	= Variable(self.input_B)         	#self.real_B is original sharp
        '''nimga = self.NormalizeImg(self.real_A.detach().cpu())
        nimgb = self.NormalizeImg(self.fake_B.detach().cpu())
        nimgc = self.NormalizeImg(self.real_B.detach().cpu())
        self.show_MNIST(nimga)
        self.show_MNIST(nimgb)
        self.show_MNIST(nimgc)'''

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def train_update(self):
        self.forward()

        #for iter_d in range(self.criticUpdates): #self.criticUpdates = 1 for gan
        for i in range(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.loss_D = self.discLoss.get_lossD(self.netD, self.fake_B, self.real_B)
            self.loss_D.backward(retain_graph=True)
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.lossG      = self.discLoss.get_lossG(self.netD, self.fake_B)
        self.lossC      = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.content_weight
        self.loss_GC	= self.lossG + self.lossC
        self.loss_GC.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_Generator', self.lossG.item()),
                        	('G_Content', 	self.lossC.item()),
                        	('D_real+fake', self.loss_D.item())])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

    '''
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr 	= self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    '''
