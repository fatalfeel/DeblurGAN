import os
import torch
import util.util as util
from torch.autograd import Variable
from collections import OrderedDict
from models.net_module import define_G, print_network

class TestModel():
    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__()

        self.s_epoch    = 1
        self.opt        = opt
        self.gpu_cuda   = opt.cuda
        self.isTrain    = opt.isTrain
        self.Tensor     = torch.cuda.FloatTensor if self.gpu_cuda else torch.Tensor
        self.save_dir   = opt.checkpoints_dir
        self.input_A    = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG       = define_G( opt.input_nc,
                                    opt.output_nc,
                                    opt.ngf,
                                    opt.which_model_netG,
                                    opt.n_layers_G,
                                    opt.n_blocks_G,
                                    opt.norm,
                                    not opt.no_dropout,
                                    self.gpu_cuda,
                                    False,
                                    opt.learn_residual)

        #which_epoch = opt.which_epoch
        #self.load_network(self.netG, 'G', which_epoch)
        self.load()

        print('---------- Networks initialized -------------')
        print_network(self.netG)
        print('-----------------------------------------------')

    def name(self):
        return 'TestModel'

    # helper loading function that can be used by subclasses
    def load(self):
        last_path       = os.path.join(self.save_dir, 'net_last.pth')
        checkpoint      = torch.load(last_path)
        self.netG.load_state_dict(checkpoint['state_dict_G'])
        self.s_epoch    = checkpoint['epoch']

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
