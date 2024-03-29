import os
import torch
import argparse
from util import util

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataroot',          type=str,       default='./data/combined',  help='path to images (should have subfolders train, val, test)')
        self.parser.add_argument('--batchSize',         type=int,       default=1,                  help='input batch size')
        self.parser.add_argument('--loadSizeX',         type=int,       default=640,                help='scale images to this size')
        self.parser.add_argument('--loadSizeY',         type=int,       default=360,                help='scale images to this size')
        self.parser.add_argument('--fineSize',          type=int,       default=256,                help='then crop to this size')
        self.parser.add_argument('--input_nc',          type=int,       default=3,                  help='# of input image channels')
        self.parser.add_argument('--output_nc',         type=int,       default=3,                  help='# of output image channels')
        self.parser.add_argument('--ngf',               type=int,       default=64,                 help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf',               type=int,       default=64,                 help='# of discrim filters in first conv layer')

        self.parser.add_argument('--which_model_netG',  type=str,       default='RESNET',       help='RESNET, FPN50, FPN101, FPN152')
        self.parser.add_argument('--learn_residual',    type=str2bool,  default=True,           help='if specified, model would learn only the residual to the input')
        self.parser.add_argument('--resume',            type=str2bool,  default=False,          help='continue training')
        self.parser.add_argument('--gan_type',          type=str,       default='gan',          help='gan is faster, wgan-gp is stable')
        self.parser.add_argument('--n_layers_D',        type=int,       default=3,              help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_G',        type=int,       default=3,              help='2 layers features 2^6~2^8, 3 layers features 2^6~2^9')
        self.parser.add_argument('--n_blocks_G',        type=int,       default=12,             help='ResnetBlocks at 6, 9, 12...')
        #self.parser.add_argument('--gpu_ids',           type=str,       default='0',            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--cuda',              type=str2bool,  default=False,          help='using gpu training')
        self.parser.add_argument('--dataset_mode',      type=str,       default='aligned',      help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model',             type=str,       default='content_gan',  help='chooses which model to use. content_gan, pix2pix, test')
        self.parser.add_argument('--which_direction',   type=str,       default='AtoB',         help='AtoB or BtoA')
        self.parser.add_argument('--nThreads',          type=int,       default=1,              help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir',   type=str,       default='./checkpoint', help='models are saved here')
        self.parser.add_argument('--norm',              type=str,       default='instance',     help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches',    action='store_true',                    help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize',   type=int,       default=256,            help='display window size')
        self.parser.add_argument('--display_id',        type=int,       default=-1,             help='window id of the web display')
        self.parser.add_argument('--display_port',      type=int,       default=8097,           help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout',        action='store_true',                    help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size',  type=int,       default=float("inf"),   help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop',    type=str,       default='crop',         help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip',           action='store_true',                    help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--display_freq',      type=int,       default=100,            help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq',        type=int,       default=20,             help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq',   type=int,       default=10,             help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--phase',             type=str,       default='train',        help='train, val, test, etc')
        self.parser.add_argument('--e_epoch',           type=int,       default=2000,           help='number repeat to train')
        self.parser.add_argument('--beta1',             type=float,     default=0.5,            help='momentum term of adam')
        self.parser.add_argument('--lr',                type=float,     default=0.00001,        help='initial learning rate for adam')
        self.parser.add_argument('--content_weight',    type=float,     default=100.0,          help='fast-neural-style content weight')
        self.parser.add_argument('--pool_size',         type=int,       default=50,             help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html',           action='store_true',                    help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        self.isTrain = True

    def GetOption(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain	#train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        #print('-------------- End ----------------')

        # save to the disk
        #expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        expr_dir = self.opt.checkpoints_dir
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt
