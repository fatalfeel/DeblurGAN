# DeblurGAN
[arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)

Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.

Our network takes blurry image as an input and procude the corresponding sharp estimate, as in the example:
<img src="images/animation3.gif" width="400px"/> <img src="images/animation4.gif" width="400px"/>

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

### Prerequisites data
- ./install_data.sh
- bash -x ./install_data.sh

### Train
#open step 1 terminal window
- pip3 install visdom
- python3 -m visdom.server

#open step 2 terminal window
- cd /root/PycharmProjects/DeblurGAN
- python3 ./train.py --dataroot /root/PycharmProjects/DeblurGAN/data/combined --learn_residual True --resize_or_crop crop --fineSize 256

- If you do not want to use visdom.server then
- python3 ./train.py --dataroot /root/PycharmProjects/DeblurGAN/data/combined --learn_residual True --resize_or_crop crop --fineSize 256 --display_id -1

- If you want to use cpu to step debug and do not want to use visdom.server
- python3 ./train.py --dataroot /root/PycharmProjects/DeblurGAN/data/combined --learn_residual True --resize_or_crop crop --fineSize 256 --gpu_ids -1 --display_id -1
 
### Test
- python3 ./test.py --dataroot /root/PycharmProjects/DeblurGAN/data/combined --model test --dataset_mode single --learn_residual
