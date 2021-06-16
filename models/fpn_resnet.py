import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as tnf

class FPN(nn.Module):
    def __init__(self, type, num_filters_fpn):
        super(FPN, self).__init__()
        if type   == 'FPN50':
            resnet = torchvision.models.resnet50(pretrained=True)
        elif type == 'FPN101':
            resnet = torchvision.models.resnet101(pretrained=True)
        elif type == 'FPN152':
            resnet = torchvision.models.resnet152(pretrained=True)

        children        = list(resnet.children())
        self.conv1      = children[0]
        self.bn1        = children[1]
        self.relu1      = children[2]
        self.maxpool1   = children[3]

        self.layer1     = children[4]
        self.layer2     = children[5]
        self.layer3     = children[6]
        self.layer4     = children[7]

        # Lateral layers
        self.latlayer5  = nn.Conv2d(2048,    num_filters_fpn,        kernel_size=1, bias=False)
        self.latlayer4  = nn.Conv2d(1024,    num_filters_fpn,        kernel_size=1, bias=False)
        self.latlayer3  = nn.Conv2d(512,     num_filters_fpn,        kernel_size=1, bias=False)
        self.latlayer2  = nn.Conv2d(256,     num_filters_fpn,        kernel_size=1, bias=False)
        self.latlayer1  = nn.Conv2d(64,      num_filters_fpn // 2,   kernel_size=1, bias=False)

        self.smooth1    = nn.Sequential(nn.Conv2d(num_filters_fpn, num_filters_fpn, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(num_filters_fpn),
                                        nn.ReLU(inplace=True))

        self.smooth2    = nn.Sequential(nn.Conv2d(num_filters_fpn, num_filters_fpn, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(num_filters_fpn),
                                        nn.ReLU(inplace=True))

        self.smooth3    = nn.Sequential(nn.Conv2d(num_filters_fpn, num_filters_fpn, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(num_filters_fpn),
                                        nn.ReLU(inplace=True))

    def forward(self, input):
        # Bottom-up
        c1      = self.conv1(input)
        c1      = self.bn1(c1)
        c1      = self.relu1(c1)
        pool1   = self.maxpool1(c1)

        c2 = self.layer1(pool1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5          = self.latlayer5(c5)
        lateral4    = self.latlayer4(c4)
        lateral3    = self.latlayer3(c3)
        lateral2    = self.latlayer2(c2)
        lateral1    = self.latlayer1(c1)

        p4 = self.smooth1(tnf.interpolate(p5, scale_factor=2, mode="nearest") + lateral4)
        p3 = self.smooth2(tnf.interpolate(p4, scale_factor=2, mode="nearest") + lateral3)
        p2 = self.smooth3(tnf.interpolate(p3, scale_factor=2, mode="nearest") + lateral2)

        return lateral1, p2, p3, p4, p5

#########################################Deblurred class#####################################
class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, data):
        output = tnf.relu(self.block0(data),    inplace=True)
        output = tnf.relu(self.block1(output),  inplace=True)

        return output

class FPN_RESNET(nn.Module):
    def __init__(self, type='FPN101', output_ch=3, num_filters_fpn=256, num_filters=128):
        super().__init__()
        self.fpn    = FPN(type, num_filters_fpn)

        # The segmentation heads on top of the FPN
        self.head1  = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2  = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3  = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4  = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth1 = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
                                     nn.InstanceNorm2d(num_filters),
                                     nn.ReLU())

        self.smooth2 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
                                     nn.InstanceNorm2d(num_filters // 2),
                                     nn.ReLU())

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def forward(self, input):
        map1, map2, map3, map4, map5 = self.fpn(input)

        map5 = tnf.interpolate(self.head4(map5), scale_factor=8, mode="nearest")
        map4 = tnf.interpolate(self.head3(map4), scale_factor=4, mode="nearest")
        map3 = tnf.interpolate(self.head2(map3), scale_factor=2, mode="nearest")
        map2 = tnf.interpolate(self.head1(map2), scale_factor=1, mode="nearest")

        smoothed = self.smooth1(torch.cat([map5, map4, map3, map2], dim=1))
        smoothed = tnf.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map1)
        smoothed = tnf.interpolate(smoothed, scale_factor=2, mode="nearest")

        final       = self.final(smoothed)
        residual    = torch.tanh(final) + input
        output      = torch.clamp(residual, min=-1, max=1)

        return output