import torch
import torch.nn as nn
import torch.nn.functional as tnf

#########################################Resnet fpn#####################################
class ResnetBackbone(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(ResnetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes*self.expansion:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes*self.expansion))
    def forward(self, data):
        #residual = data
        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        '''if self.downsample is not None:
            residual = self.downsample(data)
        out += residual'''

        if self.downsample is None:
            out += data
        else:
            out += self.downsample(data)

        out = self.relu(out)

        return out

class FPN(nn.Module):
    def __init__(self, num_blocks, num_filters=256):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = nn.BatchNorm2d(64)
        self.relu1      = nn.ReLU(inplace=True)
        self.maxpool1   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1     = self._make_layer(64,  num_blocks[0], stride=1)
        self.layer2     = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3     = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4     = self._make_layer(512, num_blocks[3], stride=2)

        # Top-down layers, use latlayer5 instead
        #self.toplayer   = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        '''self.latlayer3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)'''
        self.latlayer5  = nn.Conv2d(2048,    num_filters,        kernel_size=1, bias=False)
        self.latlayer4  = nn.Conv2d(1024,    num_filters,        kernel_size=1, bias=False)
        self.latlayer3  = nn.Conv2d(512,     num_filters,        kernel_size=1, bias=False)
        self.latlayer2  = nn.Conv2d(256,     num_filters,        kernel_size=1, bias=False)
        self.latlayer1  = nn.Conv2d(64,      num_filters // 2,   kernel_size=1, bias=False)

        # Smooth layers
        #self.smooth3    = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        #self.smooth2    = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        #self.smooth1    = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        '''self.smooth3    = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(num_filters),
                                        nn.ReLU(inplace=True))

        self.smooth2    = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(num_filters),
                                        nn.ReLU(inplace=True))

        self.smooth1    = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(num_filters),
                                        nn.ReLU(inplace=True))'''

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []

        for val in strides:
            block  = ResnetBackbone(self.in_planes, planes, val)
            layers.append(block)
            self.in_planes = planes * ResnetBackbone.expansion

        return nn.Sequential(*layers)

    '''def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        #upsample = interpolate
        return tnf.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y #align_corners=True can up iou'''

    def forward(self, data):
        # Bottom-up
        c1      = self.conv1(data)
        c1      = self.bn1(c1)
        c1      = self.relu1(c1)
        pool1   = self.maxpool1(c1)

        c2 = self.layer1(pool1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        '''p5 = self.toplayer(c5) #use latlayer5 instead

        lateral3 = self.latlayer3(c4)
        lateral2 = self.latlayer2(c3)
        lateral1 = self.latlayer1(c2)

        p4 = self._upsample_add(p5, lateral3)
        p3 = self._upsample_add(p4, lateral2)
        p2 = self._upsample_add(p3, lateral1)

        # Smooth
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)'''

        p5          = self.latlayer5(c5)
        lateral4    = self.latlayer4(c4)
        lateral3    = self.latlayer3(c3)
        lateral2    = self.latlayer2(c2)
        lateral1    = self.latlayer1(c1)

        #If use self.smooth(=self.td), sometimes there will be broken pieces of picture
        '''p4 = self.smooth3(tnf.interpolate(p5, scale_factor=2, mode="nearest") + lateral4)
        p3 = self.smooth2(tnf.interpolate(p4, scale_factor=2, mode="nearest") + lateral3)
        p2 = self.smooth1(tnf.interpolate(p3, scale_factor=2, mode="nearest") + lateral2)'''

        p4 = tnf.interpolate(p5, scale_factor=2, mode="nearest") + lateral4
        p3 = tnf.interpolate(p4, scale_factor=2, mode="nearest") + lateral3
        p2 = tnf.interpolate(p3, scale_factor=2, mode="nearest") + lateral2

        return lateral1, p2, p3, p4, p5

#########################################Deblurred class#####################################
class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid,  kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, data):
        output = tnf.relu(self.block0(data),    inplace=True)
        output = tnf.relu(self.block1(output),  inplace=True)

        return output

class FPN_RESNET(nn.Module):
    def __init__(self, type='FPN152', output_ch=3, num_filters_fpn=256, num_filters=128):
        super().__init__()

        if type == 'FPN152':
            self.fpn = self.FPN152()
        elif type == 'FPN101':
            self.fpn = self.FPN101()
        else:
            self.fpn = self.FPN50()

        # The segmentation heads on top of the FPN
        self.head1  = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2  = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3  = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4  = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth2 = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
                                     nn.InstanceNorm2d(num_filters),
                                     nn.ReLU())

        self.smooth1 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
                                     nn.InstanceNorm2d(num_filters // 2),
                                     nn.ReLU())

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def FPN50(self):
        return FPN([3, 4, 6, 3])

    def FPN101(self):
        return FPN([3, 4, 23, 3])

    def FPN152(self):
        return FPN([3, 8, 36, 3])

    def forward(self, data):
        map1, map2, map3, map4, map5 = self.fpn(data)

        map5 = tnf.interpolate(self.head4(map5), scale_factor=8, mode="nearest")
        map4 = tnf.interpolate(self.head3(map4), scale_factor=4, mode="nearest")
        map3 = tnf.interpolate(self.head2(map3), scale_factor=2, mode="nearest")
        map2 = tnf.interpolate(self.head1(map2), scale_factor=1, mode="nearest")

        smoothed = self.smooth2(torch.cat([map5, map4, map3, map2], dim=1))
        smoothed = tnf.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth1(smoothed + map1)
        smoothed = tnf.interpolate(smoothed, scale_factor=2, mode="nearest")

        final       = self.final(smoothed)
        residual    = torch.tanh(final) + data
        output      = torch.clamp(residual, min=-1, max=1)

        return output