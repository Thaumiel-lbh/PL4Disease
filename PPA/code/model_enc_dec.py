import torch
import math
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, final_relu=1):
        super(BasicBlock, self).__init__()
        self.final_relu = final_relu
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        if self.final_relu:
            out = self.relu(out)
        
        return out


class ResNet_front(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, final_tanh=1):
        super(ResNet_front, self).__init__()
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       final_relu=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, final_relu=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for ss in range(1, blocks):
            if ss == blocks - 1 and final_relu == 0:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, final_relu=final_relu))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x


class Generator(nn.Module):
    def __init__(self, feature_num=6, is_ESPCN=0, mid_channel=1024):
        super(Generator, self).__init__()
        self.conv_blocks_feature = nn.Sequential(
            # 128*64*64 - 5256*32*32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # 256*32*32 - 512*32*32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks1 = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100 + feature_num, out_channels=2048, kernel_size=4, stride=1, padding=0),
            # up4
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4 to btc*512*8*8
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            
            # 8*8 - 16*16
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            
            # 16*16 - 32*32
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            # 32*32 - 64*64
            nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # 64*64 - 64*64
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        self.is_ESPCN = is_ESPCN
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        if self.is_ESPCN == 1:
            self.ESPCN = ESPCN(mid_channel=mid_channel)
    
    def forward(self, z, feature):
        mid_feature_1 = self.conv_blocks1(z)
        mid_feature_2 = self.conv_blocks_feature(feature)
        x = torch.cat((mid_feature_1, mid_feature_2), 1)
        featuremap = self.conv_blocks2(x)
        
        if self.is_ESPCN == 1:
            featuremap = self.ESPCN(featuremap)
        
        return featuremap
