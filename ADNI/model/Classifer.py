# coding=utf-8
# For feature map to classification result

import numpy as np
import torch.nn as nn
import torch


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, final_relu=1):
        super(BasicBlock, self).__init__()
        self.final_relu = final_relu
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
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


class RN18_Res_Classifer(nn.Module):

    def __init__(self, block=BasicBlock, layers=None, num_classes=3, in_planes=64,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, concat_num=1, feature=False):
        super(RN18_Res_Classifer, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.concat_num = concat_num

        self.inplanes = in_planes * concat_num
        self.dilation = 1
        self.feature = feature
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = []

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc2 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=256)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, res):

        res = self.relu(res)
        res = self.layer3(res)
        res = self.layer4(res)

        res = self.avgpool(res)
        res = torch.flatten(res, 1)

        if self.feature:
            return self.fc(res), res
        else:
            return self.fc(res)


class RN18_Classifer(nn.Module):
    def __init__(self, block=BasicBlock, layers=None, in_planes=64, num_classes=3, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout=0, dp=0.5):
        super(RN18_Classifer, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = in_planes
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
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if dropout == 0:
            self.fc_0 = nn.Linear(512 * block.expansion, num_classes)
        elif dropout == 1:
            self.fc_0 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.Dropout(dp),
                nn.Linear(512, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_final = self.fc_0(x)

        return x_final


class RN18_Classifer_future(nn.Module):
    # 用于由 gradN 预测 gradM 的任务，其中N<M
    def __init__(self, block=BasicBlock, layers=None, in_planes=64, num_classes=3, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, max_grad_diff=10,
                 norm_layer=None, dropout=0, dp=0.5):
        super(RN18_Classifer_future, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = in_planes
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
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 不同的头预测不同的未来时间，比如grad1=0，grad2=5，grad2-grad1=5，就用index为(5-1)//2=2的头预测。
        self.pre_heads = []
        if dropout == 0:
            for _ in range(max_grad_diff // 2):
                self.pre_heads.append(nn.Linear(512 * block.expansion, num_classes))
        elif dropout == 1:
            for _ in range(max_grad_diff // 2):
                self.pre_heads.append(
                    nn.Sequential(
                        nn.Linear(512 * block.expansion, 512),
                        nn.Dropout(dp),
                        nn.Linear(512, num_classes))
                )
        self.pre_heads = nn.ModuleList(self.pre_heads)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, feature, grad1, grad2):
        batch_size = feature.size(0)
        for i in range(batch_size):
            fc_index = int(torch.div(torch.abs(grad2[i] - grad1[i]) - 1, 2, rounding_mode='floor'))
            x = self.relu(feature[i].unsqueeze(0))
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if i == 0:
                res = self.pre_heads[fc_index](x)
            else:
                res = torch.cat((res, self.pre_heads[fc_index](x)), dim=0)

        return res


if __name__ == '__main__':
    C_net = RN18_Classifer(block=BasicBlock, layers=[2, 2, 2, 2])
    C_net_future = RN18_Classifer_future(block=BasicBlock, layers=[2, 2, 2, 2])
    data = torch.rand([4, 64, 64, 64, 64])
    grad1 = torch.tensor([0, 1, 2, 1])
    grad2 = torch.tensor([2, 10, 8, 4])
    output = C_net_future(data, grad1, grad2)
    print(output.shape)
