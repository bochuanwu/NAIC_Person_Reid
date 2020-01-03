# encoding: utf-8
"""
@author:  bochuanwu
@contact: 1670306646@qq.com
"""

import torch
from torch import nn
import copy
from .backbones.resnet import *
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
import torch.nn.init as init
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,with_ibn,gcb,stage_with_gcb):
        super(Baseline, self).__init__()
        in_planes = 2048
        num_conv_out_channels = 128
        global_conv_out_channels = 256
        num_stripes = 6  # number of sub-parts
        used_levels = [1, 1, 1, 1, 1, 1]
        print("num_stripes:{}".format(num_stripes))
        print("num_conv_out_channels:{},".format(num_conv_out_channels))

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet.from_name(model_name, last_stride,with_ibn, gcb, stage_with_gcb)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet.from_name(model_name, last_stride,with_ibn, gcb, stage_with_gcb)
        elif model_name == 'resnet50':
            self.base = ResNet.from_name(model_name, last_stride, with_ibn, gcb, stage_with_gcb)
        elif model_name == 'resnet101':
            self.base = ResNet.from_name(model_name, last_stride, with_ibn, gcb, stage_with_gcb)
        elif model_name == 'resnet152':
            self.base = ResNet.from_name(model_name, last_stride, with_ibn, gcb, stage_with_gcb)


        if pretrain_choice == 'imagenet':

            if with_ibn:
                state_dict = torch.load(model_path)['state_dict']
                state_dict.pop('module.fc.weight')
                state_dict.pop('module.fc.bias')
                new_state_dict = {}
                for k in state_dict:
                    new_k = '.'.join(k.split('.')[1:])  # remove module in name
                    if self.base.state_dict()[new_k].shape == state_dict[k].shape:
                        new_state_dict[new_k] = state_dict[k]
                state_dict = new_state_dict
                self.base.load_state_dict(state_dict, strict=False)
            else:
                self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.dropout_layer = nn.Dropout(p=0.2)
        self.num_classes = num_classes

        # ============================================================================== pyramid
        self.num_stripes = num_stripes
        self.used_levels = used_levels

        # ==============================================================================pyramid
        self.pyramid_conv_list0 = nn.ModuleList()
        self.pyramid_fc_list0 = nn.ModuleList()
        Baseline.register_basic_branch(self, num_conv_out_channels,
                                      in_planes,
                                      self.pyramid_conv_list0,
                                      self.pyramid_fc_list0)

        # ==============================================================================pyramid
        input_size = 1024
        self.pyramid_conv_list1 = nn.ModuleList()
        self.pyramid_fc_list1 = nn.ModuleList()
        Baseline.register_basic_branch(self, num_conv_out_channels,
                                      input_size,
                                      self.pyramid_conv_list1,
                                      self.pyramid_fc_list1)


    def forward(self, x):

        feat = self.base(x)

        assert feat.size(2) % self.num_stripes == 0
        # ============================================================================== pyramid
        feat_list = []
        logits_list = []

        Baseline.pyramid_forward(self, feat,
                                self.pyramid_conv_list0,
                                self.pyramid_fc_list0,
                                feat_list,
                                logits_list)

        if self.training:
            return logits_list, feat_list
        else:
            return feat_list


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            if 'classifier' in k:
                # print(i[0])
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])

    @staticmethod
    def register_basic_branch(self, num_conv_out_channels,
                              input_size,
                              pyramid_conv_list,
                              pyramid_fc_list):
        # the level indexes are defined from fine to coarse,
        # the branch will contain one more part than that of its previous level
        # the sliding step is set to 1
        self.num_in_each_level = [i for i in range(self.num_stripes, 0, -1)]
        self.num_levels = len(self.num_in_each_level)
        self.num_branches = sum(self.num_in_each_level)

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            pyramid_conv_list.append(nn.Sequential(
                nn.Conv2d(input_size, num_conv_out_channels, 1),
                nn.BatchNorm2d(num_conv_out_channels),
                nn.ReLU(inplace=True)))

        # ============================================================================== pyramid
        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            fc = nn.Linear(num_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            pyramid_fc_list.append(fc)

    @staticmethod
    def pyramid_forward(self, feat,
                        pyramid_conv_list,
                        pyramid_fc_list,
                        feat_list,
                        logits_list):

        basic_stripe_size = int(feat.size(2) / self.num_stripes)

        idx_levels = 0
        used_branches = 0
        for idx_branches in range(self.num_branches):

            if idx_branches >= sum(self.num_in_each_level[0:idx_levels + 1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            idx_in_each_level = idx_branches - sum(self.num_in_each_level[0:idx_levels])

            stripe_size_in_level = basic_stripe_size * (idx_levels + 1)

            st = idx_in_each_level * basic_stripe_size
            ed = st + stripe_size_in_level

            local_feat = F.avg_pool2d(feat[:, :, st: ed, :],
                                      (stripe_size_in_level, feat.size(-1))) + F.max_pool2d(feat[:, :, st: ed, :],
                                                                                            (stripe_size_in_level,
                                                                                             feat.size(-1)))

            local_feat = pyramid_conv_list[used_branches](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)

            local_logits = pyramid_fc_list[used_branches](self.dropout_layer(local_feat))
            logits_list.append(local_logits)

            used_branches += 1


