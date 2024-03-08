from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import generic_atten as ga


class AttentionTAME(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size], num_classes=1000):
        super().__init__(ft_size)
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(
            inp, size=(feat_height, feat_height), mode="bilinear", align_corners=False
        )
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
                for in_channels in in_channels_list
            ]
        )
        self.bn_channels = in_channels_list
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(channels) for channels in self.bn_channels]
        )
        self.relu = nn.ReLU()
        # for each extra layer we need 1000 more channels to input to the fuse
        # convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features
        # Now all feature map sets are of the same HxW
        # conv
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        # batch norm
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.relu(class_map) for class_map in class_maps]
        # upscale
        class_maps = [self.interpolate(feature) for feature in class_maps]
        # concat
        class_maps = torch.cat(class_maps, 1)
        # fuse into 1000 channels
        c = self.fuser(class_maps)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c


class AttentionTTAME(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size], num_classes=1000):
        super().__init__(ft_size)
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(
            inp, size=(feat_height, feat_height), mode="bilinear", align_corners=False
        )
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
                for in_channels in in_channels_list
            ]
        )
        self.bn_channels = in_channels_list
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(channels) for channels in self.bn_channels]
        )
        self.relu = nn.ReLU()
        # for each extra layer we need 1000 more channels to input to the fuse
        # convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features
        # Now all feature map sets are of the same HxW
        # conv
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        # batch norm
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.relu(class_map) for class_map in class_maps]
        # upscale
        class_maps = [self.interpolate(feature) for feature in class_maps]
        # concat
        class_maps = torch.cat(class_maps, 1)
        # fuse into 1000 channels
        c = self.fuser(class_maps)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c


class Attention3DTAME(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size], num_classes=1000):
        super(Attention3DTAME, self).__init__(ft_size)
        if len(ft_size[0]) == 4:
            feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
            self.interpolate = lambda inp: F.interpolate(
                inp,
                size=(feat_height, feat_height),
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.interpolate = lambda inp: F.interpolate(
                inp, size=ft_size[0][-3:], mode="trilinear", align_corners=False
            )
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
                for in_channels in in_channels_list
            ]
        )
        self.bn_channels = in_channels_list
        self.bns = nn.ModuleList(
            [nn.BatchNorm3d(channels) for channels in self.bn_channels]
        )
        self.relu = nn.ReLU()
        # for each extra layer we need 1000 more channels to input to the fuse
        # convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv3d(
            in_channels=fuse_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features
        # Now all feature map sets are of the same HxW
        # conv
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        # batch norm
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.relu(class_map) for class_map in class_maps]
        # upscale
        class_maps = [self.interpolate(feature) for feature in class_maps]
        # concat
        class_maps = torch.cat(class_maps, 1)
        # fuse into 1000 channels
        c = self.fuser(class_maps)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c
