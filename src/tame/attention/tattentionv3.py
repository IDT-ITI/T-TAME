from functools import partial
from math import sqrt
from typing import List

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from . import generic_atten as ga


class TAttentionV3(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size]):
        super().__init__(ft_size)
        # noinspection PyTypeChecker
        self.lns_1 = nn.ModuleList(
            [
                nn.LayerNorm(
                    ft[2],
                    eps=1e-06,
                )
                for ft in ft_size
            ]
        )
        self.mhas = nn.ModuleList(
            [
                nn.MultiheadAttention(ft[2], int(ft[2] / 64), batch_first=True)
                for ft in ft_size
            ]
        )
        self.lns_2 = nn.ModuleList(
            [
                nn.LayerNorm(
                    ft[2],
                    eps=1e-06,
                )
                for ft in ft_size
            ]
        )

        self.mlps = nn.ModuleList(
            [
                torchvision.ops.MLP(ft[2], [4 * ft[2], ft[2]], activation_layer=nn.GELU)
                for ft in ft_size
            ]
        )

        # special initialization of MLP layers
        for mlp in self.mlps:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, std=1e-6)

        def reshape_transform(height, tensor):
            # check for class token and remove it if present
            if not sqrt(tensor.size(1)).is_integer():
                tensor = tensor[:, 1:, :]
            result = tensor.reshape(tensor.size(0), height, height, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        ft_heights = [
            int(sqrt(ft[1])) if sqrt(ft[1]).is_integer() else int(sqrt(ft[1] - 1))
            for ft in ft_size
        ]
        self.reshape_tsfms = [
            partial(
                reshape_transform,
                int(sqrt(ft[1])) if sqrt(ft[1]).is_integer() else int(sqrt(ft[1] - 1)),
            )
            for ft in ft_size
        ]
        fuse_channels = sum(ft[2] for ft in ft_size)

        self.cnn_fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=1000,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        if not ft_heights.count(ft_heights[0]) == len(ft_heights):
            feat_height = ft_heights[0] if ft_heights[0] <= 56 else 56
            self.interpolate = lambda inp: F.interpolate(
                inp,
                size=(feat_height, feat_height),
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.interpolate = lambda inp: inp

    def forward(self, features):
        feature_maps = features
        # layer norm 1
        xs = [op(feature) for op, feature in zip(self.lns_1, feature_maps)]
        # Multihead Attention
        xs = [op(x, x, x, need_weights=False)[0] for op, x in zip(self.mhas, xs)]
        # add (skip connection 1)
        xs = [x + feature_map for x, feature_map in zip(xs, feature_maps)]
        # layer norm 2
        ys = [op(x) for op, x in zip(self.lns_1, xs)]
        # MLP
        ys = [op(y) for op, y in zip(self.mlps, ys)]
        # add (skip connection 2)
        ys = [y + x for y, x in zip(ys, xs)]
        # Reshape
        ys = [reshape(y) for y, reshape in zip(ys, self.reshape_tsfms)]
        # upscale when used on cnns
        ys = [self.interpolate(y) for y in ys]
        # Concat
        ys = torch.cat(ys, 1)
        # fuse into 1000 channels
        c = self.cnn_fuser(ys)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c
