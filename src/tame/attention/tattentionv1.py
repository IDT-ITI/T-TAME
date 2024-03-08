from typing import List

import torch
import torch.nn as nn

from . import generic_atten as ga


class TAttentionV1_2(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size]):
        super().__init__(ft_size)
        # noinspection PyTypeChecker
        self.mhas = nn.ModuleList(
            [nn.MultiheadAttention(ft[2], 12, batch_first=True) for ft in ft_size]
        )
        ln_dim = ft_size[0][2]
        self.lns = nn.ModuleList(
            [
                nn.LayerNorm(
                    ln_dim,
                    eps=1e-06,
                )
                for _ in ft_size
            ]
        )
        self.gelu = nn.GELU()

        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        self.reshape = reshape_transform
        fuse_channels = sum(ft[2] for ft in ft_size)
        # noinspection PyTypeChecker

        self.mha_fuser = nn.MultiheadAttention(fuse_channels, 12, batch_first=True)

        self.cnn_fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=1000,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        feature_maps = features
        # layer norm
        class_maps = [op(feature) for op, feature in zip(self.lns, feature_maps)]
        # Multihead Attention
        class_maps = [
            op(feature, feature, feature, need_weights=False)[0]
            for op, feature in zip(self.mhas, class_maps)
        ]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.gelu(class_map) for class_map in class_maps]
        # concat
        class_map = torch.cat(class_maps, 2)
        # Multihead Attention
        class_map = self.mha_fuser(class_map, class_map, class_map, need_weights=False)[
            0
        ]
        # Reshape
        class_map = self.reshape(class_map)
        # fuse into 1000 channels
        c = self.cnn_fuser(class_map)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c


class TAttentionV1_1(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size]):
        super().__init__(ft_size)
        # noinspection PyTypeChecker
        self.mhas = nn.ModuleList(
            [nn.MultiheadAttention(ft[2], 12, batch_first=True) for ft in ft_size]
        )
        ln_dim = ft_size[0][2]
        self.lns = nn.ModuleList(
            [
                nn.LayerNorm(
                    ln_dim,
                    eps=1e-06,
                )
                for _ in ft_size
            ]
        )
        self.gelu = nn.GELU()

        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        self.reshape = reshape_transform
        fuse_channels = sum(ft[2] for ft in ft_size)
        # noinspection PyTypeChecker

        self.mha_fuser = nn.MultiheadAttention(fuse_channels, 12, batch_first=True)

        self.cnn_fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=1000,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        feature_maps = features
        # Multihead Attention
        class_maps = [
            op(feature, feature, feature, need_weights=False)[0]
            for op, feature in zip(self.mhas, feature_maps)
        ]
        # layer norm
        class_maps = [op(feature) for op, feature in zip(self.lns, class_maps)]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.gelu(class_map) for class_map in class_maps]
        # concat
        class_map = torch.cat(class_maps, 2)
        # Multihead Attention
        class_map = self.mha_fuser(class_map, class_map, class_map, need_weights=False)[
            0
        ]
        # Reshape
        class_map = self.reshape(class_map)
        # fuse into 1000 channels
        c = self.cnn_fuser(class_map)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c


class TAttentionV1(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size]):
        super().__init__(ft_size)
        # noinspection PyTypeChecker
        self.mhas = nn.ModuleList(
            [nn.MultiheadAttention(ft[2], 12, batch_first=True) for ft in ft_size]
        )
        ln_dim = ft_size[0][2]
        self.lns = nn.ModuleList(
            [
                nn.LayerNorm(
                    ln_dim,
                    eps=1e-06,
                )
                for _ in ft_size
            ]
        )
        self.relu = nn.ReLU()

        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        self.reshape = reshape_transform
        fuse_channels = sum(ft[2] for ft in ft_size)
        # noinspection PyTypeChecker

        self.mha_fuser = nn.MultiheadAttention(fuse_channels, 12, batch_first=True)

        self.cnn_fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=1000,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        feature_maps = features
        # Multihead Attention
        class_maps = [
            op(feature, feature, feature, need_weights=False)[0]
            for op, feature in zip(self.mhas, feature_maps)
        ]
        # layer norm
        class_maps = [op(feature) for op, feature in zip(self.lns, class_maps)]
        # add (skip connection)
        class_maps = [
            class_map + feature_map
            for class_map, feature_map in zip(class_maps, feature_maps)
        ]
        # activation
        class_maps = [self.relu(class_map) for class_map in class_maps]
        # concat
        class_map = torch.cat(class_maps, 2)
        # Multihead Attention
        class_map = self.mha_fuser(class_map, class_map, class_map, need_weights=False)[
            0
        ]
        # Reshape
        class_map = self.reshape(class_map)
        # fuse into 1000 channels
        c = self.cnn_fuser(class_map)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c
