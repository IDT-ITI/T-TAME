from typing import List, Tuple

import torch
import torch.nn as nn


class AttentionMech(nn.Module):
    def __init__(self, ft_size: List[torch.Size], **kwargs):
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        raise NotImplementedError
