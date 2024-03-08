# checked, should be working correctly
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from typing_extensions import Literal

from .attention import AMBuilder, Arrangement


class Generic(nn.Module):
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(
        self,
        name: str,
        mdl: nn.Module,
        feature_layers: Optional[List[str]],
        attention_version: str,
        masking: Literal["random", "diagonal", "max"] = "random",
        train_method: Literal[
            "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
        ] = "new",
        input_dim: Optional[torch.Size] = None,
        num_classes=1000,
    ):
        """Args:
        mdl (nn.Module): the model which we would like to use for interpretability
        feature_layers (list): the layers, as printed by get_graph_node_names,
            which we would like to get feature maps from
        """
        super().__init__()
        # get model feature extractor
        train_names, eval_names = get_graph_node_names(mdl)
        if feature_layers == [] or feature_layers is None:
            print(train_names)
            quit()

        output = (train_names[-1], eval_names[-1])
        if output[0] != output[1]:
            print("WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE")

        self.output = output[0]

        self.body = create_feature_extractor(
            mdl, return_nodes=(feature_layers + [self.output])
        )

        # Dry run to get number of channels for the attention mechanism
        if input_dim:
            inp = torch.randn(input_dim)
        else:
            inp = torch.randn(2, 3, 224, 224)
        self.body.eval()
        with torch.no_grad():
            out = self.body(inp)
        out.pop(self.output)

        # Required for attention mechanism initialization
        ft_size = [o.shape for o in out.values()]
        print(f"Dimensions of features: {ft_size}")
        # Build AM
        if num_classes != 1000:
            self.attn_mech = AMBuilder.create_attention(
                name, mdl, attention_version, ft_size, num_classes=num_classes
            )
        else:
            self.attn_mech = AMBuilder.create_attention(
                name, mdl, attention_version, ft_size
            )
        # Get loss and forward training method
        self.train_method: Literal[
            "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
        ] = train_method
        self.arrangement = Arrangement(self.train_method, self.body, self.output)
        self.train_policy, self.get_loss = (
            self.arrangement.train_policy,
            self.arrangement.loss,
        )

        self.a: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.masking: Literal["random", "diagonal", "max"] = masking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_method == "raw_normalize":
            x_norm = Generic.normalization(x)
        else:
            x_norm = x

        features: Dict[str, torch.Tensor] = self.body(x_norm)
        old_logits = features.pop(self.output)
        self.labels = old_logits.argmax(dim=1)

        # features now only has the feature maps since we popped the output in case we
        # are in eval mode

        # Attention mechanism
        a, c = self.attn_mech(features.values())
        self.a = a
        self.c = c
        # if in training mode we need to do another forward pass with our masked input
        # as input

        if self.training:
            logits = self.train_policy(a, self.labels, x)
            self.logits = logits
            return logits
        else:
            self.logits = old_logits
            return old_logits

    @staticmethod
    def select_max_masks(
        masks: torch.Tensor, logits: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Select the N masks with the max logits"""
        if logits.size(0) < N:
            max_indexes = logits.topk(logits.size(0))[1]
        else:
            max_indexes = logits.topk(N)[1]
        return masks[max_indexes, :, :]

    def get_c(self, labels: torch.Tensor) -> torch.Tensor:
        assert self.c is not None
        if self.masking == "random":
            return self.c[:, labels, :, :]
        elif self.masking == "diagonal":
            batches = self.c.size(0)
            return self.c[torch.arange(batches), labels, :, :].unsqueeze(1)

        elif self.masking == "max":
            batched_select_max_masks = torch.vmap(
                Generic.select_max_masks, in_dims=(0, 0, None)
            )
            return batched_select_max_masks(self.c, self.logits, self.logits.size(0))
        else:
            raise NotImplementedError

    def get_a(self) -> torch.Tensor:
        assert self.a is not None
        assert self.labels is not None
        labels = self.labels
        if self.masking == "random":
            return self.a[:, labels, :, :]
        elif self.masking == "diagonal":
            batches = self.a.size(0)
            return self.a[torch.arange(batches), labels, :, :].unsqueeze(1)
        elif self.masking == "max":
            batched_select_max_masks = torch.vmap(
                Generic.select_max_masks, in_dims=(0, 0, None)
            )
            return batched_select_max_masks(self.a, self.logits, self.logits.size(0))
        else:
            raise NotImplementedError
