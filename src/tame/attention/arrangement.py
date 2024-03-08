from torch import nn
import torch
from torch.nn import functional as F
from typing import List
from torchvision import transforms
from softadapt import LossWeightedSoftAdapt


def tv_loss(masks, var_coeff):
    var_loss = torch.tensor(0)
    for i in range(masks.ndim - 2):
        last_elem = masks.size(i + 2) - 1
        var_loss += torch.abs(
            torch.index_select(masks, i + 2, torch.arange(1, last_elem))
            - torch.index_select(masks, i + 2, torch.arange(0, last_elem - 1))
        ).mean()
    var_loss /= masks.ndim - 2
    mean_dist = torch.mean(torch.square(torch.full_like(masks, 0.5) - masks))
    return var_coeff * var_loss + mean_dist, var_loss, mean_dist


class Arrangement(nn.Module):
    r"""The train_policy and get_loss components of Generic"""

    def __init__(self, version: str, body: nn.Module, output_name: str):
        super(Arrangement, self).__init__()
        arrangements = {
            "adapt_all": (self.new_train_policy, self.ada_all_loss),
            "adapt": (self.new_train_policy, self.ada_loss),
            "new": (self.new_train_policy, self.classic_loss),
            "renormalize": (self.old_train_policy, self.classic_loss),
            "raw_normalize": (self.legacy_train_policy, self.classic_loss),
            "layernorm": (self.ln_train_policy, self.classic_loss),
            "batchnorm": (self.bn_train_policy, self.classic_loss),
        }

        if version == "layernorm":
            self.norm = nn.LayerNorm([3, 224, 224])
        elif version == "batchnorm":
            self.norm = nn.BatchNorm2d(3)
        elif version == "new":
            self.new_coeff = 0.1
        elif version == "adapt":
            self.new_coeff = 0.1
            self.softadapt_object = LossWeightedSoftAdapt(beta=0.1)
            self.steps_to_make_updates = 10000
            self.count = 0

            # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
            self.values_of_component_1 = []
            self.values_of_component_2 = []
            # Initializing adaptive weights to all ones.
            self.adapt_weights = torch.tensor([1, 1])
        elif version == "adapt_all":
            self.new_coeff = 0.1
            self.softadapt_object = LossWeightedSoftAdapt(beta=0.1)
            self.steps_to_make_updates = 10000
            self.count = 0

            # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
            self.values_of_component_1 = []
            self.values_of_component_2 = []
            self.values_of_component_3 = []
            # Initializing adaptive weights to all ones.
            self.adapt_weights = torch.tensor([1, 1, 1])

        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.body = body
        self.output_name = output_name

        self.ce_coeff = 1.5  # lambda3
        self.area_loss_coeff = 2  # lambda2
        self.smoothness_loss_coeff = 0.01  # lambda1
        self.area_loss_power = 0.3  # lambda4

        self.train_policy, self.loss = arrangements[version]

    def area_loss(self, masks):
        if self.area_loss_power != 1:
            # add e to prevent nan (derivative of sqrt at 0 is inf)
            masks = (masks + 0.0005) ** self.area_loss_power
        return torch.mean(masks)

    def ada_all_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        labels = labels.long()
        tvd_loss, var_loss, mean_dist_loss = tv_loss(masks, self.new_coeff)
        cross_entropy = self.loss_cross_entropy(logits, labels)
        self.values_of_component_1.append(cross_entropy)
        self.values_of_component_2.append(var_loss)
        self.values_of_component_3.append(mean_dist_loss)
        if self.count == self.steps_to_make_updates:
            self.adapt_weights = self.softadapt_object.get_component_weights(
                torch.tensor(self.values_of_component_1),  # type: ignore
                torch.tensor(self.values_of_component_2),  # type: ignore
                torch.tensor(self.values_of_component_3),  # type: ignore
                verbose=False,
            )

            # Resetting the lists to start fresh (this part is optional)
            self.values_of_component_1 = []
            self.values_of_component_2 = []
            self.values_of_component_3 = []

        loss = (
            self.adapt_weights[0] * cross_entropy
            + self.adapt_weights[1] * var_loss
            + self.adapt_weights[2] * mean_dist_loss
        )
        return [loss, cross_entropy, mean_dist_loss, var_loss]

    def ada_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        labels = labels.long()
        tvd_loss, var_loss, mean_dist_loss = tv_loss(masks, self.new_coeff)
        cross_entropy = self.loss_cross_entropy(logits, labels)
        self.values_of_component_1.append(cross_entropy)
        self.values_of_component_2.append(tvd_loss)
        if self.count == self.steps_to_make_updates:
            self.adapt_weights = self.softadapt_object.get_component_weights(
                torch.tensor(self.values_of_component_1),  # type: ignore
                torch.tensor(self.values_of_component_2),  # type: ignore
                verbose=False,
            )

            # Resetting the lists to start fresh (this part is optional)
            self.values_of_component_1 = []
            self.values_of_component_2 = []

        loss = self.adapt_weights[0] * cross_entropy + self.adapt_weights[1] * tvd_loss
        return [loss, cross_entropy, mean_dist_loss, tvd_loss]

    def very_new_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        labels = labels.long()
        tvd_loss, var_loss, mean_dist_loss = tv_loss(masks, self.new_coeff)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        loss = cross_entropy + tvd_loss
        return [loss, cross_entropy, mean_dist_loss, var_loss]

    @staticmethod
    def smoothness_loss(masks, power=2, border_penalty=0.3):
        if masks.dim() == 4:
            B, _, _, _ = masks.size()
            x_loss = torch.sum(
                (torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])) ** power
            )
            y_loss = torch.sum(
                (torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])) ** power
            )
            if border_penalty > 0:
                border = float(border_penalty) * torch.sum(
                    masks[:, :, -1, :] ** power
                    + masks[:, :, 0, :] ** power
                    + masks[:, :, :, -1] ** power
                    + masks[:, :, :, 0] ** power
                )
            else:
                border = 0.0
            return (x_loss + y_loss + border) / float(
                power * B
            )  # watch out, normalised by the batch size!
        else:
            B, _, _, _, _ = masks.size()
            x_loss = torch.sum(
                (torch.abs(masks[:, :, :, 1:, :] - masks[:, :, :, :-1, :])) ** power
            )
            y_loss = torch.sum(
                (torch.abs(masks[:, :, :, :, 1:] - masks[:, :, :, :, :-1])) ** power
            )
            z_loss = torch.sum(
                (torch.abs(masks[:, :, 1:, :, :] - masks[:, :, :-1, :, :])) ** power
            )

            if border_penalty > 0:
                border = float(border_penalty) * torch.sum(
                    (masks[:, :, :, -1, :] ** power).sum()
                    + (masks[:, :, :, 0, :] ** power).sum()
                    + (masks[:, :, :, :, -1] ** power).sum()
                    + (masks[:, :, :, :, 0] ** power).sum()
                    + (masks[:, :, -1, :, :] ** power).sum()
                    + (masks[:, :, 0, :, :] ** power).sum()
                )
            else:
                border = 0.0
            return (x_loss + y_loss + z_loss + border) / float(
                power * B
            )  # watch out, normalised by the batch size!

    def classic_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        labels = labels.long()
        variation_loss = self.smoothness_loss_coeff * Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss_coeff * self.area_loss(masks)
        cross_entropy = self.ce_coeff * self.loss_cross_entropy(logits, labels)

        loss = cross_entropy + area_loss + variation_loss

        return [loss, cross_entropy, area_loss, variation_loss]

    def bn_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        # normalize the mask
        x_norm = self.norm(masks * inp)
        return self.body(x_norm)[self.output_name]

    def ln_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        # normalize the mask
        x_norm = self.norm(masks * inp)
        return self.body(x_norm)[self.output_name]

    def new_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        if masks.dim() == 4:
            masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
            masks = F.interpolate(
                masks, size=(224, 224), mode="bilinear", align_corners=False
            )
        else:
            masks = masks[torch.arange(batches), labels, :, :, :].unsqueeze(1)
            masks = F.interpolate(
                masks, size=inp.shape[-3:], mode="trilinear", align_corners=False
            )
        x_norm = masks * inp
        return self.body(x_norm)[self.output_name]

    def old_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        x_norm = normalize(masks * invTrans(inp))
        return self.body(x_norm)[self.output_name]

    def legacy_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        x_norm = normalize(masks * inp)
        return self.body(x_norm)[self.output_name]
