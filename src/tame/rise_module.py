import os
import sys
from typing import List, Optional, Union

import cv2
import lightning.pytorch as pl
import numpy as np
import timm
import torch
import torchshow as ts
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirst, ROADMostRelevantFirst
from torchvision import transforms

from . import metrics
from . import proj_utilities as pu
from .rise import RISE


def count_FP(self, input, output):
    self.fp_count += 1


class RISEModel(pl.LightningModule):
    def __init__(
        self,
        raw_model=None,
        stats=None,
        normalized_data=True,
        input_dim: Optional[torch.Size] = None,
        img_size: Union[int, List[int]] = 224,
        batch_size: int = 32,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        count_fp=True,
        LeRF=False,
        maskpath=None,
        generate_new=False,
    ):
        super().__init__()
        if raw_model is None:
            self.raw_model = timm.create_model("deit_base_patch16_224", pretrained=True)
        else:
            self.raw_model = raw_model
        self.raw_model.fp_count = 0
        self.raw_model.register_forward_hook(count_FP)
        self.cam_model = RISE(self.raw_model, img_size, batch_size)
        self.img_size = img_size
        self.eval_length = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.raw_model,
            self.img_size,
            percent_list=percent_list,
            normalized_data=normalized_data,
            stats=stats,
        )
        self.metric_ROAD = metrics.ROAD(self.raw_model, ROADMostRelevantFirst)
        if LeRF:
            self.LeRF = True
            self.metric_ROAD2 = metrics.ROAD(self.raw_model, ROADLeastRelevantFirst)
        self.count_fp = count_fp
        self.maskpath = maskpath
        self.generate_new = generate_new

    def get_3dmask(self, image):
        with torch.set_grad_enabled(True):
            image = image.unsqueeze(0)
            image.requires_grad = True
            print(image.requires_grad)
            mask = self.cam_model(input_tensor=image)
            return mask

    @torch.no_grad()
    def save_masked_image(
        self,
        image,
        id,
        ground_truth,
        ground_truth_label,
        mdl_truth,
        mdl_truth_label,
        denormalize=False,
    ):
        image = image.unsqueeze(0)
        ts.save(image, f"_torchshow/rise/{id}/image{id}.png")
        mask = self.cam_model(image)[mdl_truth, :, :].unsqueeze(0).unsqueeze(0)
        mask = metrics.normalizeMinMax(mask)
        ts.save(mask, f"_torchshow/rise/{id}/small_mask{id}.png")
        mask = torch.nn.functional.interpolate(
            mask,
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        ts.save(mask, f"_torchshow/rise/{id}/big_mask{id}.png")
        ts.save(mask * image, f"_torchshow/rise/{id}/masked_image{id}.png")
        if denormalize:
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
            image = invTrans(image)

        opencvImage = cv2.cvtColor(
            image.squeeze().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR
        )
        opencvImage = (np.asarray(opencvImage, np.float32) * 255).astype(np.uint8)
        np_mask = np.array(mask.squeeze().cpu().numpy() * 255, dtype=np.uint8)
        np_mask = cv2.applyColorMap(np_mask, cv2.COLORMAP_JET)
        mask_image = cv2.addWeighted(np_mask, 0.5, opencvImage, 0.5, 0)
        cv2.imwrite(
            f"_torchshow/rise/{id}/mdl_{mdl_truth_label}({mdl_truth})_gr"
            f"{ground_truth_label}({ground_truth})_{id}.png",
            mask_image,
        )

    def on_test_epoch_end(self):
        ADs, ICs = self.metric_AD_IC.get_results()
        ROADs = None
        ROADs2 = None
        if self.eval_length == "long":
            ROADs = self.metric_ROAD.get_results()
            if self.LeRF is True:
                ROADs2 = self.metric_ROAD2.get_results()
        pu.on_test_epoch_end(ADs, ICs, self.logger, ROADs, ROADs2)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_epoch_start(self) -> None:
        if self.generate_new or not os.path.isfile(self.maskpath):
            self.cam_model.generate_masks(N=8000, s=7, p1=0.5, savepath=self.maskpath)
        else:
            self.cam_model.load_masks(self.maskpath)
            print("Masks are loaded.")

    def test_step(self, batch, batch_idx):
        images, _ = batch

        # this is the test loop
        logits = self.raw_model(images)
        a = 0
        if self.count_fp:
            a = self.raw_model.fp_count
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        masks = self.cam_model(images)

        if self.count_fp:
            print(f"Forward Passes: {self.raw_model.fp_count - a}")
            self.count_fp = False
        if torch.isnan(masks).any():
            sys.exit("NaNs in masks")
        masks = masks[model_truth, :].unsqueeze(dim=1)
        masks = metrics.normalizeMinMax(masks)

        self.metric_AD_IC(images, chosen_logits, model_truth, masks)
        if self.eval_length == "long":
            masks = torch.nn.functional.interpolate(
                masks,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            masks = masks.squeeze(1).cpu().detach().numpy()
            self.metric_ROAD(images, model_truth, masks)
            if self.LeRF:
                self.metric_ROAD2(images, model_truth, masks)
