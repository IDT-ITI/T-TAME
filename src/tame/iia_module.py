import sys
from pathlib import Path
from typing import List

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torchshow as ts
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirst, ROADMostRelevantFirst
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as transforms

from . import metrics
from . import proj_utilities as pu


class EvalMasks(pl.LightningModule):
    def __init__(
        self,
        model,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        LeRF=False,
        maskpath=None,
        img_size=224,
    ):
        super().__init__()
        self.model = model
        self.eval_length = eval_length
        self.img_size = img_size
        self.metric_AD_IC = metrics.AD_IC(
            self.model,
            self.img_size,
            percent_list=percent_list,
            normalized_data=True,
        )
        self.metric_ROAD = metrics.ROAD(self.model, ROADMostRelevantFirst)
        self.LeRF = LeRF
        if LeRF:
            self.metric_ROAD2 = metrics.ROAD(self.model, ROADLeastRelevantFirst)

    def save_masked_image(
        self,
        image,
        mask,
        id,
        ground_truth,
        ground_truth_label,
        mdl_truth,
        mdl_truth_label,
        denormalize=True,
    ):
        self.method = "iia"
        image = image.unsqueeze(0).to(self.device)
        ts.save(image, f"_torchshow/{self.method}/{id}/image{id}.png")
        mask = mask.unsqueeze(0).unsqueeze(0).to(self.device)
        mask = metrics.normalizeMinMax(mask)
        ts.save(mask, f"_torchshow/{self.method}/{id}/small_mask{id}.png")
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        ts.save(mask, f"_torchshow/{self.method}/{id}/big_mask{id}.png")
        ts.save(mask * image, f"_torchshow/{self.method}/{id}/masked_image{id}.png")
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
            f"_torchshow/{self.method}/{id}/mdl_{mdl_truth_label}({mdl_truth})_gr"
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

    def test_step(self, batch, batch_idx):
        images, masks, _ = batch
        masks = masks.unsqueeze(1)
        # this is the test loop
        logits = self.model(images)
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)

        if torch.isnan(masks).any():
            sys.exit("NaNs in mask")
        masks = metrics.normalizeMinMax(masks)
        self.metric_AD_IC(images, chosen_logits, model_truth, masks)
        if self.eval_length == "long":
            masks = masks.squeeze(1).cpu().detach().numpy()
            self.metric_ROAD(images, model_truth, masks)
            if self.LeRF:
                self.metric_ROAD2(images, model_truth, masks)


class MaskDataset(Dataset):
    def __init__(
        self,
        maskpath: Path,
        masksuffix: str,
        datapath: Path,
        datalist_path: Path,
        transform,
    ):
        self.transform = transform
        self.samples = self.load_samples(datapath, maskpath, masksuffix, datalist_path)

    @staticmethod
    def load_samples(
        datapath: Path, maskpath: Path, masksuffix: str, datalist_path: Path
    ):
        with open(datalist_path, "r") as f:
            samples = []
            for i, line in enumerate(f):
                try:
                    image, label = line.strip().split()
                    if "." not in image:
                        image += ".jpg"
                    label = int(label)
                    image_path = datapath / image
                    mask_path = maskpath / (
                        str(Path(image).stem) + "_" + masksuffix + ".npy"
                    )
                    item = image_path, mask_path, label
                    samples.append(item)
                except ValueError:
                    print(i, line)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, label = self.samples[idx]
        image = read_image(str(image_path), ImageReadMode.RGB)
        mask_np = np.load(mask_path).astype(np.float32)
        mask = torch.from_numpy(mask_np)
        to_float = transforms.ToDtype(torch.float)
        mask = to_float(mask)
        mask = mask.max() - mask
        if self.transform:
            image = self.transform(image)
        return image, mask, label
