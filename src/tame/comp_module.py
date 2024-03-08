import sys
from typing import List, Optional, Union

import cv2
import lightning.pytorch as pl
import numpy as np
import timm
import torch
import torchshow as ts
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayer, AblationLayerVit
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirst, ROADMostRelevantFirst
from torchvision import transforms

from tame.transformer_explainability.baselines.ViT.ViT_explanation_generator import LRP
from tame.transformer_explainability.baselines.ViT.ViT_LRP import (
    deit_base_patch16_224 as vit_LRP,
)

from . import metrics
from . import proj_utilities as pu


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def count_FP(self, input, output):
    self.fp_count += 1


# define the LightningModule
class CompareModel(pl.LightningModule):
    def __init__(
        self,
        name: str = "gradcam",
        raw_model=None,
        target_layers=None,
        stats=None,
        normalized_data=True,
        mdl_name: str = "vit_b_16",
        input_dim: Optional[torch.Size] = None,
        img_size: Union[int, List[int]] = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        example_gen: bool = False,
        count_fp=True,
        LeRF=False,
    ):
        super().__init__()
        self.method = name
        cam_method = {
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            # "fullgrad": FullGrad,
        }
        if raw_model is None:
            self.raw_model = timm.create_model("deit_base_patch16_224", pretrained=True)
        else:
            self.raw_model = raw_model
        self.raw_model.fp_count = 0
        self.raw_model.register_forward_hook(count_FP)
        if target_layers is not None:
            pass
        elif "vit" in mdl_name:
            target_layers = [self.raw_model.blocks[-1].norm1]  # type: ignore
        elif "vgg" in mdl_name:
            target_layers = [self.raw_model.features[29]]  # type: ignore
        elif "resnet" in mdl_name:
            target_layers = [self.raw_model.layer4[-1]]  # type: ignore
        else:
            raise ValueError(
                "Model not supported by default and target_layers not specified"
            )
        if name == "ablationcam":
            if "vit" in mdl_name:
                self.cam_model = cam_method[name](
                    model=self.raw_model,
                    target_layers=target_layers,
                    use_cuda=True,
                    reshape_transform=reshape_transform,
                    ablation_layer=AblationLayerVit(),  # type: ignore
                )
            else:
                self.cam_model = cam_method[name](
                    model=self.raw_model,
                    target_layers=target_layers,
                    use_cuda=True,
                    ablation_layer=AblationLayer(),  # type: ignore
                )
        else:
            self.cam_model = cam_method[name](
                model=self.raw_model,
                target_layers=target_layers,  # type: ignore
                reshape_transform=reshape_transform if raw_model is None else None,
                use_cuda=True,
            )
        if name == "scorecam" or name == "ablationcam":
            self.cam_model.batch_size = 8  # type: ignore
        if input_dim:
            self.img_size = list(input_dim[-3:])
        else:
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

    def get_3dmask(self, image):
        with torch.set_grad_enabled(True):
            image = image.unsqueeze(0)
            image.requires_grad = True
            print(image.requires_grad)
            mask = self.cam_model(input_tensor=image)
            return mask

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
        image = image.unsqueeze(0).to(self.device)
        ts.save(image, f"_torchshow/{self.method}/{id}/image{id}.png")
        mask = (
            torch.tensor(self.cam_model(input_tensor=image))
            .unsqueeze(dim=1)
            .to(self.device)
        )
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
        with torch.inference_mode(False):
            images, _ = batch
            images = images.clone()

            # this is the test loop
            logits = self.raw_model(images)
            if self.count_fp:
                print(self.raw_model.fp_count)
            logits = logits.softmax(dim=1)
            chosen_logits, model_truth = logits.max(dim=1)
            masks = self.cam_model(input_tensor=images)
            if self.count_fp:
                print(self.raw_model.fp_count)
                self.count_fp = False
            if np.isnan(masks).any():
                sys.exit("NaNs in masks")
            masks = torch.tensor(masks).unsqueeze(dim=1).to(self.device)
            masks = metrics.normalizeMinMax(masks)

            self.metric_AD_IC(images, chosen_logits, model_truth, masks)
            if self.eval_length == "long":
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                masks = masks.squeeze().cpu().detach().numpy()
                self.metric_ROAD(images, model_truth, masks)
                if self.LeRF:
                    self.metric_ROAD2(images, model_truth, masks)


class HILAVIT(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        example_gen: bool = False,
        LeRF=False,
    ):
        super().__init__()
        self.model = vit_LRP(pretrained=True)
        self.attribution_generator = LRP(self.model)
        self.model.eval()

        def gen_mask(image):
            transformer_attribution = self.attribution_generator.generate_LRP(
                image.cuda(),
                method="transformer_attribution",
            ).detach()
            transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
            transformer_attribution = torch.nn.functional.interpolate(
                transformer_attribution, scale_factor=16, mode="bilinear"
            )
            return transformer_attribution

        self.gen_mask = gen_mask
        self.model.fp_count = 0  # type: ignore
        self.model.register_forward_hook(count_FP)

        self.img_size = img_size
        self.eval_length = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.model, img_size, percent_list=percent_list, normalized_data=True
        )
        self.metric_ROAD = metrics.ROAD(self.model, ROADMostRelevantFirst)
        if LeRF:
            self.LeRF = True
            self.metric_ROAD2 = metrics.ROAD(self.model, ROADLeastRelevantFirst)
        self.once = True

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
        self.model.cuda()
        image = image.unsqueeze(0).cuda()
        ts.save(image, f"_torchshow/hila/{id}/image{id}.png")
        mask = self.gen_mask(image).clone().detach().cuda()
        mask = metrics.normalizeMinMax(mask)
        ts.save(mask, f"_torchshow/hila/{id}/small_mask{id}.png")
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        ts.save(mask, f"_torchshow/hila/{id}/big_mask{id}.png")
        ts.save(mask * image, f"_torchshow/hila/{id}/masked_image{id}.png")
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
            f"_torchshow/hila/{id}/mdl_{mdl_truth_label}({mdl_truth})_gr"
            f"{ground_truth_label}({ground_truth})_{id}.png",
            mask_image,
        )
        cv2.imwrite(
            "_torchshow/hila/heatmap.png",
            np_mask,
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
        with torch.inference_mode(False):
            images, _ = batch
            images = images.clone()

            # this is the test loop
            logits = self.model(images)
            if self.once:
                print(self.model.fp_count)
            logits = logits.softmax(dim=1)
            chosen_logits, model_truth = logits.max(dim=1)
            masks = self.gen_mask(images)
            if self.once:
                print(self.model.fp_count)
                self.once = False
            masks = metrics.normalizeMinMax(masks)
            self.metric_AD_IC(images, chosen_logits, model_truth, masks)
            if self.eval_length == "long":
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                masks = masks.squeeze(1).cpu().detach().numpy()
                self.metric_ROAD(images, model_truth, masks)
                if self.LeRF:
                    self.metric_ROAD2(images, model_truth, masks)
