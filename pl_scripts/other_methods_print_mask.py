import os
from copy import copy
from pathlib import Path

import torch
from dotenv import load_dotenv
from imagenet_classes import imagenet_classes
from torchvision import models
from torchvision.transforms import v2 as transforms

from tame.comp_module import HILAVIT, CompareModel
from tame.iia_module import EvalMasks, MaskDataset
from tame.pl_module import LightnightDataset
from tame.rise_module import RISEModel

load_dotenv()

torch.set_float32_matmul_precision("medium")

models_list = ["vgg16", "resnet50", "vit_b_16"]

model_name = models_list[1]
image_ids = [227]
mdl = models.__dict__[model_name](pretrained=False).cuda()
cam_method = ["gradcam", "scorecam", "gradcam++", "ablationcam", "hila", "rise", "iia"]
datamodule = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=1,
)
datamodule.test_dataloader(True)
dataset = datamodule.test_dataset
for id in image_ids:
    mdl_truth = mdl(dataset[id][0].unsqueeze(0).cuda()).argmax().item()
    print("Truth: ", imagenet_classes[dataset[id][1]])
    print("Model Truth: ", imagenet_classes[mdl_truth])
    for method in cam_method:
        model = CompareModel(name=method, raw_model=mdl, mdl_name=model_name)

        model.save_masked_image(
            dataset[id][0],
            id,
            dataset[id][1],
            imagenet_classes[dataset[id][1]],
            mdl_truth,
            imagenet_classes[mdl_truth],
            denormalize=True,
        )
    if "hila" in cam_method:
        model = HILAVIT()
        model.save_masked_image(
            dataset[id][0],
            id,
            dataset[id][1],
            imagenet_classes[dataset[id][1]],
            mdl_truth,
            imagenet_classes[mdl_truth],
            denormalize=True,
        )
    if "rise" in cam_method:
        model = RISEModel(
            raw_model=copy.deepcopy(mdl),
            maskpath="./masks.npy",
            generate_new=True,
            img_size=[224, 224],
            batch_size=800,
        )
        model.on_test_epoch_start()
        model.cuda()
        model.save_masked_image(
            dataset[id][0].cuda(),
            id,
            dataset[id][1],
            imagenet_classes[dataset[id][1]],
            mdl_truth,
            imagenet_classes[mdl_truth],
            denormalize=True,
        )
    if "iia" in cam_method:
        transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToDtype(torch.float32, True),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dataset = MaskDataset(
            Path(os.getenv("IIA", "./")),
            "iia_resnet50",
            Path(os.getenv("DATA", "./")) / "ILSVRC2012_img_val",
            Path(os.getenv("LIST", "./")) / "Evaluation_2000.txt",
            transform,
        )
        model = EvalMasks(
            model=mdl,
            LeRF=True,
        )
        model.cuda()
        model.save_masked_image(
            dataset[id][0].cuda(),
            dataset[id][1].cuda(),
            id,
            dataset[id][2],
            imagenet_classes[dataset[id][2]],
            mdl_truth,
            imagenet_classes[mdl_truth],
        )
