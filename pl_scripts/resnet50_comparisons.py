import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from nvitop.callbacks.pytorch_lightning import GpuStatsLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2 as transforms

import wandb
from tame import CompareModel
from tame.attention.factory import AMBuilder
from tame.attention.generic_atten import AttentionMech
from tame.iia_module import EvalMasks, MaskDataset
from tame.pl_module import TAMELIT, LightnightDataset
from tame.rise_module import RISEModel

load_dotenv()

torch.set_float32_matmul_precision("medium")

model_name = "resnet50"
raw_mdl = models.__dict__[model_name](pretrained=True).eval()

versions = [
    "gradcam",
    "gradcam++",
    "scorecam",
    "ablationcam",
    "rise",
    "tame",
    "lcam",
    "iia",
]
gpu_stats = GpuStatsLogger()
for version in versions:
    if version == "tame":
        dataset = LightnightDataset(
            dataset_path=Path(os.getenv("DATA", "./")),
            datalist_path=Path(os.getenv("LIST", "./")),
            model=model_name,
            batch_size=32,
            normalize=False,
        )
        model = TAMELIT.load_from_checkpoint(
            "T-TAME-models/checkpoints/resnet50_TTAME.ckpt",
            LeRF=True,
            train_method="raw_normalize",
            normalized_data=False,
        )
    elif version == "rise":
        dataset = LightnightDataset(
            dataset_path=Path(os.getenv("DATA", "./")),
            datalist_path=Path(os.getenv("LIST", "./")),
            model=model_name,
            batch_size=1,
        )
        model = RISEModel(
            raw_model=raw_mdl,
            eval_length="long",
            LeRF=True,
            maskpath="./masks.npy",
            generate_new=True,
            img_size=[224, 224],
            batch_size=400,
        )
    elif version == "lcam":
        dataset = LightnightDataset(
            dataset_path=Path(os.getenv("DATA", "./")),
            datalist_path=Path(os.getenv("LIST", "./")),
            model=model_name,
            batch_size=32,
            normalize=False,
        )

        class LCAM(AttentionMech):
            def __init__(self, ft_size):
                super(AttentionMech, self).__init__()
                in_channel = ft_size[0][1]
                self.op = nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=1000,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )

            def forward(self, features):
                features = list(features)
                c = self.op(features[0])
                a = torch.sigmoid(c)
                return a, c

        AMBuilder.register_attention(version, LCAM)
        model = TAMELIT.load_from_checkpoint(
            "T-TAME-models/checkpoints/resnet50_lcam.ckpt",
            LeRF=True,
            train_method="raw_normalize",
            normalized_data=False,
        )
    elif version == "iia":
        transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToDtype(torch.float, True),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dataset = DataLoader(
            MaskDataset(
                Path(os.getenv("IIA", "./")),
                "iia_resnet50",
                Path(os.getenv("DATA", "./")) / "ILSVRC2012_img_val",
                Path(os.getenv("LIST", "./")) / "Evaluation_2000.txt",
                transform,
            ),
            batch_size=32,
            num_workers=11,
            pin_memory=True,
        )
        model = EvalMasks(
            model=raw_mdl,
            LeRF=True,
        )
    else:
        if version != "ablationcam" or version != "scorecam":
            dataset = LightnightDataset(
                dataset_path=Path(os.getenv("DATA", "./")),
                datalist_path=Path(os.getenv("LIST", "./")),
                model=model_name,
                batch_size=32,
            )
        else:
            dataset = LightnightDataset(
                dataset_path=Path(os.getenv("DATA", "./")),
                datalist_path=Path(os.getenv("LIST", "./")),
                model=model_name,
                batch_size=8,
            )

        model = CompareModel(
            mdl_name=model_name, name=version, raw_model=raw_mdl, LeRF=True
        )
    wandb_logger = WandbLogger(project="T-TAME", name=model_name + "_" + version)
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[gpu_stats])
    trainer.test(model, dataset)
    wandb.finish()
