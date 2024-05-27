import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from nvitop.callbacks.pytorch_lightning import GpuStatsLogger
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2 as transforms

import wandb
from tame import HILAVIT, CompareModel
from tame.iia_module import EvalMasks, MaskDataset
from tame.pl_module import TAMELIT, LightnightDataset
from tame.rise_module import RISEModel

load_dotenv()

torch.set_float32_matmul_precision("medium")

model_name = "vit_b_16"
raw_mdl = models.__dict__[model_name](pretrained=True).eval()

versions = [
    "gradcam",
    "gradcam++",
    "scorecam",
    "ablationcam",
    "rise",
    "tame",
    "hila",
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
        )

        model = TAMELIT.load_from_checkpoint(
            "checkpoints/vit_b_16_TTAME.ckpt",
            LeRF=True,
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
            batch_size=800,
        )
    elif version == "hila":
        dataset = LightnightDataset(
            dataset_path=Path(os.getenv("DATA", "./")),
            datalist_path=Path(os.getenv("LIST", "./")),
            model=model_name,
            batch_size=1,
        )

        model = HILAVIT(LeRF=True)
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
                "iia_vit-base",
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
                batch_size=16,
            )

        model = CompareModel(mdl_name=model_name, name=version, LeRF=True)
    wandb_logger = WandbLogger(
        project="T-TAME", name=model_name + "_" + "LeRF" + version
    )
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[gpu_stats])
    trainer.test(model, dataset)
    wandb.finish()
