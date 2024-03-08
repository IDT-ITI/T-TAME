import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from tame.pl_module import TAMELIT, LightnightDataset

load_dotenv()

torch.set_float32_matmul_precision("medium")
version = "TAME"
model_name = "resnet50"
layers = [
    "layer2",
    "layer3",
    "layer4",
]
epochs = 8

model = TAMELIT(
    model_name=model_name,
    layers=layers,
    attention_version=version,
    train_method="raw_normalize",
    normalized_data=False,
    lr=0.001,
    epochs=epochs,
    LeRF=True,
)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
    normalize=False,
)

checkpointer = ModelCheckpoint(
    every_n_epochs=1, save_on_train_epoch_end=False, save_top_k=-1
)

# torch._dynamo.config.verbose=True
wandb_logger = WandbLogger(project="T-TAME", name=model_name + "_" + version)

trainer = pl.Trainer(
    precision="16-mixed",
    gradient_clip_algorithm="norm",
    max_epochs=epochs,
    logger=wandb_logger,
    callbacks=[checkpointer],
)
trainer.fit(model, dataset)

trainer.test(model, dataset)
wandb.finish()
