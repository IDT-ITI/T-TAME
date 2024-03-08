import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger

from tame.pl_module import TAMELIT, LightnightDataset

load_dotenv()
torch.set_float32_matmul_precision("medium")

version = "TAME"
model_name = "vit_b_16"
layers = [
    "encoder.layers.encoder_layer_9.ln",
    "encoder.layers.encoder_layer_10.ln",
    "encoder.layers.encoder_layer_11.ln",
]
epochs = 8

model = TAMELIT(
    model_name=model_name,
    layers=layers,
    attention_version=version,
    train_method="new",
    lr=0.001,
    epochs=epochs,
)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
)

checkpointer = ModelCheckpoint(every_n_epochs=1, save_on_train_epoch_end=False)

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
