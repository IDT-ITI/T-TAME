import os
from pathlib import Path

from dotenv import load_dotenv
from imagenet_classes import imagenet_classes

from tame.pl_module import TAMELIT, LightnightDataset

load_dotenv()
model_list = ["vgg16", "resnet50", "vit_b_16"]
chosen_model = model_list[0]
# vgg
if chosen_model == "vgg16":
    model_name = "vgg16"
    model = TAMELIT.load_from_checkpoint(
        "logs/TAME_vgg16_oldnorm/version_0/checkpoints/epoch=7-step=320296.ckpt",
        train_method="raw_normalize",
    )
# vit
elif chosen_model == "vit_b_16":
    model_name = "vit_b_16"
    model = TAMELIT.load_from_checkpoint(
        "logs/TAME_vit_b_16/version_0/checkpoints/epoch=7-step=320296.ckpt"
    )
# resnet50
elif chosen_model == "resnet50":
    model_name = "resnet50"
    model = TAMELIT.load_from_checkpoint(
        "logs/TAME_resnet50/version_5/checkpoints/epoch=3-step=160148.ckpt",
        train_method="raw_normalize",
    )

# model: pl.LightningModule = torch.compile(model)  # type: ignore
# FOR CNN
if chosen_model in ["vgg16", "resnet50"]:
    for id in [42]:
        dataset = LightnightDataset(
            dataset_path=Path(os.getenv("DATA", "./")),
            datalist_path=Path(os.getenv("LIST", "./")),
            model=model_name,
            batch_size=32,
            normalize=False,
        )
        dataset.test_dataloader(dev=True)
        dataset = dataset.test_dataset
        model.save_masked_image(
            dataset[id][0],
            id,
            model_name,
            dataset[id][1],
            imagenet_classes[dataset[id][1]],
            imagenet_classes,
            select_mask=129,
            # denormalize=True,
        )

# FOR VIT
elif chosen_model == "vit_b_16":
    for id in [456]:
        dataset = LightnightDataset(
            dataset_path=Path(os.getenv("DATA", "./")),
            datalist_path=Path(os.getenv("LIST", "./")),
            model=model_name,
            batch_size=32,
        )
        dataset.test_dataloader(dev=True)
        dataset = dataset.test_dataset
        model.save_masked_image(
            dataset[id][0],
            id,
            model_name,
            dataset[id][1],
            imagenet_classes[dataset[id][1]],
            imagenet_classes,
            select_mask=196,
            denormalize=True,
        )
