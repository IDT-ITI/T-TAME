# checked, should be working correctly, there are differences arising from the use of torchvision.transforms
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def data_loader(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tsfm_train = transforms.Compose(
        [
            transforms.Resize(cfg["input_size"]),
            transforms.RandomCrop(cfg["crop_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    tsfm_val = transforms.Compose(
        [
            transforms.Resize(cfg["input_size"]),
            transforms.CenterCrop(cfg["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset_train = MyDataset(
        Path(cfg["dataset"]) / "ILSVRC2012_img_train",
        Path(cfg["datalist"]) / cfg["datalist_file"]
        if cfg["datalist_file"]
        else Path(cfg["datalist"]) / f"{cfg['model']}_train.txt",
        transform=tsfm_train,
    )
    dataset_val = MyDataset(
        Path(cfg["dataset"]) / "ILSVRC2012_img_val",
        Path(cfg["datalist"]) / "Validation_2000.txt",
        transform=tsfm_val,
    )
    dataset_test = MyDataset(
        Path(cfg["dataset"]) / "ILSVRC2012_img_val",
        Path(cfg["datalist"]) / "Evaluation_2000.txt",
        transform=tsfm_val,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class MyDataset(datasets.ImageFolder):
    def __init__(self, img_dir, img_list, transform=None):
        super(MyDataset, self).__init__(img_dir, transform=transform)
        self.samples = read_labeled_image_list(img_dir, img_list)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return [""], {"": 0}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return [("", 0)]


def read_labeled_image_list(directory, data_list):
    """
    Reads txt file containing paths to images and labels.

    Args:
      directory: path to the directory with images.
      data_list: path to the file with lines of the form '/path/to/image label'.

    Returns:
      List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    with open(data_list, "r") as f:
        samples = []
        for i, line in enumerate(f):
            try:
                image, label = line.strip().split()
                if "." not in image:
                    image += ".jpg"
                label = int(label)
                item = os.path.join(directory, image), label
                samples.append(item)
            except ValueError:
                print(i, line)
    return samples
