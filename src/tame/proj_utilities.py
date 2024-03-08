from pathlib import Path
from typing import Any, Dict
import torch

import yaml
from lightning.pytorch.loggers import WandbLogger


def load_config(cfg: str) -> Dict[str, Any]:
    ROOT_DIR = get_project_root()
    cfg_path = ROOT_DIR / "configs" / f"{cfg}.yaml"
    with open(cfg_path) as f:
        cfg_dict: Dict[str, Any] = yaml.safe_load(f)
    with open(cfg_path.parents[0] / "default.yaml") as f:
        defaults: Dict[str, Any] = yaml.safe_load(f)
    defaults.update(cfg_dict)
    return defaults


def get_project_root() -> Path:
    return Path(__file__).parents[3]


def on_test_epoch_end(ADs, ICs, logger, ROADs=None, ROADs2=None):
    if type(logger) is WandbLogger:
        columns_adic = [
            "AD 100%",
            "IC 100%",
            "AD 50%",
            "IC 50%",
            "AD 15%",
            "IC 15%",
        ]
        data = [
            [
                ADs[0],
                ICs[0],
                ADs[1],
                ICs[1],
                ADs[2],
                ICs[2],
            ]
        ]
        logger.log_table(key="ADIC", columns=columns_adic, data=data)
        if ROADs is not None:
            columns_road = [
                "ROAD 10%",
                "ROAD 20%",
                "ROAD 30%",
                "ROAD 40%",
                "ROAD 50%",
                "ROAD 70%",
                "ROAD 90%",
            ]
            data_road = [ROADs]
            logger.log_table(key="ROAD", columns=columns_road, data=data_road)
            if ROADs2 is not None:
                logger.log_table(key="ROAD2", columns=columns_road, data=[ROADs2])

    else:
        logger.log_dict(
            {
                "AD 100%": torch.tensor(ADs[0]),
                "IC 100%": torch.tensor(ICs[0]),
                "AD 50%": torch.tensor(ADs[1]),
                "IC 50%": torch.tensor(ICs[1]),
                "AD 15%": torch.tensor(ADs[2]),
                "IC 15%": torch.tensor(ICs[2]),
            }
        )
        if ROADs is not None:
            logger.log_dict(
                {
                    "ROAD 10%": torch.tensor(ROADs[0]),
                    "ROAD 20%": torch.tensor(ROADs[1]),
                    "ROAD 30%": torch.tensor(ROADs[2]),
                    "ROAD 40%": torch.tensor(ROADs[3]),
                    "ROAD 50%": torch.tensor(ROADs[4]),
                    "ROAD 70%": torch.tensor(ROADs[5]),
                    "ROAD 90%": torch.tensor(ROADs[6]),
                }
            )
            if ROADs2 is not None:
                logger.log_dict(
                    {
                        "ROAD2 10%": torch.tensor(ROADs2[0]),
                        "ROAD2 20%": torch.tensor(ROADs2[1]),
                        "ROAD2 30%": torch.tensor(ROADs2[2]),
                        "ROAD2 40%": torch.tensor(ROADs2[3]),
                        "ROAD2 50%": torch.tensor(ROADs2[4]),
                        "ROAD2 70%": torch.tensor(ROADs2[5]),
                        "ROAD2 90%": torch.tensor(ROADs2[6]),
                    }
                )
