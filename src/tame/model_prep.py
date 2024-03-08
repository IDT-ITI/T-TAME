from typing import Any, Dict, List, Literal, Optional

from torch import nn, optim, Size
from torch.optim import lr_scheduler
from torchvision import models

from .composite_models import Generic
from .sam import SAM


def get_new_model(
    mdl: nn.Module,
    input_dim: Optional[Size] = None,
    cfg: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    layers: Optional[List[str]] = None,
    version: Optional[str] = None,
    masking: Optional[Literal["random", "diagonal", "max"]] = "random",
    train_method: Literal[
        "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
    ] = "new",
    num_classes=1000,
) -> Generic:
    if cfg:
        model_name = cfg["model_name"]
        layers = cfg["layers"]
        version = cfg["version"]
        masking = cfg["masking"]
        train_method = cfg["train_method"]
    assert model_name
    assert layers or layers == []
    assert version
    assert masking
    mdl = Generic(
        model_name,
        mdl,
        layers,
        version,
        masking,
        train_method,
        input_dim=input_dim,
        num_classes=num_classes,
    )
    mdl.cuda()
    return mdl


def get_model(
    cfg: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    layers: Optional[List[str]] = None,
    version: Optional[str] = None,
    masking: Optional[Literal["random", "diagonal", "max"]] = "random",
    train_method: Literal[
        "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
    ] = "new",
) -> Generic:
    if cfg:
        model_name = cfg["model_name"]
        layers = cfg["layers"]
        version = cfg["version"]
        masking = cfg["masking"]
        train_method = cfg["train_method"]

    assert model_name
    assert layers
    assert version
    assert masking

    mdl = model_prep(model_name)
    mdl = Generic(model_name, mdl, layers, version, masking, train_method)
    return mdl


def pl_get_config(
    model_name: str,
    layers: List[str],
    attention_version: str,
    masking: Literal["random", "diagonal", "max"],
    train_method: Literal[
        "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
    ],
    net_type: Literal["cnn", "transformer"],
    optimizer_type: Literal["Adam", "AdamW", "RMSProp", "SGD", "OLDSGD"],
    momentum: float,
    weight_decay: float,
    schedule_type: Literal["equal", "new_classic", "old_classic"],
    lr: float,
    epochs: int,
) -> Dict:
    cfg = {
        "model_name": model_name,
        "layers": layers,
        "version": attention_version,
        "masking": masking,
        "train_method": train_method,  # "new" or "old"
        "net_type": net_type,
        "optimizer_type": optimizer_type,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "schedule_type": schedule_type,
        "lr": lr,
        "epochs": epochs,
    }
    return cfg


def model_prep(model_name: str) -> nn.Module:
    model = models.__dict__[model_name](weights="IMAGENET1K_V1")
    return model


def get_optim(
    model: Generic,
    cfg: Optional[Dict[str, Any]] = None,
    use_sam: bool = False,
    momentum: Optional[float] = None,
    optimizer_type: Optional[
        Literal["Adam", "AdamW", "RMSProp", "SGD", "OLDSGD"]
    ] = None,
    weight_decay: Optional[float] = None,
) -> optim.Optimizer:
    if cfg:
        optimizer_type = cfg["optimizer_type"]
        momentum = cfg["momentum"]
        weight_decay = cfg["weight_decay"]
    assert optimizer_type
    assert momentum
    assert weight_decay
    g = [], [], []  # optimizer parameter groups
    # normalization layers, i.e. BatchNorm2d()
    # bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for v in model.attn_mech.modules():
        for p_name, p in v.named_parameters(recurse=False):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            # elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
            #     g[1].append(p)
            else:  # weight (with decay)
                g[0].append(p)
    if not use_sam:
        if optimizer_type == "Adam":
            # adjust beta1 to momentum
            optimizer = optim.Adam(g[2], lr=1e-7, betas=(momentum, 0.999))
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                g[2], lr=1e-7, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif optimizer_type == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=1e-7, momentum=momentum)
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(g[2], lr=1e-7, momentum=momentum, nesterov=True)
        elif optimizer_type == "OLDSGD":
            optimizer = optim.SGD(g[2], lr=2 * 1e-7, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")

        # add g0 with weight_decay
        optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})
        # # add g1 (BatchNorm2d weights)
        # optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    else:
        if optimizer_type == "Adam":
            # adjust beta1 to momentum
            optimizer = SAM(
                model.attn_mech.parameters(),
                optim.Adam,
                lr=1e-7,
                betas=(momentum, 0.999),
            )
        elif optimizer_type == "AdamW":
            optimizer = SAM(
                model.attn_mech.parameters(),
                optim.AdamW,
                lr=1e-7,
                betas=(momentum, 0.999),
                weight_decay=0.0,
            )
        elif optimizer_type == "RMSProp":
            optimizer = SAM(
                model.attn_mech.parameters(),
                optim.RMSprop,
                lr=1e-7,
                momentum=momentum,
            )
        elif optimizer_type == "SGD":
            optimizer = SAM(
                model.attn_mech.parameters(),
                optim.SGD,
                lr=1e-7,
                momentum=momentum,
                nesterov=True,
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")

    return optimizer


def get_schedule(
    optimizer: optim.Optimizer,
    currect_epoch: int,
    schedule_type: Optional[Literal["equal", "new_classic", "old_classic"]] = None,
    lr: Optional[float] = None,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    total_steps: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> lr_scheduler.LRScheduler:
    if cfg:
        schedule_type = cfg["schedule_type"]
        epochs = cfg["epochs"]
        lr = cfg["lr"]
    assert schedule_type
    assert epochs
    assert lr
    if schedule_type == "equal":
        schedule = lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            lr,
            epochs=epochs if total_steps is None else None,  # type:ignore
            steps_per_epoch=steps_per_epoch,  # type:ignore
            total_steps=total_steps,  # type:ignore
            # this denotes the last iteration, if we are just starting out it should be
            # its default
            # value, -1
            last_epoch=(currect_epoch * steps_per_epoch)  # type:ignore
            if currect_epoch != 0
            else -1,
        )
    elif schedule_type == "new_classic":
        schedule = lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            [lr, 2 * lr, 2 * lr],
            epochs=epochs if total_steps is None else None,  # type:ignore
            steps_per_epoch=steps_per_epoch,  # type:ignore
            total_steps=total_steps,  # type:ignore
            # this denotes the last iteration, if we are just starting out it should be
            # its default
            # value, -1
            last_epoch=(currect_epoch * steps_per_epoch)  # type:ignore
            if currect_epoch != 0
            else -1,
        )
    elif schedule_type == "old_classic":
        schedule = lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            [2 * lr, lr],
            epochs=epochs if total_steps is None else None,  # type:ignore
            steps_per_epoch=steps_per_epoch,  # type:ignore
            total_steps=total_steps,  # type:ignore
            # this denotes the last iteration, if we are just starting out it should be
            # its default
            # value, -1
            last_epoch=(currect_epoch * steps_per_epoch)  # type:ignore
            if currect_epoch != 0
            else -1,
        )
    else:
        raise NotImplementedError(f'Schedule type "{schedule_type}" not implemented.')
    return schedule
