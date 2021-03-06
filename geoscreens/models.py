import logging
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Tuple, Union, cast

import torch
import torch.nn as nn
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.mmdet.models.faster_rcnn.backbones.resnet_fpn import (
    MMDetFasterRCNNBackboneConfig,
)
from icevision.models.mmdet.models.retinanet.backbones.backbone_config import (
    MMDetRetinanetBackboneConfig,
)
from icevision.models.mmdet.utils import MMDetBackboneConfig
from icevision.models.ross.efficientdet.utils import EfficientDetBackboneConfig
from icevision.models.torchvision.backbones.backbone_config import TorchvisionBackboneConfig
from icevision.models.ultralytics.yolov5.utils import YoloV5BackboneConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torchvision.models.detection.anchor_utils import AnchorGenerator

from geoscreens.modules import build_module


def get_model_torchvision(
    config: DictConfig, extra_args: dict
) -> Tuple[ModuleType, TorchvisionBackboneConfig]:
    import icevision.models.torchvision.retinanet as tv

    backbone = tv.backbones.resnet50_fpn
    model_config = config.model_config.torchvision[config.model_config.name]
    sizes = model_config.params.get("anchor_sizes", [32, 64, 128, 256, 512])
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in sizes)
    ratios = model_config.params.get("aspect_ratios", [0.5, 1.0, 2.0])
    aspect_ratios = (tuple(ratios),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    # For possible params, see: $ENV/site-packages/torchvision/models/detection/retinanet.py
    extra_args.update(
        {
            "detections_per_img": model_config.params.get("detections_per_img", 300),
            "score_thresh": model_config.params.get("score_thresh", 0.05),
            "nms_thresh": model_config.params.get("nms_thresh", 0.5),
            "fg_iou_thresh": model_config.params.get("fg_iou_thresh", 0.5),
            "bg_iou_thresh": model_config.params.get("bg_iou_thresh", 0.4),
            "anchor_generator": anchor_generator,
        }
    )
    return tv, backbone


def get_model_mmdet(config: DictConfig, extra_args: dict) -> Tuple[ModuleType, MMDetBackboneConfig]:
    import icevision.models.mmdet as ice_mmdet

    # Look in ~/.icevision/mmdetection_configs/mmdetection_configs-2.20.1/configs/ for the
    # downloaded configs

    model_name = config.model_config.name
    backbone_name = config.model_config.backbone
    model_config: DictConfig = config.model_config.mmdet[model_name]

    if "cfg_options" in model_config:
        extra_args["cfg_options"] = {
            k.replace("--", "."): v for k, v in model_config.cfg_options.items()
        }
        print("Using cfg_options: ", extra_args["cfg_options"])

    if model_name.lower() == "faster_rcnn":
        backbone = cast(
            MMDetFasterRCNNBackboneConfig,
            getattr(ice_mmdet.faster_rcnn.backbones, backbone_name),
        )
        return ice_mmdet.faster_rcnn, backbone
    elif model_name.lower() == "retinanet":
        backbone = cast(
            MMDetRetinanetBackboneConfig,
            getattr(ice_mmdet.retinanet.backbones, backbone_name),
        )
        return ice_mmdet.retinanet, backbone
    elif model_name.lower() == "vfnet":
        warnings.warn(
            "VFNet issue needs to be fixed. Solution is here, but not"
            " straightforward to implement: https://github.com/open-mmlab/mmdetection/issues/6871."
        )
        backbone = ice_mmdet.vfnet.backbones.swin_t_p4_w7_fpn_1x_coco
        return ice_mmdet.retinanet, backbone
    raise NotImplementedError(f"Unsupported model: {model_name}")


def get_model_ross(
    config: DictConfig, extra_args: dict
) -> Tuple[ModuleType, EfficientDetBackboneConfig]:
    import icevision.models.ross.efficientdet as tv

    backbone = EfficientDetBackboneConfig(config.model_config.backbone)
    extra_args["img_size"] = config.dataset_config.img_size
    return tv, backbone


def get_model_ultralytics(
    config: DictConfig, extra_args: dict
) -> Tuple[ModuleType, YoloV5BackboneConfig]:
    import icevision.models.ultralytics.yolov5 as tv

    backbone = tv.backbones.small
    # The yolov5 model requires an img_size parameter
    extra_args["img_size"] = config.dataset_config.img_size
    return tv, backbone


def get_model(config: DictConfig, pretrained=True) -> Tuple[nn.Module, ModuleType]:

    model_config: DictConfig = config.model_config
    extra_args = {}

    if model_config.framework == "mmdet":
        model_module, backbone = get_model_mmdet(config, extra_args)
    elif model_config.framework == "torchvision":
        model_module, backbone = get_model_torchvision(config, extra_args)
    elif model_config.framework == "ross":
        model_module, backbone = get_model_ross(config, extra_args)
    elif model_config.framework == "ultralytics":
        model_module, backbone = get_model_ross(config, extra_args)
    else:
        raise NotImplementedError()

    model: nn.Module = model_module.model(
        backbone=backbone(pretrained=pretrained),
        num_classes=config.dataset_config.num_classes,
        **extra_args,
    )

    # print("BOOM")
    # print(model.rpn_head.anchor_generator.scales)
    # print(model.rpn_head.anchor_generator.ratios)
    # print(model.rpn_head.anchor_generator.strides)
    # sys.exit()
    modules = [backbone, model]
    if hasattr(model, "backbone"):
        modules.append(model.backbone)
    for obj in modules:
        if hasattr(obj, "param_groups"):
            delattr(obj, "param_groups")
    return model, model_module


def load_model_from_path(
    path: Union[str, Path], config_path: Union[str, Path] = None, device: torch.device = None
) -> Tuple[DictConfig, ModuleType, nn.Module, LightningModelAdapter]:
    if isinstance(path, str):
        path = Path(path).resolve()
    assert path.exists(), f"Model path does not exist: {path}"

    if path.is_dir():
        ckpt_dir = path
        ckpt_path = (
            (path / "best.ckpt") if (path / "best.ckpt").exists() else (path / "current.ckpt")
        )
        assert (
            ckpt_path.exists()
        ), f"Could not find 'best' or 'current' .ckpt file at location: {ckpt_path}"
    else:
        assert (
            path.suffix == ".ckpt"
        ), f"Invalid file extension. Expected .ckpt but got {path.suffix}"
        ckpt_path = path
        ckpt_dir = path.parent
        assert ckpt_path.exists(), f".ckpt file does not exist: {ckpt_path}"
    if not config_path:
        config_path = ckpt_dir / "config.yaml"
    elif isinstance(config_path, str):
        config_path = Path(config_path).resolve()
    assert config_path.exists(), f"Model config path does not exist: {config_path}"

    config = cast(DictConfig, OmegaConf.load(config_path))
    logging.disable(logging.INFO)
    model, module = get_model(config, pretrained=False)
    logging.disable(logging.CRITICAL)
    light_model = build_module(model, config)
    if device:
        light_model = cast(LightningModelAdapter, light_model.to(device))
    light_model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])

    return config, module, model, light_model
