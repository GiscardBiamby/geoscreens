from types import ModuleType
from typing import Any, Tuple

import torch.nn as nn
from icevision.backbones import BackboneConfig
from icevision.models.mmdet.utils import MMDetBackboneConfig
from icevision.models.ross.efficientdet.utils import EfficientDetBackboneConfig
from icevision.models.torchvision.backbones.backbone_config import TorchvisionBackboneConfig
from icevision.models.ultralytics.yolov5.utils import YoloV5BackboneConfig
from omegaconf import DictConfig
from torchvision.models.detection.anchor_utils import AnchorGenerator


def get_model_torchvision(
    config: DictConfig, extra_args: dict
) -> Tuple[ModuleType, TorchvisionBackboneConfig]:
    import icevision.models.torchvision.retinanet as tv

    backbone = tv.backbones.resnet50_fpn
    model_config = config.model_config
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
    import icevision.models.mmdet as mmdet

    model_config: DictConfig = config.model_config
    if model_config.name.lower() == "retinanet":
        backbone = mmdet.retinanet.backbones.resnet50_fpn_1x
        return mmdet.retinanet, backbone
    elif model_config.name.lower() == "vfnet":
        backbone = mmdet.vfnet.backbones.swin_t_p4_w7_fpn_1x_coco
        return mmdet.retinanet, backbone
    raise NotImplementedError(f"Unsupported model: {model_config.name}")


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
    modules = [backbone, model]
    if hasattr(model, "backbone"):
        modules.append(model.backbone)
    for obj in modules:
        if hasattr(obj, "param_groups"):
            delattr(obj, "param_groups")
    return model, model_module
