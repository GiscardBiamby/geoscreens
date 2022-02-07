from types import ModuleType
from typing import Any, Tuple

from icevision import models
from omegaconf import DictConfig
from torchvision.models.detection.anchor_utils import AnchorGenerator


def get_model_torchvision(config: DictConfig, extra_args):
    # The Retinanet model is also implemented in the torchvision library
    model_type = models.torchvision.retinanet
    backbone = models.torchvision.retinanet.backbones.resnet50_fpn
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
    return model_type, backbone


def get_model(config: DictConfig, parser, pretrained=True) -> Tuple[Any, ModuleType]:

    model_config: DictConfig = config.model_config
    extra_args = {}

    if model_config.framework == "mmdet":
        model_type = models.mmdet.retinanet
        backbone = model_type.backbones.resnet50_fpn_1x

    elif model_config.framework == "torchvision":
        model_type, backbone = get_model_torchvision(config, extra_args)

    elif model_config.framework == "ross":
        model_type = models.ross.efficientdet
        backbone = model_type.backbones.tf_lite0
        # The efficientdet model requires an img_size parameter
        extra_args["img_size"] = config.datataset_config.img_size

    elif model_config.framework == "ultralytics":
        model_type = models.ultralytics.yolov5
        backbone = model_type.backbones.small
        # The yolov5 model requires an img_size parameter
        extra_args["img_size"] = image_size
    else:
        raise NotImplementedError()

    model = model_type.model(
        backbone=backbone(pretrained=pretrained), num_classes=len(parser.class_map), **extra_args
    )
    # print(model)
    for obj in [backbone, model, model.backbone]:
        if hasattr(obj, "param_groups"):
            delattr(obj, "param_groups")
    return model, model_type
