from types import ModuleType
from typing import Any, Tuple

from icevision import models
from omegaconf import DictConfig
from torchvision.models.detection.anchor_utils import AnchorGenerator

from geoscreens.consts import IMG_SIZE


def get_model(
    config: DictConfig, parser, backend_type: str = "efficientdet", pretrained=True
) -> Tuple[Any, ModuleType]:

    extra_args = {}

    if backend_type == "mmdet":
        model_type = models.mmdet.retinanet
        backbone = model_type.backbones.resnet50_fpn_1x

    elif backend_type == "torchvision":
        # The Retinanet model is also implemented in the torchvision library
        model_type = models.torchvision.retinanet
        backbone = model_type.backbones.resnet50_fpn

        anchor_sizes = tuple(
            (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]
        )
        aspect_ratios = ((0.08, 0.16, 0.25, 0.36, 0.5, 0.7, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        extra_args.update(
            {
                "detections_per_img": 512,
                "anchor_generator": anchor_generator,
            }
        )

    elif backend_type == "efficientdet":
        model_type = models.ross.efficientdet
        backbone = model_type.backbones.tf_lite0
        # The efficientdet model requires an img_size parameter
        extra_args["img_size"] = config.datataset_config.img_size

    elif backend_type == "ultralytics":
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
