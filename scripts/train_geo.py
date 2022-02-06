from argparse import ArgumentParser
from pathlib import Path
from typing import List
import torch.nn as nn
import wandb
from icevision import models, tfms
from icevision.data import Dataset
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from icevision.parsers.coco_parser import COCOBBoxParser
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from torch.optim import SGD
from torchvision.models.detection.anchor_utils import AnchorGenerator

from geoscreens.consts import GEO_SCREENS, IMG_SIZE, PROJECT_ROOT
from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.modules import LightModelTorch, build_module


def get_model(parser, backend_type: str = "efficientdet", pretrained=True):

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
        extra_args["img_size"] = IMG_SIZE

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


def main(args):
    seed_everything(42, workers=True)
    geo_screens = GeoScreensDataModule(num_workers=args.num_workers, batch_size=args.batch_size)
    # model_name = "efficientdet"
    model_name = "torchvision"
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox, show_pbar=True)]

    # # Auto batch_size / Learning rate search:
    # # suggested_lr = 1.9e-7
    # if False:
    #     # Note: disable DDP if using .tune() (https://github.com/PyTorchLightning/pytorch-lightning/issues/10560)
    #     model, model_type = get_model(geo_screens.parser, backend_type=model_name)
    #     geo_screens.set_model_type(model_type)
    #     light_model = build_module(model_name, model, metrics=metrics)
    #     trainer = Trainer(gpus=[0], auto_lr_find=True)
    #     trainer.tune(light_model, datamodule=geo_screens)
    #     print("Suggested learning rate:", light_model.hparams.learning_rate)
    #     suggested_lr = light_model.hparams.learning_rate

    print("creating model")
    suggested_lr = args.lr
    exp_id = f"gs005_model_{model_name}-lr_{suggested_lr}-ratios_0.08_to_2.0-sizes_32_to_512-detsperimg_512"
    save_dir = Path(f"./output/{exp_id}").resolve()
    save_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="COCOMetric",
        mode="max",
        dirpath=save_dir,
        filename="geoscreens-{epoch:02d}-coco_ap50_{COCOMetric:.2f}",
    )
    model, model_type = get_model(geo_screens.parser, backend_type=model_name)
    geo_screens.set_model_type(model_type)
    light_model = build_module(model_name, model, metrics=metrics, learning_rate=suggested_lr)
    wandb_logger = WandbLogger(project=GEO_SCREENS, log_model=True, name=exp_id)
    wandb_logger.watch(light_model)
    steps_per_batch = len(geo_screens.train_dataloader())
    trainer = Trainer(
        max_epochs=80,
        gpus=[1],
        strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
        amp_backend="native",
        logger=wandb_logger,
        check_val_every_n_epoch=5,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=min(steps_per_batch // 4, 50),
    )

    print("training")
    trainer.fit(light_model, datamodule=geo_screens)
    print("Best model: ", checkpoint_callback.best_model_path)
    # trainer.test(light_model, datamodule=geo_screens)
    wandb_logger.close()
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    # parser.add_argument("--img_dir", type=Path, default=Path("/shared/gbiamby/geo/screenshots/screen_samples_auto"))
    parser.add_argument("--img_dir", type=Path, default=(PROJECT_ROOT / "datasets/images"))
    # parser.add_argument()
    args = parser.parse_args()
    main(args)
