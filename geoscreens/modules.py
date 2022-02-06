from pathlib import Path
from typing import List, Union
import torch.nn as nn
from icevision import models, tfms
from icevision.data import Dataset
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from icevision.models.ross.efficientdet.lightning.model_adapter import ModelAdapter as EffDetAdapter
from icevision.models.torchvision.retinanet.lightning.model_adapter import (
    ModelAdapter as TorchRetinaNetAdapter,
)
from icevision.parsers.coco_parser import COCOBBoxParser
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from geoscreens.consts import GEO_SCREENS, IMG_SIZE
from geoscreens.geo_data import GeoScreensDataModule


class LightModelEffdet(EffDetAdapter):
    def __init__(self, model: nn.Module, metrics: List[Metric] = None, learning_rate: float = 1e-1):
        super().__init__(model, metrics=metrics)
        self.learning_rate = learning_rate
        print("learning_rate: ", learning_rate)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-8
                ),
                # The unit of the scheduler's step size, could also be 'step'. 'epoch' updates the
                # scheduler on epoch end whereas 'step' updates it after a optimizer update.
                "interval": "epoch",
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "train_loss",
                # If set to `True`, will enforce that the value specified 'monitor' is available
                # when the scheduler is updated, thus stopping training if not found. If set to
                # `False`, it will only produce a warning
                "strict": True,
                # How many epochs/steps should pass between calls to `scheduler.step()`. 1
                # corresponds to updating the learning rate after every epoch/step.
                #
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress,
                # this keyword can be used to specify a custom logged name
                "name": None,
            },
        }

    def forward(self, *args, **kwargs):
        # print("")
        # print("=" * 40)
        # print("args: ", type(args), ", len: ", len(args))
        # for i, a in enumerate(args):
        #     print(f"  args[{i}]: {type(a)}")
        #     if hasattr(a, "__len__"):
        #         print(f"  len(args[{i}]): {len(a)}, {type(a[0])}")

        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], tuple):
            # print("  Weird fix")
            args = args[0][0]
            # print("  args AFTER FIX: ")
            # print("  args: ", type(args), ", len: ", len(args))
            # for i, a in enumerate(args):
            #     print(f"    args[{i}]: {type(a)}")
            #     if hasattr(a, "__len__"):
            #         print(f"    len(args[{i}]): {len(a)}, {type(a[0])}")
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # print("batch: ", batch)
        # print("batch_Idx: ", batch_idx)
        result = super().training_step(batch, batch_idx)
        # print("training_step().result type: ", type(result))
        return result

    def training_step_end(self, *args, **kwargs):
        # print("training_step_end")
        result = super().training_step_end(*args, **kwargs)
        # print("training_step_end.result type: ", type(result))
        return result


class LightModelTorch(TorchRetinaNetAdapter):
    def __init__(
        self,
        model: nn.Module,
        metrics: List[Metric] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
    ):
        super().__init__(model, metrics=metrics)
        self.learning_rate = learning_rate
        print("learning_rate: ", learning_rate)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.6, patience=10, min_lr=1e-8
                ),
                # The unit of the scheduler's step size, could also be 'step'. 'epoch' updates the
                # scheduler on epoch end whereas 'step' updates it after a optimizer update.
                "interval": "epoch",
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "train_loss",
                # If set to `True`, will enforce that the value specified 'monitor' is available
                # when the scheduler is updated, thus stopping training if not found. If set to
                # `False`, it will only produce a warning
                "strict": True,
                # How many epochs/steps should pass between calls to `scheduler.step()`. 1
                # corresponds to updating the learning rate after every epoch/step.
                #
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress,
                # this keyword can be used to specify a custom logged name
                "name": None,
            },
        }

    def forward(self, *args, **kwargs):
        # print("")
        # print("=" * 40)
        # print("args: ", type(args), ", len: ", len(args))
        # for i, a in enumerate(args):
        #     print(f"  args[{i}]: {type(a)}")
        #     if hasattr(a, "__len__"):
        #         print(f"  len(args[{i}]): {len(a)}, {type(a[0])}")

        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], tuple):
            # print("  Weird fix")
            args = args[0][0]
            # print("  args AFTER FIX: ")
            # print("  args: ", type(args), ", len: ", len(args))
            # for i, a in enumerate(args):
            #     print(f"    args[{i}]: {type(a)}")
            #     if hasattr(a, "__len__"):
            #         print(f"    len(args[{i}]): {len(a)}, {type(a[0])}")
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # print("batch: ", batch)
        # print("batch_Idx: ", batch_idx)
        result = super().training_step(batch, batch_idx)
        # print("training_step().result type: ", type(result))
        return result

    def training_step_end(self, *args, **kwargs):
        # print("training_step_end")
        result = super().training_step_end(*args, **kwargs)
        # print("training_step_end.result type: ", type(result))
        return result


def build_module(model_type: str, model, **kwargs):
    if model_type == "efficientdet":
        return LightModelEffdet(model, **kwargs)
    elif model_type == "torchvision":
        return LightModelTorch(model, **kwargs)
    else:
        raise NotImplementedError()
