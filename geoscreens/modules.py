from pathlib import Path
from typing import List, Union

import torch.nn as nn
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from icevision.models.mmdet.common.bbox.lightning.model_adapter import ModelAdapter as MMDetAdapter
from icevision.models.ross.efficientdet.lightning.model_adapter import ModelAdapter as EffDetAdapter
from icevision.models.torchvision.retinanet.lightning.model_adapter import (
    ModelAdapter as TorchRetinaNetAdapter,
)
from icevision.models.ultralytics.yolov5.lightning.model_adapter import ModelAdapter as YoloAdapter
from omegaconf import DictConfig
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau


class LightModelEffdet(EffDetAdapter):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        metrics: List[Metric] = None,
    ):
        super().__init__(model, metrics=metrics)
        self.config = config
        self.learning_rate = config.optimizer.params.lr
        print("learning_rate: ", self.learning_rate)
        self.save_hyperparameters(self.config.training)

    def configure_optimizers(self):
        optimizer_type = self.config.optimizer.type
        if optimizer_type.lower() == "adam":
            optimizer = Adam(self.parameters(), **self.config.optimizer.params)
        else:
            raise NotImplementedError()

        scheduler_type = self.config.scheduler.type
        if scheduler_type.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, **self.config.scheduler.params)
        else:
            raise NotImplementedError()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.config.scheduler.config,
            },
        }

    def forward(self, *args, **kwargs):
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], tuple):
            args = args[0][0]
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        return result

    def training_step_end(self, *args, **kwargs):
        result = super().training_step_end(*args, **kwargs)
        return result


class LightModelTorch(TorchRetinaNetAdapter):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        metrics: List[Metric] = None,
    ):
        super().__init__(model, metrics=metrics)
        self.config = config
        self.learning_rate = config.optimizer.params.lr
        print("learning_rate: ", self.learning_rate)
        self.save_hyperparameters(self.config.training)

    def configure_optimizers(self):
        optimizer_type = self.config.optimizer.type
        if optimizer_type.lower() == "adam":
            optimizer = Adam(self.parameters(), **self.config.optimizer.params)
        else:
            raise NotImplementedError()

        scheduler_type = self.config.scheduler.type
        if scheduler_type.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, **self.config.scheduler.params)
        else:
            raise NotImplementedError()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.config.scheduler.config,
            },
        }

    def forward(self, *args, **kwargs):
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], tuple):
            args = args[0][0]
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        return result

    def training_step_end(self, *args, **kwargs):
        result = super().training_step_end(*args, **kwargs)
        return result


class LightModelMMDet(MMDetAdapter):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        metrics: List[Metric] = None,
    ):
        super().__init__(model, metrics=metrics)
        self.config = config
        self.learning_rate = config.optimizer.params.lr
        print("learning_rate: ", self.learning_rate)
        self.save_hyperparameters(self.config.training)

    def configure_optimizers(self):
        optimizer_type = self.config.optimizer.type
        if optimizer_type.lower() == "adam":
            optimizer = Adam(self.parameters(), **self.config.optimizer.params)
        else:
            raise NotImplementedError()

        scheduler_type = self.config.scheduler.type
        if scheduler_type.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, **self.config.scheduler.params)
        else:
            raise NotImplementedError()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.config.scheduler.config,
            },
        }

    def forward(self, *args, **kwargs):
        # if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], tuple):
        #     args = args[0][0]
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        return result

    def training_step_end(self, *args, **kwargs):
        result = super().training_step_end(*args, **kwargs)
        return result


class LightModelUltralytics(YoloAdapter):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        metrics: List[Metric] = None,
    ):
        super().__init__(model, metrics=metrics)
        self.config = config
        self.learning_rate = config.optimizer.params.lr
        print("learning_rate: ", self.learning_rate)
        self.save_hyperparameters(self.config.training)

    def configure_optimizers(self):
        optimizer_type = self.config.optimizer.type
        if optimizer_type.lower() == "adam":
            optimizer = Adam(self.parameters(), **self.config.optimizer.params)
        else:
            raise NotImplementedError()

        scheduler_type = self.config.scheduler.type
        if scheduler_type.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, **self.config.scheduler.params)
        else:
            raise NotImplementedError()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.config.scheduler.config,
            },
        }

    def forward(self, *args, **kwargs):
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], tuple):
            args = args[0][0]
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        return result

    def training_step_end(self, *args, **kwargs):
        result = super().training_step_end(*args, **kwargs)
        return result


def build_module(model, config: DictConfig, **kwargs):
    model_config: DictConfig = config.model_config
    if model_config.framework == "mmdet":
        return LightModelMMDet(model, config, **kwargs)
    elif model_config.framework == "ross":
        return LightModelEffdet(model, config, **kwargs)
    elif model_config.framework == "torchvision":
        return LightModelTorch(model, config, **kwargs)
    elif model_config.framework == "ultralytics":
        return LightModelUltralytics(model, config, **kwargs)
    else:
        raise NotImplementedError()
