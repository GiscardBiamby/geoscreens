from pathlib import Path
from typing import List, Union

import torch.nn as nn
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
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

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def log(self, *args, **kwargs):
        if self.config.training.get("distributed", False):
            kwargs["sync_dist"] = True
        super().log(*args, **kwargs)

    def finalize_metrics(self) -> None:
        _metrics = {}
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, prog_bar=True)
                    metric_key = str(k)
                    if "StatKeyPerClass" in metric_key:
                        _metrics[f"{metric.name}-PC/{k}"] = v
                    else:
                        _metrics[f"{metric.name}/{k}"] = v

        self.trainer.logger.log_metrics(_metrics, step=self.trainer.global_step)


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

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def log(self, *args, **kwargs):
        if self.config.training.get("distributed", False):
            kwargs["sync_dist"] = True
        super().log(*args, **kwargs)

    def finalize_metrics(self) -> None:
        _metrics = {}
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, prog_bar=True)
                    metric_key = str(k)
                    if "StatKeyPerClass" in metric_key:
                        _metrics[f"{metric.name}-PC/{k}"] = v
                    else:
                        _metrics[f"{metric.name}/{k}"] = v

        self.trainer.logger.log_metrics(_metrics, step=self.trainer.global_step)


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
            scheduler = ReduceLROnPlateau(optimizer, **self.config.scheduler[scheduler_type].params)
        elif scheduler_type.lower() == "multisteplr":
            scheduler = MultiStepLR(optimizer, **self.config.scheduler[scheduler_type].params)
        else:
            raise NotImplementedError(f"Unsupported scheduler type: {scheduler_type}")

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

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def log(self, *args, **kwargs):
        if self.config.training.get("distributed", False):
            kwargs["sync_dist"] = True
        super().log(*args, **kwargs)

    def finalize_metrics(self) -> None:
        _metrics = {}
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, prog_bar=True)
                    metric_key = str(k)
                    if "StatKeyPerClass" in metric_key:
                        _metrics[f"{metric.name}-PC/{k}"] = v
                    else:
                        _metrics[f"{metric.name}/{k}"] = v

        self.trainer.logger.log_metrics(_metrics, step=self.trainer.global_step)


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

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def log(self, *args, **kwargs):
        if self.config.training.get("distributed", False):
            kwargs["sync_dist"] = True
        super().log(*args, **kwargs)

    def finalize_metrics(self) -> None:
        _metrics = {}
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, prog_bar=True)
                    metric_key = str(k)
                    if "StatKeyPerClass" in metric_key:
                        _metrics[f"{metric.name}-PC/{k}"] = v
                    else:
                        _metrics[f"{metric.name}/{k}"] = v

        self.trainer.logger.log_metrics(_metrics, step=self.trainer.global_step)


def build_module(model, config: DictConfig, **kwargs) -> LightningModelAdapter:
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
