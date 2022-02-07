import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Union, cast

import wandb
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger, LoggerCollection
from pytorch_lightning.plugins import DDPPlugin

from geoscreens.config import build_config
from geoscreens.consts import GEO_SCREENS, PROJECT_ROOT
from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.models import get_model
from geoscreens.modules import LightModelTorch, build_module


def param_search(config, geo_screens, model_name, metrics):
    # Auto batch_size / Learning rate search:
    # suggested_lr = 1.9e-7
    if False:
        # Note: disable DDP if using .tune()
        # (https://github.com/PyTorchLightning/pytorch-lightning/issues/10560)
        model, model_type = get_model(geo_screens.parser, backend_type=model_name)
        geo_screens.set_model_type(model_type)
        light_model = build_module(model_name, model, metrics=metrics)
        trainer = Trainer(gpus=[0], auto_lr_find=True)
        trainer.tune(light_model, datamodule=geo_screens)
        print("Suggested learning rate:", light_model.hparams.learning_rate)
        suggested_lr = light_model.hparams.learning_rate
    pass


def monitor_criteria(config: DictConfig):
    monitor_criteria = config.training.early_stop.get("criteria", None)
    assert monitor_criteria, "monitor criteria is required when early stop is specified."
    if "val" not in monitor_criteria:
        monitor_criteria = f"val/{monitor_criteria}"
    mode = "min" if config.training.early_stop.get("minimize", False) else "max"
    return monitor_criteria, mode


def configure_monitor_callbacks(config: DictConfig) -> List[ModelCheckpoint]:
    criteria, mode = monitor_criteria(config)
    monitor_callback = ModelCheckpoint(
        monitor=criteria,
        dirpath=config.env.save_dir,
        filename="best",
        mode=mode,
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    return [monitor_callback]


def configure_earlystop_callback(config: DictConfig) -> List[ModelCheckpoint]:
    return []


def configure_checkpoint_callbacks(config: DictConfig) -> List[ModelCheckpoint]:
    checkpoint_callback = ModelCheckpoint(
        monitor="COCOMetric",
        every_n_train_steps=config.training.checkpoint_interval,
        dirpath=config.env.save_dir,
        filename="geoscreens-{epoch:02d}-coco_ap50_{COCOMetric:.2f}",
        mode="max",
        save_last=True,
        verbose=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "current"
    return [checkpoint_callback]


def configure_callbacks(config: DictConfig) -> List[Callback]:
    callbacks = []
    callbacks += configure_checkpoint_callbacks(config)
    if config.training.get("early_stop", None) and config.training.early_stop.get("enabled", False):
        callbacks += configure_monitor_callbacks(config)
        callbacks += configure_earlystop_callback(config)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def train_geo(config: DictConfig) -> None:
    seed_everything(config.seed, workers=True)
    geoscreens_data = GeoScreensDataModule(config)
    # TODO: Remove this, use config object instead
    model_name = config.model_config.framework
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox, show_pbar=True)]

    print("creating model")

    model, model_type = get_model(config, geoscreens_data.parser, backend_type=model_name)
    geoscreens_data.set_model_type(model_type)
    light_model = build_module(model_name, model, config, metrics=metrics)
    wandb_logger = build_wandb_logger(config, light_model)
    callbacks = configure_callbacks(config)
    trainer = Trainer(
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=wandb_logger or DummyLogger(),
        callbacks=callbacks,
        log_every_n_steps=min(len(geoscreens_data.train_dataloader()) // 4, 50),
        **config.training.params,
    )

    trainer.fit(light_model, datamodule=geoscreens_data)
    checkpoint_callback = get_checkpoint_callback(callbacks)
    print("Best model: ", checkpoint_callback.best_model_path)
    # trainer.test(light_model, datamodule=geo_screens)
    if config.training.wandb.enabled and wandb_logger:
        wandb_logger.finalize("")
        wandb.finish()


def get_checkpoint_callback(callbacks):
    checkpoint_callback = cast(
        ModelCheckpoint,
        next(
            (
                cb
                for cb in callbacks
                if isinstance(cb, ModelCheckpoint) and cb.CHECKPOINT_NAME_LAST == "current"
            ),
            None,
        ),
    )

    return checkpoint_callback


def build_wandb_logger(config, light_model) -> Optional[WandbLogger]:
    if config.training.wandb.enabled:
        wandb_config: DictConfig = config.training.wandb.copy()
        OmegaConf.set_struct(wandb_config, False)
        wandb_config.pop("enabled")
        wandb_logger = WandbLogger(
            config=OmegaConf.to_container(config, resolve=True),
            **wandb_config,
        )
        wandb_logger.watch(light_model)
        return wandb_logger
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=Path,
        default=(PROJECT_ROOT / "configs" / "torchvision.retinanet.yaml"),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=svalue arguments to override config values "
        "(use dots for.nested=overrides)",
    )
    args = parser.parse_args()
    config = build_config(args)
    train_geo(config)
