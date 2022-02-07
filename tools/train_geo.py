import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import wandb
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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


def train_geo(config: DictConfig) -> None:
    seed_everything(config.seed, workers=True)
    print("CONFIG: ", *config.training)
    geo_screens = GeoScreensDataModule(config)
    # model_name = "efficientdet"
    model_name = config.model_config.framework
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox, show_pbar=True)]

    # param_search(config, geo_screens, model_name, metrics)

    print("creating model")
    exp_id = f"gs005_model_{model_name}-lr_{config.training.learning_rate}-ratios_0.08_to_2.0-sizes_32_to_512-detsperimg_512"
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
    light_model = build_module(model_name, model, config, metrics=metrics)
    if config.training.wandb.enabled:
        wandb_config: DictConfig = config.training.wandb.copy()
        OmegaConf.set_struct(wandb_config, False)
        wandb_config.pop("enabled")
        wandb_logger = WandbLogger(
            config=OmegaConf.to_container(config, resolve=True),
            **wandb_config,
        )
        wandb_logger.watch(light_model)
    steps_per_batch = len(geo_screens.train_dataloader())
    trainer = Trainer(
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=wandb_logger if config.training.wandb.enabled else DummyLogger(),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=min(steps_per_batch // 4, 50),
        **config.training.params,
    )

    print("training")
    trainer.fit(light_model, datamodule=geo_screens)
    print("Best model: ", checkpoint_callback.best_model_path)
    # trainer.test(light_model, datamodule=geo_screens)
    if config.training.wandb.enabled:
        wandb_logger.close()
        wandb.finish()


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
