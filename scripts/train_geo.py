from argparse import ArgumentParser
from pathlib import Path

import wandb
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from geoscreens.consts import GEO_SCREENS, IMG_SIZE, PROJECT_ROOT
from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.models import get_model
from geoscreens.modules import LightModelTorch, build_module


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
