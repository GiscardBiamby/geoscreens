from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Union, cast

from icevision import tfms
from icevision.data import Dataset, DataSplitter, RandomSplitter
from icevision.data.data_splitter import SingleSplitSplitter
from icevision.parsers import Parser
from icevision.parsers.coco_parser import COCOBBoxParser
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule


class GeoScreensDataModule(LightningDataModule):
    def __init__(self, config: Union[DictConfig, ListConfig], model_type: ModuleType):
        super().__init__()
        self.dataset_config = dataset_config = config.dataset_config
        print(self.dataset_config)

        #
        ds_base_path = (Path(dataset_config.data_root) / dataset_config.dataset_name).resolve()
        ds_path = ds_base_path / f"{dataset_config.dataset_name}.json"
        train_ds_path = ds_base_path / f"{dataset_config.dataset_name}_train.json"
        val_ds_path = ds_base_path / f"{dataset_config.dataset_name}_val.json"

        self.cache_path = (
            Path(dataset_config.data_root) / dataset_config.dataset_name / "dataset_cache.pkl"
        )
        self.cache_path_train = ds_path.with_name(f"dataset_cache_train.pkl")
        self.cache_path_valid = ds_path.with_name(f"dataset_cache_valid.pkl")

        if train_ds_path.exists():
            self.parser = COCOBBoxParser(
                annotations_filepath=train_ds_path, img_dir=dataset_config.img_dir
            )
            self.val_parser = COCOBBoxParser(
                annotations_filepath=val_ds_path, img_dir=dataset_config.img_dir
            )
            self.train_records, _ = self.parser.parse(
                data_splitter=RandomSplitter([1.0, 0.0], seed=233),
                cache_filepath=self.cache_path_train,
            )
            self.valid_records, _ = self.val_parser.parse(
                data_splitter=RandomSplitter([1.0, 0.0], seed=233),
                cache_filepath=self.cache_path_valid,
            )
        else:
            self.parser = COCOBBoxParser(
                annotations_filepath=ds_path, img_dir=dataset_config.img_dir
            )
            self.train_records, self.valid_records = self.parser.parse(
                data_splitter=SingleSplitSplitter(), cache_filepath=self.cache_path
            )
        print("classes: ", self.parser.class_map)
        self.id_to_class = cast(
            Dict[int, str],
            {cat_id: label for label, cat_id in self.parser.class_map._class2id.items()},
        )
        self.class_to_id = self.parser.class_map._class2id

        # size is set to 384 because EfficientDet requires its inputs to be divisible by 128
        # train_tfms = tfms.A.Adapter(
        #     [
        #         *tfms.A.aug_tfms(
        #             size=dataset_config.img_size, presize=int(dataset_config.img_size * 1.25)
        #         ),
        #         tfms.A.Normalize(),
        #     ]
        # )
        train_tfms = tfms.A.Adapter(
            [
                *tfms.A.resize_and_pad(dataset_config.img_size),
                *tfms.A.aug_tfms(
                    size=dataset_config.img_size,
                    presize=None,
                    horizontal_flip=None,
                    shift_scale_rotate=tfms.A.ShiftScaleRotate(
                        rotate_limit=0,
                        shift_limit=0.2,
                        scale_limit=0.2,
                        p=0.5,
                    ),
                    lightning=None,
                    crop_fn=None,
                    pad=None,
                ),
                tfms.A.ColorJitter(),
                tfms.A.Blur(),
                tfms.A.ImageCompression(),
                tfms.A.Normalize(),
            ],
        )
        valid_tfms = tfms.A.Adapter(
            [*tfms.A.resize_and_pad(dataset_config.img_size), tfms.A.Normalize()]
        )

        # Datasets
        self.train_ds = Dataset(self.train_records, train_tfms)
        self.valid_ds = Dataset(self.valid_records, valid_tfms)
        print("train_ds num_images: ", len(self.train_ds))
        print("valid_ds num_images: ", len(self.valid_ds))
        self.ModelType = model_type

    def train_dataloader(self):
        # Can pass any arguments here as **kwargs, and they will be passed into the constructor for
        # a pytorch DataLoader:
        return self.ModelType.train_dl(
            self.train_ds,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=True,
            pin_memory=self.dataset_config.pin_memory,
        )

    def val_dataloader(self):
        return self.ModelType.valid_dl(
            self.valid_ds,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=False,
            pin_memory=self.dataset_config.pin_memory,
        )

    def test_dataloader(self):
        return self.ModelType.valid_dl(
            self.valid_ds,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=False,
            pin_memory=self.dataset_config.pin_memory,
        )

    def predict_dataloader(self):
        return self.ModelType.valid_dl(
            self.valid_ds,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            shuffle=False,
            pin_memory=self.dataset_config.pin_memory,
        )
