from pathlib import Path
from types import ModuleType
from typing import Union

from icevision import tfms
from icevision.data import Dataset, DataSplitter, RandomSplitter
from icevision.parsers import Parser
from icevision.parsers.coco_parser import COCOBBoxParser
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule


class GeoScreensDataModule(LightningDataModule):
    def __init__(self, config: Union[DictConfig, ListConfig], model_type: ModuleType):
        super().__init__()
        self.dataset_config = dataset_config = config.dataset_config
        print(self.dataset_config)
        self.cache_path = (
            Path(dataset_config.data_root) / dataset_config.dataset_name / "dataset_cache.pkl"
        )
        self.parser = COCOBBoxParser(
            annotations_filepath=(
                Path(dataset_config.data_root)
                / dataset_config.dataset_name
                / f"{dataset_config.dataset_name}.json"
            ).resolve(),
            img_dir=dataset_config.img_dir,
        )
        self.train_records, self.valid_records = self.parser.parse(
            data_splitter=RandomSplitter([0.7, 0.3], seed=233), cache_filepath=self.cache_path
        )
        print("classes: ", self.parser.class_map)

        # TODO: Remove the unnecessary augmentations, like rotations, not needed for this dataset?
        # Transforms
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
            [*tfms.A.resize_and_pad(dataset_config.img_size), tfms.A.Normalize()]
        )
        valid_tfms = tfms.A.Adapter(
            [*tfms.A.resize_and_pad(dataset_config.img_size), tfms.A.Normalize()]
        )

        # Datasets
        self.train_ds = Dataset(self.train_records, train_tfms)
        self.valid_ds = Dataset(self.valid_records, valid_tfms)
        print("train_ds num_batches: ", len(self.train_ds))
        print("valid_ds num_batches: ", len(self.valid_ds))
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
