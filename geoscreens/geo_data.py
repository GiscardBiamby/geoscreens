from pathlib import Path
from typing import Union

from icevision import tfms
from icevision.data import Dataset, DataSplitter, RandomSplitter
from icevision.parsers import Parser
from icevision.parsers.coco_parser import COCOBBoxParser
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from geoscreens.consts import PROJECT_ROOT


class GeoScreensDataModule(LightningDataModule):
    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
    ):
        super().__init__()
        self.config = config = config.dataset_config
        print(self.config)
        self.cache_path = Path(config.data_root) / config.dataset_name / "dataset_cache.pkl"
        self.parser = COCOBBoxParser(
            annotations_filepath=(
                Path(config.data_root) / config.dataset_name / f"{config.dataset_name}.json"
            ).resolve(),
            img_dir=(PROJECT_ROOT / config.img_dir).resolve(),
        )
        self.train_records, self.valid_records = self.parser.parse(
            data_splitter=RandomSplitter([0.7, 0.3], seed=233), cache_filepath=self.cache_path
        )
        print("classes: ", self.parser.class_map)

        # TODO: Remove the unnecessary augmentations, like rotations, not needed for this dataset?
        # Transforms
        # size is set to 384 because EfficientDet requires its inputs to be divisible by 128
        # train_tfms = tfms.A.Adapter(
        #     [*tfms.A.aug_tfms(size=IMG_SIZE, presize=int(IMG_SIZE * 1.25)), tfms.A.Normalize()]
        # )
        train_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(config.img_size), tfms.A.Normalize()])
        valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(config.img_size), tfms.A.Normalize()])

        # Datasets
        self.train_ds = Dataset(self.train_records, train_tfms)
        self.valid_ds = Dataset(self.valid_records, valid_tfms)
        print("train_ds: ", len(self.train_ds))
        print("valid_ds: ", len(self.valid_ds))

    def set_model_type(self, model_type):
        self.ModelType = model_type

    def train_dataloader(self):
        return self.ModelType.train_dl(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.ModelType.valid_dl(
            self.valid_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.ModelType.valid_dl(
            self.valid_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.ModelType.valid_dl(
            self.valid_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
