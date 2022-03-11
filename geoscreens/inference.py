import json
import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from icevision import models, tfms
from icevision.core import ClassMap, tasks
from icevision.core.record import BaseRecord
from icevision.core.record_components import ClassMapRecordComponent, ImageRecordComponent
from icevision.data import Dataset
from icevision.tfms import Transform
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningDataModule, seed_everything
from tqdm.contrib.bells import tqdm

from geoscreens.consts import FRAMES_METADATA_PATH
from geoscreens.data.metadata import FramesList, get_geoguessr_split_metadata
from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.models import load_model_from_path
from geoscreens.utils import Singleton, load_json

__all__ = ["GeoscreensInferenceDataset", "get_model_for_inference", "get_detections"]


def get_model_for_inference(args):
    seed_everything(42, workers=True)
    DEVICE = torch.device(f"cuda:{args.device}")

    print("Loading model to device: ", DEVICE)
    config, module, model, light_model = load_model_from_path(args.checkpoint_path, device=DEVICE)
    light_model.eval()
    geoscreens_data = GeoScreensDataModule(config, module)
    return config, module, model, light_model, geoscreens_data


class GeoscreensInferenceDataset(object):
    """
    Only usable for inference.

    Provides a dataset over a folder with video frames in form::

        <video_id_1>/
            frame_....jpg
        <video_id_2>/
            frame_....jpg

    If no video_id specified, the dataset will loop over all <video_id>
    subfolders and include all frames in each.
    """

    def __init__(
        self,
        frames_path: Union[str, Path],
        class_map: ClassMap,
        video_ids: Optional[Union[str, List[str]]] = None,
        tfm: Optional[Transform] = None,
    ):
        self.frames_path = Path(frames_path).resolve()
        assert self.frames_path.exists(), f"Frames path not found: {self.frames_path}"
        assert self.frames_path.is_dir(), f"Frames path is not a directory: {self.frames_path}"
        self.video_ids = (
            video_ids if isinstance(video_ids, list) else [] if video_ids is None else [video_ids]
        )
        self.tfm = tfm
        self.class_map = class_map
        all_frames = FramesList().get()
        self.frames = []
        record_id: int = 0
        for video_id in self.video_ids:
            frames = [self.frames_path / fp for fp in all_frames[video_id]]
            # frames = sorted((self.frames_path / video_id).glob("*.jpg"))
            print("Num frames found: ", len(frames))
            for f in frames:
                record = BaseRecord((ImageRecordComponent(),))
                record.set_record_id(record_id)
                record.add_component(ClassMapRecordComponent(task=tasks.detection))
                if class_map is not None:
                    record.detection.set_class_map(class_map)
                parts = f.stem.replace("frame_", "").replace("s", "").split("-")
                self.frames.append(
                    {
                        "video_id": video_id,
                        "file_path": f,
                        "frame_idx": int(parts[0]),
                        "seconds": round(float(parts[1]), 2),
                        "record": record,
                    }
                )
                record_id += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        meta = self.frames[i]
        record = meta["record"]
        img = np.array(Image.open(str(meta["file_path"])))
        record.set_img(img)
        record.load()
        if self.tfm is not None:
            record = self.tfm(record)

        return record

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}, video_id:{self.video_ids}, " f"len: {len(self.frames)}>"
        )


class GeoscreensInferenceFromTaskListDataset(object):
    """
    Only usable for inference.

    Provides a dataset over tasks object, which is a list of Dict's that looks like::

        tasks = [
            {
                "id": ..., "data": {
                    "full_path": "/path/to/img.jpg", ...
                },
                ...
            }
        ]

    If no video_id specified, the dataset will loop over all <video_id> subfolders and include all
    frames in each.
    """

    def __init__(
        self,
        _tasks: list[dict],
        class_map: ClassMap,
        tfm: Optional[Transform] = None,
    ):
        self.tasks = _tasks
        self.tfm = tfm
        self.class_map = class_map
        self.frames = []
        record_id: int = 0
        for t in _tasks:
            img_path = Path(t["data"]["full_path"])
            record = BaseRecord((ImageRecordComponent(),))
            record.set_record_id(record_id)
            record.add_component(ClassMapRecordComponent(task=tasks.detection))
            if class_map is not None:
                record.detection.set_class_map(class_map)
            self.frames.append(
                {
                    "file_path": str(img_path),
                    "record": record,
                }
            )
            record_id += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        meta = self.frames[i]
        record = meta["record"]
        img = np.array(Image.open(str(meta["file_path"])))
        record.set_img(img)
        record.load()
        if self.tfm is not None:
            record = self.tfm(record)

        return record

    def __repr__(self):
        return f"<{self.__class__.__name__}, " f"len: {len(self.frames)}>"


def get_detections(
    args,
    config: DictConfig,
    module: ModuleType,
    model: nn.Module,
    geoscreens_data: LightningDataModule,
    video_id: str,
):
    """
    Returns:
        Dict: keys = frame index, value = a dict of detections that looks something like::

            {
                "frame_idx": 0,
                "seconds": 0.00,
                "time":: "00:00:00.0000",
                "label_ids": [17, 39],
                "scores": [0.5707356929779053, 0.5458141565322876],
                "bboxes": [
                    {
                        "xmin": 522.35400390625,
                        "ymin": 177.13229370117188,
                        "xmax": 640.0,
                        "ymax": 362.1326599121094,
                    },
                    {
                        "xmin": 537.4188232421875,
                        "ymin": 139.51719665527344,
                        "xmax": 635.33642578125,
                        "ymax": 157.04588317871094,
                    },
                ],
            }
    """
    infer_tfms = tfms.A.Adapter(
        [*tfms.A.resize_and_pad(config.dataset_config.img_size), tfms.A.Normalize()]
    )
    infer_ds = GeoscreensInferenceDataset(
        args.video_frames_path, geoscreens_data.parser.class_map, video_id, infer_tfms
    )
    infer_dl = module.infer_dl(infer_ds, batch_size=8, shuffle=False, num_workers=16)
    preds = module.predict_from_dl(model, infer_dl, detection_threshold=0.5)
    detections = {}
    frame_counter = 0

    for frame_info, pred in zip(infer_ds.frames, preds):
        detections[frame_counter] = {
            "frame_idx": frame_info["frame_idx"],
            "seconds": frame_info["seconds"],
            "time": datetime.utcfromtimestamp(frame_info["seconds"]).strftime("%H:%M:%S:%f"),
            "label_ids": [int(l) for l in pred.detection.label_ids],
            "scores": pred.detection.scores.tolist(),
            "bboxes": [
                {
                    "xmin": float(box.xmin),
                    "ymin": float(box.ymin),
                    "xmax": float(box.xmax),
                    "ymax": float(box.ymax),
                }
                for box in pred.detection.bboxes
            ],
        }
        frame_counter += 1

    return detections
