import datetime
import json
import uuid
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from icevision import tfms
from icevision.core.class_map import ClassMap
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything
from tqdm.contrib.bells import tqdm

from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.inference import GeoscreensInferenceFromTaskListDataset, get_model_for_inference


def get_raw_preds(
    args: Namespace,
    tasks: List[Dict],
    config: DictConfig,
    module: ModuleType,
    model: nn.Module,
    geoscreens_data: LightningDataModule,
):
    """
    Updates all tasks with a new "preds_raw" key, which has all the detector predictions (given
    confidence threshold)
    """
    infer_tfms = tfms.A.Adapter(
        [*tfms.A.resize_and_pad(config.dataset_config.img_size), tfms.A.Normalize()]
    )
    infer_ds = GeoscreensInferenceFromTaskListDataset(
        tasks, geoscreens_data.parser.class_map, infer_tfms
    )
    infer_dl = module.infer_dl(infer_ds, batch_size=8, shuffle=False, num_workers=16)
    with torch.no_grad():
        preds = module.predict_from_dl(model, infer_dl, detection_threshold=0.5)

    for i, (t, pred) in enumerate(zip(tasks, preds)):
        # if i >= 200:
        #     break
        if not (pred and hasattr(pred, "detection")):
            continue
        dets = pred.detection
        dets = {
            "label_ids": [int(l) for l in dets.label_ids],
            "scores": dets.scores.tolist(),
            "bboxes": [
                {
                    "xmin": float(box.xmin),
                    "ymin": float(box.ymin),
                    "xmax": float(box.xmax),
                    "ymax": float(box.ymax),
                }
                for box in dets.bboxes
            ],
        }
        t["preds_raw"] = dets


def get_best_pred_per_label(t, class_map: ClassMap):
    # "ignore" functionality: don't do any aggregation for certain categories. e.g., "other" Some
    # categories can appear multiple times in one UI, so we want to include those.
    ignore = {class_map.get_by_name(c) for c in ["play", "other", "challenge_btn_orange", "video"]}
    results = []
    best = {}
    for i, (bbox, score, label_id) in enumerate(
        zip(t["preds_raw"]["bboxes"], t["preds_raw"]["scores"], t["preds_raw"]["label_ids"])
    ):
        if label_id in ignore:
            results.append((bbox, score, label_id))
        else:
            if label_id not in best:
                best[label_id] = (bbox, score, label_id)
            if score > best[label_id][1]:
                best[label_id] = (bbox, score, label_id)
    results.extend(best.values())
    return results


def reverse_point(x, y, width, height, curr_dim):
    """
    Transform bbox coordinates from (curr_dim, curr_dim) pixel space to size=(width, height) pixel
    space. assumes width is greater than height. This is used because the detector bbox coordinates
    are in a square pixel space (config.dataset_config.img_size)**2, and we need to convert the bbox
    coordinates back to the original image pixel space (e.g., 1280*720).
    """
    # Back to width*width:
    new_x = x * (width / curr_dim)
    new_y = y * (width / curr_dim)
    # Remove vertical padding
    y_pad = (width - height) / 2
    new_y -= y_pad
    return new_x, new_y


def transform_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    target_width: int,
    target_height: int,
    curr_dim=640,
) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Transform bbox coordinates from (curr_dim, curr_dim) pixel space to size=(width, height) pixel
    space. assumes width is greater than height. This is used because the detector bbox coordinates
    are in a square pixel space (config.dataset_config.img_size)**2, and we need to convert the bbox
    coordinates back to the original image pixel space (e.g., 1280*720).

    TLDR: Go from detector output (640,640) -> original image dims (img_w, img_h)

    Args:
        xmin, ymin, xmax, ymax

    Returns:
        Tuple[[xmin, ymin, xmax, ymax], area]
    """
    # Back to width*width:
    new_xmin = xmin * (target_width / curr_dim)
    new_ymin = ymin * (target_width / curr_dim)
    new_xmax = xmax * (target_width / curr_dim)
    new_ymax = ymax * (target_width / curr_dim)
    # Remove vertical padding
    y_pad = (target_width - target_height) / 2
    new_ymin -= y_pad
    new_ymax -= y_pad
    new_area = (new_xmax - new_xmin + 1) * (new_ymax - new_ymin + 1)
    return (new_xmin, new_ymin, new_xmax, new_ymax), new_area


def get_bboxes(t: Dict, config: DictConfig, class_map: ClassMap) -> List[Dict]:
    """
    From t (a single task json), return list[dict], each dict containing bounding boxes. Results are
    limited to one bbox per label_id (highest confidence is used to pick the best one for each
    label_id)
    """
    if "preds_raw" not in t:
        return
    width, height = t["data"]["width"], t["data"]["height"]
    results = []
    for i, (bbox, score, label_id) in enumerate(get_best_pred_per_label(t, class_map)):
        xmin, ymin = reverse_point(
            bbox["xmin"], bbox["ymin"], width, height, config.dataset_config.img_size
        )
        xmax, ymax = reverse_point(
            bbox["xmax"], bbox["ymax"], width, height, config.dataset_config.img_size
        )
        pixel_x = (xmin * 100.0) / width
        pixel_y = (ymin * 100.0) / height
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1
        pixel_width = (box_width * 100.0) / width
        pixel_height = (box_height * 100.0) / height
        result = {
            "value": {
                "rotation": 0,
                "rectanglelabels": [class_map.get_by_id(label_id)],
                "width": pixel_width,
                "height": pixel_height,
                "x": pixel_x,
                "y": pixel_y,
                "score": score,
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            },
            "bbox": bbox,
            "data": t["data"],
        }
        results.append(result)
    return results


def compute_labelstudio_preds(args: Namespace, tasks: List[Dict]):
    config, module, model, light_model, geoscreens_data = get_model_for_inference(args)
    get_raw_preds(args, tasks, config, module, model, geoscreens_data)

    for i, t in enumerate(tqdm(tasks, total=len(tasks), desc="compute_labelstudio_preds")):
        # if i >= 100:
        #     break
        results = []
        bboxes = get_bboxes(t, config, geoscreens_data.parser.class_map)
        if not bboxes:
            t["predictions"] = [{"result": []}]
            continue
        for i, bbox in enumerate(bboxes):
            # print("")
            uid = str(uuid.uuid4()).replace("-", "")[:10]
            result = {
                "from_name": "label",
                "id": f"{uid}",
                "image_rotation": 0,
                "origin": "manual",
                "original_height": t["data"]["height"],
                "original_width": t["data"]["width"],
                "to_name": "image",
                "type": "rectanglelabels",
                "value": deepcopy(bbox["value"]),
            }
            results.append(result)
        t["predictions"] = [{"result": results}]
        t["data"]["preds_ckpt"] = str(args.checkpoint_path)
        t["data"]["preds_model"] = (
            f"{config.model_config.framework}-"
            f"{config.model_config.name}-"
            f"{config.model_config.backbone if 'backbone' in config.model_config else ''}"
        )
        t["data"]["preds_model_dataset"] = config.dataset_config.dataset_name
