import datetime
import uuid
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from icevision import tfms
from icevision.data import Dataset
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from tqdm.contrib import tenumerate, tmap, tzip
from tqdm.contrib.bells import tqdm, trange

from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.models import load_model_from_path
from geoscreens.utils import batchify


def get_model_for_inference(args):
    seed_everything(42, workers=True)
    DEVICE = torch.device(f"cuda:{args.device}")

    config, module, model, light_model = load_model_from_path(args.checkpoint_path, device=DEVICE)
    light_model.eval()
    geoscreens_data = GeoScreensDataModule(config, module)
    return config, module, model, light_model, geoscreens_data


def get_raw_preds(
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

    # Weird but batch_size of 1 is the fastest here. Maybe because there is no data loader with
    # threading to feed the GPU? If time, look intro the other prediction methods (as alternatives
    # to module.build_infer_batch). Maybe create a dataloader over all the images.
    batch_size = 1
    num_batches = (len(tasks) // batch_size) + (1 if (len(tasks) % batch_size) > 0 else 0)
    for i, _batch in tqdm(
        enumerate(batchify(tasks, batch_size=1)), desc="make_predictions", total=num_batches
    ):
        # if i >= 40:
        #     break
        imgs = []
        for t in _batch:
            img_path = t["data"]["full_path"]
            img = np.array(Image.open(img_path))
            imgs.append(img)
        # Predict
        infer_ds = Dataset.from_images(imgs, infer_tfms, class_map=geoscreens_data.parser.class_map)
        batch, samples = module.build_infer_batch(infer_ds)
        preds = module.predict(model, infer_ds, detection_threshold=0.5)
        for t, pred in zip(_batch, preds):
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


def get_best_pred_per_label(t):
    # TODO: Update this to not do any aggregation for certain categories. e.g., "other" categories
    # can appear multiple times in one UI, so we want to include those.
    best = {}
    for i, (bbox, score, label_id) in enumerate(
        zip(t["preds_raw"]["bboxes"], t["preds_raw"]["scores"], t["preds_raw"]["label_ids"])
    ):
        if label_id not in best:
            best[label_id] = (bbox, score, label_id)
        if score > best[label_id][1]:
            best[label_id] = (bbox, score, label_id)
    return best.values()


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


def get_bboxes(t: Dict, config: DictConfig, class_map) -> List[Dict]:
    """
    From t (a single task json), return list[dict], each dict containing bounding boxes. Results are
    limited to one bbox per label_id (highest confidence is used to pick the best one for each
    label_id)
    """
    if "preds_raw" not in t:
        return
    width, height = t["data"]["width"], t["data"]["height"]
    results = []
    for i, (bbox, score, label_id) in enumerate(get_best_pred_per_label(t)):
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


def compute_labelstudio_preds(args, tasks: List[Dict]):
    config, module, model, light_model, geoscreens_data = get_model_for_inference(args)
    get_raw_preds(tasks, config, module, model, geoscreens_data)

    for i, t in enumerate(tqdm(tasks, total=len(tasks), desc="compute_labelstudio_preds")):
        # if i >= 10:
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
