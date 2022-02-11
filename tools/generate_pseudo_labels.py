"""
This script adds predictions from a given geoscreens detection modoel to a label-studio tasks file.
"""
import datetime
import io
import json
import os
import sys
import uuid
import zipfile
from argparse import ArgumentParser
from copy import deepcopy
from datetime import timezone
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Union, cast

import numpy as np
import PIL
import torch
import torch.nn as nn
from icevision import models, tfms
from icevision.data import Dataset
from icevision.metrics import COCOMetric, COCOMetricType, Metric
from label_studio_sdk import Client, Project
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from requests import Response
from tqdm.contrib import tenumerate, tmap, tzip
from tqdm.contrib.bells import tqdm, trange

from geoscreens.consts import PROJECT_ROOT
from geoscreens.geo_data import GeoScreensDataModule
from geoscreens.models import load_model_from_path


def get_model(args):
    seed_everything(42, workers=True)
    DEVICE = torch.device(f"cuda:{args.device}")

    config, module, model, light_model = load_model_from_path(args.checkpoint_path, device=DEVICE)
    light_model.eval()
    geoscreens_data = GeoScreensDataModule(config, module)
    return config, module, model, light_model, geoscreens_data


def get_labelstudio_export_from_api(
    project: Project, export_type: str, download_all_tasks: str = "true"
) -> Union[List[Dict], Response]:
    response = project.make_request(
        method="GET",
        url=f"/api/projects/{project.id}/export?exportType={export_type}&download_all_tasks={download_all_tasks}",
        timeout=500,
    )
    if response.headers["Content-Type"] == "application/zip":
        return response
    else:
        export = response.json()
        return export


def get_labelstudio_tasks_export(args, project: Project, export_type: str = "JSON") -> List[Dict]:
    if args.export_path:
        print("Using pre-downloaded export file: ", args.export_path)
        return json.load(open(args.export_path, "r"))
    else:
        export = cast(List[Dict], get_labelstudio_export_from_api(project, export_type))
        json.dump(
            export,
            open(
                args.save_dir / f"geoscreens_{args.target_version}-proj_id_{project.id}.json", "w"
            ),
        )
        return export


def batchify(iterable, batch_size=1):
    """Splits an iterable / list-like into batches of size n"""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def compute_image_sizes(tasks: List):
    for t in tqdm(tasks, desc="compute_img_sizes", total=len(tasks)):
        if "full_path" in t["data"]:
            t["data"]["full_path"] = t["data"]["image"].replace(
                "/data/local-files/?d=", "/shared/gbiamby/geo/screenshots/"
            )
        if not ("width" in t["data"] and "height" in t["data"]):
            width, height = PIL.Image.open(t["data"]["full_path"]).size
            t["data"]["width"] = width
            t["data"]["height"] = height


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
            img_path = t["data"]["image"]
            new_file_name = img_path.replace(
                "/data/local-files/?d=", "/shared/gbiamby/geo/screenshots/"
            )
            img_path = Path(new_file_name)
            img = np.array(PIL.Image.open(img_path))
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
            # print("preds: ", preds[0].pred.detection)
    print(t)


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


def compute_labelstudio_preds(tasks, config, geoscreens_data):
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


def generate_pseudo_labels(args):
    # Connect to the Label Studio API and check the connection
    ls = Client(url=args.ls_url, api_key=args.ls_api_key)
    ls.check_connection()
    project = ls.get_project(id=args.ls_project_id)
    tasks = get_labelstudio_tasks_export(args, project)

    if hasattr(tasks, "__len__"):
        print(tasks[:1])
        print(f"Exported {len(tasks)} tasks from label-studio")

    config, module, model, light_model, geoscreens_data = get_model(args)
    get_raw_preds(tasks, config, module, model, geoscreens_data)
    compute_image_sizes(tasks)
    compute_labelstudio_preds(tasks, config, geoscreens_data)
    print(tasks[0])
    json.dump(
        tasks,
        open(
            args.save_dir
            / f"geoscreens_{args.target_version}-proj_id_{project.id}_with_preds.json",
            "w",
        ),
    )
    create_new_ls_project(args, config, ls, project, tasks)


def clone_project(args, client: Client, old_project: Project, tasks: List[Dict]):
    proj_params = deepcopy(old_project.params)
    proj_params["title"] = f"geoscreens_{args.target_version}"
    remove_keys = set(
        [
            "id",
            "created_at",
            "model_version",
            "organization",
            "overlap_cohort_percentage",
            "num_tasks_with_annotations",
            "task_number",
            "start_training_on_annotation_update",
            "total_annotations_number",
            "num_tasks_with_annotations",
            "task_number",
            "useful_annotation_number",
            "ground_truth_number",
            "skipped_annotations_number",
            "total_annotations_number",
            "total_predictions_number",
            "ground_truth_number",
            "useful_annotation_number",
            "parsed_label_config",
            "config_has_control_tags",
            "control_weights",
            "created_by",
        ]
    )
    proj_params = {k: v for k, v in proj_params.items() if k not in remove_keys}
    print(len(proj_params.keys()))
    print(proj_params)
    project = client.start_project(**proj_params)
    print(project.id)
    return project


def create_new_ls_project(
    args, config: DictConfig, client: Client, old_project: Project, tasks: List[Dict]
):
    project = clone_project(args, client, old_project, tasks)
    print("")
    print("Importing: ")
    print(tasks[0])
    task_ids = []
    batch_size = 500
    for _batch in tqdm(
        batchify(tasks, batch_size), desc="import_tasks", total=(len(tasks) // batch_size)
    ):
        print(type(_batch))
        task_ids.extend(project.import_tasks(list(_batch)))

    print(f"Created project_id: {project.id}, with {len(task_ids)} tasks")
    # project.make_request()
    # project.create_predictions()

    # project.import_tasks()
    # project.create_prediction(task_ids[0], result="Dog", score=0.9)


def save_coco_anns(target_version: str, project: Project):
    coco_export = cast(
        Response, get_labelstudio_export_from_api(project, "COCO", download_all_tasks="false")
    )
    print("DONE API CALL")
    print(coco_export.headers)
    coco_save_dir = PROJECT_ROOT / f"datasets/geoscreens_{target_version}"
    coco_save_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(coco_export.content))
    z.extract("result.json", str(coco_save_dir))
    (coco_save_dir / "result.json").rename(coco_save_dir / f"geoscreens_{target_version}.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/shared/gbiamby/geo/exports"),
        help="Where to save the label-studio tasks export file.",
    )
    # TODO: Should comment out the default value, just have it here to make debugging easier:
    parser.add_argument(
        "--export_path",
        type=Path,
        default=Path("/shared/gbiamby/geo/exports/geoscreens_006-proj_id_5.json"),
    )
    parser.add_argument("--target_version", type=str, default="006", help="Target dataset version.")
    parser.add_argument("--ls_project_id", type=int, default=5)
    parser.add_argument("--ls_url", type=str, default="http://localhost:6008")
    parser.add_argument(
        "--ls_api_key", type=str, default="3ac2082c83061cf1056d636a25bee65771792731"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("/shared/gbiamby/geo/models/best_ap_at_iou0.50"),
    )
    args = parser.parse_args()
    generate_pseudo_labels(args)
