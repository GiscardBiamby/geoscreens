"""
This script adds predictions from a given geoscreens detection model to a label-studio.

    1. Export tasks (each "task" is an image) as json from label-studio API
    2. Run detector against all tasks, add results to the json
    3. Clone original label-studio project and upload the new tasks JSON
"""
import io
import json
import sys
import zipfile
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, cast

import numpy as np
from label_studio_sdk import Client, Project
from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from PIL import Image
from requests import Response
from tqdm.contrib import tenumerate, tmap, tzip
from tqdm.contrib.bells import tqdm, trange

from geoscreens.consts import PROJECT_ROOT
from geoscreens.utils import batchify


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
    if args.tasks_export_path:
        print("Using pre-downloaded export file: ", args.tasks_export_path)
        return json.load(open(args.tasks_export_path, "r", encoding="utf-8"))
    else:
        print("Getting tasks export from label-studio API...")
        export = cast(List[Dict], get_labelstudio_export_from_api(project, export_type))
        json.dump(
            export,
            open(
                args.save_dir / f"geoscreens_{args.target_version}-from_proj_id_{project.id}.json",
                "w",
            ),
            ensure_ascii=False,
        )
        return export


def append_image_metadata(tasks: List):
    """
    Adds image dimensions to the 'data' key of the label-studio tasks json
    """
    for t in tqdm(tasks, desc="compute_img_sizes", total=len(tasks)):
        t["data"]["full_path"] = t["data"]["image"].replace(
            "/data/local-files/?d=", "/shared/gbiamby/geo/screenshots/"
        )
        if not ("width" in t["data"] and "height" in t["data"]):
            width, height = Image.open(t["data"]["full_path"]).size
            t["data"]["width"] = width
            t["data"]["height"] = height


def append_old_preds(args: Namespace, tasks: List[Dict]) -> None:
    """
    Handle including predictions in the label pipeline. i.e., either compute new predictions, or
    convert the existing ones so they will import correctly into a new label-studio project.
    """
    if args.compute_preds:
        print("Generating pseudolabels...")
        from geoscreens.pseudolabels import compute_labelstudio_preds

        # TODO: Do we need to update this function to remove the "predictions" key from
        # t["annotations"]?
        compute_labelstudio_preds(args, tasks)
        return

    for i, t in enumerate(tasks):
        result = []
        if "annotations" in t:
            anns_with_preds = [
                a for a in t["annotations"] if "prediction" in a and "result" in a["prediction"]
            ]
            if anns_with_preds:
                result = anns_with_preds[0]["prediction"]["result"]
                # Remove the prediction from the annotations dict:
                t["annotations"] = [a for a in t["annotations"] if "prediction" not in a]

        t["predictions"] = [{"result": result}]


def get_by_label_name(results: List[Dict], label_name: str):
    return next(
        (
            result["value"]
            for result in results
            if result["value"]["rectanglelabels"][0] == label_name
        ),
        None,
    )


def get_by_labels_name(results: List[Dict], label_names: Set[str]):
    return [
        result["value"]
        for result in results
        if result["value"]["rectanglelabels"][0] in label_names
    ]


def do_label_fixes(tasks: List[Dict]) -> None:
    guess_btn_labels = set(
        [
            "guess",
            "guess_grey",
            "make_a_guess",
            "guess_w_icon_only",
            "place_your_pin_grey",
        ]
    )

    def fix_anns(anns: List[Dict], debug=False):
        for a in anns:
            if "result" not in a:
                continue
            mini_map_bbox = get_by_label_name(a["result"], "in_game_mini_map")
            if mini_map_bbox:
                if mini_map_bbox["rectanglelabels"][0] == "in_game_mini_map":
                    bbox_pct_of_total_img_area = (
                        100.0 * mini_map_bbox["width"] * mini_map_bbox["height"] / (100.0 ** 2)
                    )
                    if bbox_pct_of_total_img_area > 20.0:
                        mini_map_bbox["rectanglelabels"][0] = "in_game_map_expanded"
                        # Convert the associated "Guess" buttons:
                        for btn_bbox in get_by_labels_name(a["result"], guess_btn_labels):
                            if 0.85 <= (btn_bbox["width"] / mini_map_bbox["width"]) <= 1.15:
                                btn_bbox["rectanglelabels"][0] = (
                                    btn_bbox["rectanglelabels"][0] + "_expanded"
                                )

    for t in tasks:
        fix_anns(t["annotations"])
        if t["annotations"]:
            fix_anns([t["annotations"][0]["prediction"]], debug=True)


def other_fixes(tasks: List[Dict]) -> None:
    for t in tasks:
        if "project" in t:
            del t["project"]
        if "meta" in t:
            del t["meta"]


def debug_tasks(tasks):
    for t in tasks:
        if "Qm3FPspE6Nw/frame_00000121.jpg" in t["data"]["full_path"]:
            print("")
            print(t)


def run_label_pipeline(args):
    print("Running label pipeline")
    # Connect to the Label Studio API and check the connection
    ls = Client(url=args.ls_url, api_key=args.ls_api_key)
    ls.check_connection()
    project = ls.get_project(id=args.ls_project_id)
    print("Project views: ", project.get_views())
    tasks = get_labelstudio_tasks_export(args, project)

    if hasattr(tasks, "__len__"):
        print(f"Exported {len(tasks)} tasks from label-studio")
        print("First task from export: ", tasks[:1])
        print("")

    append_image_metadata(tasks)
    do_label_fixes(tasks)
    append_old_preds(args, tasks)
    other_fixes(tasks)
    # tasks = tasks[:2]
    print("label_pipeline finished")
    print("First task from completed pipeline: ", tasks[0])
    print("")
    save_path = (
        args.save_dir
        / f"geoscreens_{args.target_version}-from_proj_id_{project.id}_with_preds.json"
    )
    json.dump(tasks, open(save_path, "w"), ensure_ascii=False)
    create_new_ls_project(args, ls, project, tasks)


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
    print("Cloning old project_id: ", old_project.id)
    # print("Old project param keys: ", len(proj_params.keys()))
    print("New project params: ", proj_params)
    print("")
    project = client.start_project(**proj_params)
    print("New label-studio project_id: ", project.id)

    response = project.make_request(
        "POST",
        "/api/dm/views",
        json={
            "project": project.id,
            "data": {
                "filters": Filters.create(Filters.AND, []),
                "title": "Default",
                "hiddenColumns": {
                    "explore": [
                        "tasks:completed_at",
                        "tasks:annotations_results",
                        "tasks:annotations_ids",
                        "tasks:predictions_score",
                        "tasks:predictions_model_versions",
                        "tasks:predictions_results",
                        "tasks:file_upload",
                        "tasks:created_at",
                    ],
                    "labeling": [
                        "tasks:completed_at",
                        "tasks:cancelled_annotations",
                        "tasks:total_predictions",
                        "tasks:annotators",
                        "tasks:annotations_results",
                        "tasks:annotations_ids",
                        "tasks:predictions_score",
                        "tasks:predictions_model_versions",
                        "tasks:predictions_results",
                        "tasks:file_upload",
                        "tasks:created_at",
                    ],
                },
            },
        },
    )
    all_view = response.json()
    response = project.make_request(
        "POST",
        "/api/dm/views",
        json={
            "project": project.id,
            "data": {
                "filters": Filters.create(
                    Filters.AND,
                    [
                        Filters.item(
                            Column.total_annotations, Operator.EQUAL, Type.Number, Filters.value(0)
                        )
                    ],
                ),
                "title": "Unlabeled",
                "hiddenColumns": {
                    "explore": [
                        "tasks:completed_at",
                        "tasks:annotations_results",
                        "tasks:annotations_ids",
                        "tasks:predictions_score",
                        "tasks:predictions_model_versions",
                        "tasks:predictions_results",
                        "tasks:file_upload",
                        "tasks:created_at",
                    ],
                    "labeling": [
                        "tasks:completed_at",
                        "tasks:cancelled_annotations",
                        "tasks:total_predictions",
                        "tasks:annotators",
                        "tasks:annotations_results",
                        "tasks:annotations_ids",
                        "tasks:predictions_score",
                        "tasks:predictions_model_versions",
                        "tasks:predictions_results",
                        "tasks:file_upload",
                        "tasks:created_at",
                    ],
                },
            },
        },
    )
    unlabeled_view = response.json()
    return project


def create_new_ls_project(args, client: Client, old_project: Project, tasks: List[Dict]):
    project = clone_project(args, client, old_project, tasks)

    print("")
    print("Importing tasks to label-studio...")
    # print(tasks[0])
    task_ids = []
    batch_size = 500
    for _batch in tqdm(
        batchify(tasks, batch_size), desc="import_tasks", total=(len(tasks) // batch_size)
    ):
        task_ids.extend(project.import_tasks(list(_batch)))
        # break
    print(f"Created project_id: {project.id}, with {len(task_ids)} tasks")


def save_coco_anns(args):
    ls = Client(url=args.ls_url, api_key=args.ls_api_key)
    ls.check_connection()
    project = ls.get_project(id=args.ls_project_id)
    coco_export = cast(
        Response, get_labelstudio_export_from_api(project, "COCO", download_all_tasks="false")
    )
    print("DONE API CALL")
    print(coco_export.headers)
    coco_save_dir = PROJECT_ROOT / f"datasets/geoscreens_{args.target_version}"
    coco_save_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(coco_export.content))
    z.extract("result.json", str(coco_save_dir))
    ann_path = coco_save_dir / f"geoscreens_{args.target_version}.json"
    (coco_save_dir / "result.json").rename(ann_path)
    fix_anns(ann_path)


def fix_anns(ann_path: Path):
    """
    Apply some fixes to the coco annotations after they are exported from label-studio. For example,
    the image path needs to be transformed from an URL to a path we can use in a dataloader.
    """
    data = json.load(open(ann_path, "r", encoding="utf-8"))
    data["categories"][0]["name"] = "background"

    for img in data["images"]:
        img["file_name"] = img["file_name"].replace("/data/local-files/?d=screen_samples_auto/", "")

    # Exclude categories
    exclude_cat_names = [
        "game_finished_white_box",
        "game_finished_well_done_big_box",
        "video",
        "curr_state",
    ]
    exclude_cat_ids = {
        c["id"]: c["name"] for c in data["categories"] if c["name"] in exclude_cat_names
    }
    anns = cast(list, data["annotations"])
    for i, ann in enumerate(anns):
        if ann["category_id"] in exclude_cat_ids:
            anns.pop(i)
    original_cat_length = len(data["categories"])
    data["categories"] = [c for c in data["categories"] if c["id"] not in exclude_cat_ids]
    filtered_cat_length = len(data["categories"])
    print(
        f"Filtered categories. Origina len: {original_cat_length}, New len: {filtered_cat_length}"
    )
    # icevision is weird and instead of remapping the category_id's it fills them in with `None``,
    # so we have to take the max of the cat_id instead of the cardinality:
    print(f"Update num_classes to: {max([ c['id'] for c in data['categories']])}")
    json.dump(data, open(Path(ann_path), "w"), indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    sp = parser.add_subparsers(dest="cmd")
    sp.required = True
    sp_get_anns = sp.add_parser("get_anns")
    sp_label_pipeline = sp.add_parser("label_pipeline")

    def add_common_args(_sp: ArgumentParser):
        _sp.add_argument(
            "--target_version",
            type=str,
            default="007",
            help="Target dataset version.",
        )
        _sp.add_argument("--ls_project_id", type=int, default=58)
        _sp.add_argument("--ls_url", type=str, default="http://localhost:6008")
        _sp.add_argument(
            "--ls_api_key", type=str, default="3ac2082c83061cf1056d636a25bee65771792731"
        )

    add_common_args(sp_get_anns)
    add_common_args(sp_label_pipeline)
    sp_label_pipeline.add_argument("--device", type=str, default="0")
    sp_label_pipeline.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/shared/gbiamby/geo/exports"),
        help="Where to save the label-studio tasks export file.",
    )
    sp_label_pipeline.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("/shared/gbiamby/geo/models/geoscreens_009-resnest50_fpn-with_augs"),
    )
    sp_label_pipeline.add_argument(
        "--compute_preds",
        dest="compute_preds",
        action="store_true",
        help="""
        If specified, the labelling pipeline will also compute new detections using the specified
        model checkpoint and include those predictions in the new project that is pushed to
        label-studio.
        """
    )
    sp_label_pipeline.add_argument(
        "--tasks_export_path",
        type=Path,
        help=(
            """
            Path to tasks export json that has already been exported. If specified the script will
            use this file and bypass the step of exporting the tasks json from the label-studio API.
            Ex: /shared/gbiamby/geo/exports/geoscreens_006-from_proj_id_5.json

            If you have a tasks export file that has detection output and you want to import those
            to label-studio you can pass that here too. The use case would be that something went
            wrong when trying to upload the script before, you corrected the issue and want to just
            upload those detections to label-studio without re-running the detector.
            """
        ),
    )
    sp_label_pipeline.set_defaults(compute_preds=False)
    args = parser.parse_args()
    if args.cmd == "get_anns":
        save_coco_anns(args)
    elif args.cmd == "label_pipeline":
        run_label_pipeline(args)
