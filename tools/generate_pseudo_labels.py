"""
This script adds predictions from a given geoscreens detection model to a label-studio.

    1. Export tasks (each "task" is an image) as json from label-studio API
    2. Run detector against all tasks, add results to the json
    3. Clone original label-studio project and upload the new tasks JSON
"""
import io
import json
import shutil
import sys
import zipfile
from argparse import ArgumentParser, Namespace
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, cast

from label_studio_sdk import Client, Project
from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from PIL import Image
from pycocotools.helpers import reindex_coco_json
from requests import Response
from tqdm.contrib.bells import tqdm

from geoscreens.consts import PROJECT_ROOT
from geoscreens.data.splitting import generate_train_val_splits
from geoscreens.label_studio import Converter
from geoscreens.pseudolabels import get_preds_from_tasks_json
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


def get_labelstudio_tasks_export(
    args, project: Project, export_type: str = "JSON"
) -> Tuple[List[Dict], Path]:
    if "tasks_export_path" in args and args.tasks_export_path:
        print("Using pre-downloaded export file: ", args.tasks_export_path)
        return (
            cast(List[Dict], json.load(open(args.tasks_export_path, "r", encoding="utf-8"))),
            cast(Path, None),
        )
    else:
        print("Getting tasks export from label-studio API...")
        export = cast(List[Dict], get_labelstudio_export_from_api(project, export_type))
        export_file_path = (
            args.save_dir / f"geoscreens_{args.target_ls_version}-from_proj_id_{project.id}.json"
        )
        json.dump(
            export,
            open(export_file_path, "w"),
            ensure_ascii=False,
        )
        return (export, export_file_path)


def sync_images(args: Namespace, tasks: List[Dict]) -> List[Dict]:
    existing_images = {t["data"]["full_path"]: t for t in tasks}
    local_images = sorted(args.img_path.glob("**/*.jpg"))
    new_images = [img for img in local_images if str(img) not in existing_images]
    print("Num existing: ", len(existing_images))
    print("Num local_images: ", len(local_images))
    print("Num new_images: ", len(new_images))
    for img in new_images:
        tasks.append(
            {
                "data": {
                    "full_path": str(img),
                    "image": "http://localhost:6070" + str(img).replace(str(args.img_path), ""),
                    "video_id": img.parent.name,
                }
            }
        )
    local_images_lookup = {str(img) for img in local_images}
    final_tasks = [
        t
        for t in tasks
        if "full_path" in t["data"] and t["data"]["full_path"] in local_images_lookup
    ]
    return final_tasks


def append_image_metadata(args: Namespace, tasks: List):
    """
    Adds image dimensions to the 'data' key of the label-studio tasks json
    """
    img_path: str = str(args.img_path.resolve()) + "/"

    for t in tqdm(tasks, desc="compute_img_sizes", total=len(tasks)):
        t["data"]["full_path"] = (
            t["data"]["image"]
            .replace("/data/local-files/?d=", img_path)
            .replace("http://localhost:6070/", img_path)
        )
        t["data"]["image"] = t["data"]["image"].replace(
            "/data/local-files/?d=", "http://localhost:6070/"
        )
        if not ("width" in t["data"] and "height" in t["data"]):
            width, height = Image.open(t["data"]["full_path"]).size
            t["data"]["width"] = width
            t["data"]["height"] = height
        # Add which classes are annotated so we can query it in label-studio UI:
        annotated_classes = []
        if "annotations" in t and t["annotations"] and "result" in t["annotations"][0]:
            annotated_classes.extend(
                [ann["value"]["rectanglelabels"][0] for ann in t["annotations"][0]["result"]]
            )
            t["data"]["annotated_classes"] = ",".join(sorted(set(annotated_classes)))
        if "-lPrvqk2mqs/frame_00000086" in t["data"]["image"]:
            print("task: ", t)
            print("annotated_classes: ", sorted(set(annotated_classes)))
            # sys.exit()


def compute_preds(args: Namespace, tasks: List[Dict]) -> None:
    """
    Handle including predictions in the label pipeline. i.e., either compute new predictions, or
    convert the existing ones so they will import correctly into a new label-studio project.
    """
    if args.compute_preds:
        print("Generating pseudolabels...")
        from geoscreens.pseudolabels import compute_labelstudio_preds

        # Remove existing predictions:
        for i, t in enumerate(tasks):
            if "annotations" in t:
                for a in t["annotations"]:
                    if "prediction" in a:
                        del a["prediction"]
                    if "predictions" in a:
                        del a["predictions"]
            if "prediction" in t:
                del t["prediction"]
            if "predictions" in t:
                del t["predictions"]

        compute_labelstudio_preds(args, tasks)
        # get_preds_from_tasks_json(
        #     args,
        #     tasks,
        #     Path("/shared/gbiamby/geo/exports/geoscreens_011-from_proj_id_74_with_preds.json"),
        # )
        return
    # else:
    #     for i, t in enumerate(tasks):
    #         result = []
    #         if "annotations" in t:
    #             anns_with_preds = [
    #                 a for a in t["annotations"] if "prediction" in a and "result" in a["prediction"]
    #             ]
    #             if anns_with_preds:
    #                 result = deepcopy(anns_with_preds[0]["prediction"]["result"])
    #             for ann in t["annotations"]:
    #                 if "prediction" in ann:
    #                     del ann["prediction"]
    #         t["predictions"] = [{"result": result}]


def get_by_label_name(bboxes: List[Dict], label_name: str):
    """
    Finds the first 'bbox' dict that has the specified object category label, if such a bbox exists. Each bbox
    dict in the `bboxes` list looks smth like::

        {
            ...
            "original_width": 1280,
            "original_height": 720,
            "value": {
                "x": 42.99737453460697,
                "y": 83.2252102322048,
                "width": 15.921303939819348,
                "height": 4.51104668511282,
                "rotation": 0,
                "rectanglelabels": [
                    "try_another_map"
                ]
            },
            ...
        }
    """
    return next(
        (bbox["value"] for bbox in bboxes if bbox["value"]["rectanglelabels"][0] == label_name),
        None,
    )


def get_by_labels_name(bboxes: List[Dict], label_names: Set[str]):
    """
    Finds all 'bbox' Dicts that have any of the specified object category labels. Each bbox dict in
    the `bboxes` list looks smth like::

        {
            ...
            "original_width": 1280,
            "original_height": 720,
            "value": {
                "x": 42.99737453460697,
                "y": 83.2252102322048,
                "width": 15.921303939819348,
                "height": 4.51104668511282,
                "rotation": 0,
                "rectanglelabels": [
                    "try_another_map"
                ]
            },
            ...
        }
    """
    return [bbox["value"] for bbox in bboxes if bbox["value"]["rectanglelabels"][0] in label_names]


def do_label_fixes(tasks: List[Dict]) -> None:
    """
    Applies automatic 'fixes' to the annotations, e.g., convert in_game_mini_map & corresponding
    'guess' buttons to '_expanded' versions based on dimensions.
    """
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
        if "annotations" in t:
            fix_anns(t["annotations"])
            if t["annotations"]:
                fix_anns([t["annotations"][0]["prediction"]], debug=True)


def other_fixes(tasks: List[Dict]) -> None:
    for t in tasks:
        if "project" in t:
            del t["project"]
        if "meta" in t:
            del t["meta"]
        if "drafts" in t:
            del t["drafts"]
        if "annotations" in t:
            annotations = t["annotations"]
            for ann in annotations:
                for k in [
                    "id",
                    "ground_truth",
                    "lead_time",
                    "result_count",
                    "task",
                    "parent_prediction",
                    "parent_annotation",
                ]:
                    if k in ann:
                        del ann[k]
        if "prediction" in t:
            pred = t["prediction"]
            for k, v in pred.items():
                if k != "result":
                    del pred[k]


def debug_tasks(tasks, i):
    for t in tasks:
        if ("-lPrvqk2mqs/frame_00000086" in t["data"]["image"]) or (
            "DZ9JablpbhQ/frame_00000065" in t["data"]["image"]
        ):
            print("")
            print(i, ": ", t)


def run_label_pipeline(args):
    print("Running label pipeline")
    # Connect to the Label Studio API and check the connection
    ls = Client(url=args.ls_url, api_key=args.ls_api_key)
    ls.check_connection()
    project = ls.get_project(id=args.ls_project_id)
    tasks, _ = get_labelstudio_tasks_export(args, project)

    if hasattr(tasks, "__len__"):
        print(f"Exported {len(tasks)} tasks from label-studio")
        print("First task from export: ", tasks[:1])
        print("")

    tasks = sync_images(args, tasks)
    append_image_metadata(args, tasks)
    do_label_fixes(tasks)
    compute_preds(args, tasks)
    other_fixes(tasks)
    # sys.exit()
    # tasks = tasks[:2]
    print("label_pipeline finished")
    print("First task from completed pipeline: ", tasks[0])
    print("")
    save_path = (
        args.save_dir
        / f"geoscreens_{args.target_ls_version}-from_proj_id_{project.id}_with_preds.json"
    )
    json.dump(tasks, open(save_path, "w"), ensure_ascii=False)
    create_new_ls_project(args, ls, project, tasks)


def clone_project(args, client: Client, old_project: Project) -> Project:
    """
    Clones a specified a label-studio project.
    """
    proj_params = deepcopy(old_project.params)
    proj_params["title"] = f"geoscreens_{args.target_ls_version}"
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

    hidden_columns = {
        "explore": [
            "tasks:completed",
            "tasks:completed_at",
            "tasks:annotated_by",
            "tasks:annotations_results",
            "tasks:annotations_ids",
            "tasks:predictions_score",
            "tasks:predictions_model_versions",
            "tasks:predictions_results",
            "tasks:file_upload",
            "tasks:created_at",
        ],
        "labeling": [
            "tasks:completed",
            "tasks:completed_at",
            "tasks:annotated_by",
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
    }
    response = project.make_request(
        "POST",
        "/api/dm/views",
        json={
            "project": project.id,
            "data": {
                "filters": Filters.create(Filters.AND, []),
                "title": "Default",
                "hiddenColumns": hidden_columns,
            },
        },
    )

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
                "hiddenColumns": hidden_columns,
            },
        },
    )

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
                            Column.total_annotations,
                            Operator.GREATER,
                            Type.Number,
                            Filters.value(0),
                        )
                    ],
                ),
                "title": "Labeled",
                "hiddenColumns": hidden_columns,
            },
        },
    )
    return project


def create_new_ls_project(args, client: Client, old_project: Project, tasks: List[Dict]):
    project = clone_project(args, client, old_project)

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
    """
    Download ground truth annotations from label-studio, saving them in COCO format as a new version
    of geoscreens_{target_ds_version} dataset.
    """
    ls = Client(url=args.ls_url, api_key=args.ls_api_key)
    ls.check_connection()
    project = ls.get_project(id=args.ls_project_id)
    coco_export = cast(
        Response, get_labelstudio_export_from_api(project, "COCO", download_all_tasks="false")
    )
    print("DONE API CALL")
    print(coco_export.headers)
    coco_save_dir = PROJECT_ROOT / f"datasets/geoscreens_{args.target_ds_version}"
    coco_save_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(coco_export.content))
    z.extract("result.json", str(coco_save_dir))
    ann_path = coco_save_dir / f"geoscreens_{args.target_ds_version}.json"
    (coco_save_dir / "result.json").rename(ann_path)
    fix_anns(ann_path)


def fix_anns(ann_path: Path):
    """
    Apply some fixes to the coco annotations after they are exported from label-studio. For example,
    the image paths need to be transformed from URL's to a local file system paths.
    """
    data = json.load(open(ann_path, "r", encoding="utf-8"))

    for img in data["images"]:
        img["file_name"] = (
            img["file_name"]
            .replace("/data/local-files/?d=", "")
            .replace("http://localhost:6070/", "")
        )

    # Exclude categories
    exclude_cat_names = []
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


def ls_to_coco(args) -> None:
    """
    Download ground truth annotations from label-studio, saving them in COCO format as a new version
    of geoscreens_{target_ds_version} dataset. This is the new version, and should be used in place
    of `save_coco_anns()`. That older method only works for projects where label-studio manages
    serving the images, which we can no longer use because that doesn't allow us to add images after
    the project is set up.
    """
    # Add extra level of temp dir just in case user accidentally inputs a path with files that
    # shouldn't be deleted:
    args.save_dir = args.save_dir / "tmp"
    if args.save_dir.exists():
        shutil.rmtree(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    # json_path = Path(args.tasks_json).resolve()
    ls = Client(url=args.ls_url, api_key=args.ls_api_key)
    ls.check_connection()
    project = ls.get_project(id=args.ls_project_id)
    _, json_path = get_labelstudio_tasks_export(args, project)
    out_path = json_path.parent

    c = Converter(project.parsed_label_config, str(json_path.parent), download_resources=False)
    c.convert_to_coco(str(json_path.parent), str(out_path), output_image_dir="../datasets/images/")
    # Result is in out_path / result.json, copy it to our datasets dir:
    target_coco_path = (
        PROJECT_ROOT
        / f"datasets/geoscreens_{args.target_ds_version}"
        / f"geoscreens_{args.target_ds_version}.json"
    )
    target_coco_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path / "result.json", target_coco_path)
    fix_anns(target_coco_path)
    reindex_coco_json(target_coco_path)
    generate_train_val_splits(target_coco_path, split_by="author")


if __name__ == "__main__":
    parser = ArgumentParser()
    sp = parser.add_subparsers(dest="cmd")
    sp.required = True
    sp_get_anns = sp.add_parser("get_anns")
    sp_label_pipeline = sp.add_parser("label_pipeline")
    sp_ls_to_coco = sp.add_parser("ls_to_coco")

    def add_common_args(_sp: ArgumentParser):
        _sp.add_argument("--ls_project_id", type=int, default=84)
        _sp.add_argument("--ls_url", type=str, default="http://localhost:6008")
        _sp.add_argument(
            "--ls_api_key", type=str, default="3ac2082c83061cf1056d636a25bee65771792731"
        )
        _sp.add_argument(
            "--image_server", type=str, default="3ac2082c83061cf1056d636a25bee65771792731"
        )

    add_common_args(sp_get_anns)
    add_common_args(sp_label_pipeline)
    add_common_args(sp_ls_to_coco)
    sp_label_pipeline.add_argument(
        "--target_ls_version",
        type=str,
        default="011",
        help="Target label-studio project version.",
    )
    sp_label_pipeline.add_argument(
        "--img_path",
        type=Path,
        default=Path("/shared/gbiamby/geo/screenshots"),
        help="""
            Path to directory containing the images to be labeled. During the label pipeline, any
            new images in this path will be detected and imported into label-studio
        """,
    )
    sp_label_pipeline.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/shared/gbiamby/geo/exports"),
        help="Where to save the label-studio tasks export file.",
    )
    sp_label_pipeline.add_argument(
        "--tasks_export_path",
        type=Path,
        help=(
            """
            You'll rarely have to use this unless debugging

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
    # For predictions:
    sp_label_pipeline.add_argument(
        "--compute_preds",
        dest="compute_preds",
        action="store_true",
        help="""
        If specified, the labelling pipeline will also compute new detections using the specified
        model checkpoint and include those predictions in the new project that is pushed to
        label-studio.
        """,
    )
    sp_label_pipeline.add_argument("--device", type=str, default="0")
    sp_label_pipeline.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(
            "/shared/gbiamby/geo/models/gs_011_extra_augs--geoscreens_011-model_faster_rcnn-bb_resnest50_fpn-3f36fb97fa"
        ),
    )
    sp_label_pipeline.set_defaults(compute_preds=False)

    # Old version of exporting anns to COCO (no longer works since we switched to using an http
    # server to serve images instead of letting label-studio server them)
    sp_get_anns.add_argument(
        "--target_ds_version",
        type=str,
        default="010",
        help="Target dataset version (for generating coco formatted json).",
    )

    # Export anns as COCO Export
    sp_ls_to_coco.add_argument(
        "--target_ds_version",
        type=str,
        default="011",
        help="Target dataset version (for generating coco formatted json).",
    )
    sp_ls_to_coco.add_argument(
        "--target_ls_version",
        type=str,
        default="011",
        help="""
            This gets appended to the exported tasks json file name. It really doesn't matter what
            value is put here, since that is an intermediate output.
        """,
    )
    sp_ls_to_coco.add_argument(
        "--save_dir",
        type=Path,
        default=Path("./exports_tmp"),
        help="""
        Where to save the label-studio tasks export file. This is a temp folder just used for saving
        and processing. Contents of this folder will be deleted.
        """,
    )

    args = parser.parse_args()
    if args.cmd == "get_anns":
        save_coco_anns(args)
    elif args.cmd == "label_pipeline":
        run_label_pipeline(args)
    elif args.cmd == "ls_to_coco":
        ls_to_coco(args)
