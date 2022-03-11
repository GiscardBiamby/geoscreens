"""
This script runs a geoscreens detector against videos from a given location and saves the
results.The script supports parallel runs in a very basic way -- by using the 1--device1 and
1--num_devices` parameters a script launched with device=N will only process videos where the video
index MOD num_devices == device_id. So you can parallelize the work by launching 10 script
instances, each with --num_devices=10, and --device=[some value between 0 and 9].

The output from this script can be used for many things:

    1. Segment videos into contiguous chunks of "in game" state.
    2. Mask out UI elements.
    3. Categorize the game types (drone, stadium, time challenge, darts, etc) by inspecting which UI
       elements are detected.
"""
import json
import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm.contrib.bells import tqdm

from geoscreens.consts import (
    DETECTIONS_PATH,
    EXTRACTED_FRAMES_PATH,
    LATEST_DETECTION_MODEL_NAME,
    LATEST_DETECTION_MODEL_PATH,
)
from geoscreens.data.metadata import GOOGLE_SHEET_IDS, get_geoguessr_split_metadata
from geoscreens.inference import get_detections, get_model_for_inference


def make_dets_df(args, cats: Dict[int, str], frame_detections: Dict) -> pd.DataFrame:
    df_framedets = pd.DataFrame(
        [
            {
                "frame_id": k,
                "frame_idx": v["frame_idx"],
                "seconds": v["seconds"],
                "time": datetime.utcfromtimestamp(v["seconds"]).strftime("%H:%M:%S:%f"),
                "label_ids": v["label_ids"],
                "labels": [cats[l] for l in v["label_ids"]],
                "labels_set": tuple(set(cats[l] for l in v["label_ids"])),
                "scores": v["scores"],
                "bboxes": v["bboxes"],
            }
            for k, v in frame_detections.items()
        ]
    )
    if df_framedets is not None and len(df_framedets) > 0:
        df_framedets["label_set_count"] = df_framedets.merge(
            pd.DataFrame(df_framedets.groupby(["labels_set"]).agg(cnt=("frame_id", "count"))),
            left_on="labels_set",
            right_on="labels_set",
        )["cnt"]

    return df_framedets


def get_video_list(args, split: str):
    id_list = deepcopy(GOOGLE_SHEET_IDS)
    # id_list = set(
    #     [
    #         "AF9uezxZDeE",
    #         "9RQUIk1OwAY",
    #         "S5Ne5eoHxsY",
    #         "nyHeQWnm8YA",
    #         "hZWt1PYH3hI",
    #         "dY1RXh-43q4",
    #         "83m9ys4kxro",
    #         "osTwgzWluVs",
    #         "o8qQAjkaXMM",
    #     ]
    # )
    (args.save_dir / split).mkdir(parents=True, exist_ok=True)
    print("SAVE_DIR: ", args.save_dir)
    meta_data = get_geoguessr_split_metadata(split)
    # # DEBUG / HACK: Force inclusion of videos that are no longer in validation set
    if split == "val":
        meta_data.append(
            {
                "id": vid
                for vid in [
                    "osTwgzWluVs",
                    "hZWt1PYH3hI",
                    "83m9ys4kxro",
                    "9RQUIk1OwAY",
                    "dY1RXh-43q4",
                    "S5Ne5eoHxsY",
                ]
            }
        )
    # # /DEBUG / HACK

    if args.video_id:
        id_list = [args.video_id]
        meta_data = [{"id": args.video_id}]

    # meta_data = [s for i, s in enumerate(meta_data) if s["id"] in id_list]

    remove_list = []
    # Only process videos that have frames extracted:
    frames_extracted = set(
        [
            str(p.stem.replace("df_frame_dets-video_id_", ""))
            for p in sorted(EXTRACTED_FRAMES_PATH.glob("*/"))
        ]
    )
    remove_list.extend([m["id"] for m in meta_data if m["id"] not in frames_extracted])

    # Ignore videos that already have detection results saved:
    for m in meta_data:
        csv_path = Path(args.save_dir / split / f"df_frame_dets-video_id_{m['id']}.csv")
        if csv_path.exists():
            remove_list.append(m["id"])

    meta_data = [m for m in meta_data if m["id"] not in set(remove_list)]
    # meta_data = meta_filtered[:1000]

    print("Total video count (before splitting across processes): ", len(meta_data))
    meta_data = [s for i, s in enumerate(meta_data) if (i % args.num_devices == args.device)]
    print(f"Processing {len(meta_data)} videos (after 'MOD {args.num_devices}' logic applied).")
    return meta_data


def generate_detections(args, split: str):
    meta_data = get_video_list(args, split)
    config, module, model, light_model, geoscreens_data = get_model_for_inference(args)

    for video_info in tqdm(
        meta_data, total=len(meta_data), desc=f"generate_detections_{split}_vids"
    ):
        video_id = video_info["id"]
        print("")
        print(f"Detecting UI elems for video_id: {video_id}, split: {split}.")
        csv_path = Path(args.save_dir / split / f"df_frame_dets-video_id_{video_id}.csv")
        if csv_path.exists() and not args.video_id:
            print("SKIP detection, csv_path exists: ", csv_path)
            continue
        try:
            with torch.no_grad():
                frame_detections = get_detections(
                    args,
                    config,
                    module,
                    model,
                    geoscreens_data,
                    video_id,
                )
            df_frame_dets = make_dets_df(args, geoscreens_data.id_to_class, frame_detections)
            print(f"Saving output: {csv_path}")
            df_frame_dets.to_csv(csv_path, header=True, index=False)
            df_frame_dets.to_pickle(str(csv_path.with_suffix(".pkl")))
        except Exception as ex:
            print(ex)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--video_id", type=str, default="0")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=DETECTIONS_PATH,
        help="Where to save the detection outputs.",
    )
    parser.add_argument(
        "--video_frames_path",
        type=Path,
        default=EXTRACTED_FRAMES_PATH,
        help="""Path to directory containing extracted video frames.""",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=LATEST_DETECTION_MODEL_PATH,
        help="Path of model checkpoint to use for predictions.",
    )
    parser.add_argument(
        "--frame_sample_rate",
        type=float,
        default=4.0,
        help="Num frames per second to sample.",
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=10,
        help="Used to split the work across multiple processes. "
        "This process will only run on videos where: video_idx MOD num_devices == device",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=1000,
        help="Max number of mvideos to process for a given geoguessr train/val/test split",
    )
    parser.add_argument(
        "--fast_debug",
        dest="fast_debug",
        action="store_true",
        help="Only perform detection for a small number of frames in the video.",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        help="",
    )

    args = parser.parse_args()
    args.save_dir = args.save_dir / args.checkpoint_path.name
    for split in ["val", "train", "test"]:
        print("")
        print("")
        print("=" * 100)
        print("=" * 100)
        print("Processisng split: ", split)
        generate_detections(args, split)

    # Hardcoded loop to generate detections with multiple models (bypasses the checkpoint path command line arg):
    #
    # save_dir_base = Path(args.save_dir)
    # checkpoints = [
    #     # "/home/gbiamby/proj/geoscreens/tools/output/geoscreens_010-model_faster_rcnn-bb_resnest50_fpn-lr_0.00055/20",
    #     # "/home/gbiamby/proj/geoscreens/tools/output/geoscreens_010-model_faster_rcnn-bb_resnest50_fpn-lr_0.00055/24",
    #     # "/home/gbiamby/proj/geoscreens/tools/output/geoscreens_010-model_faster_rcnn-bb_resnest50_fpn-lr_0.00055/26",
    #     # "/home/gbiamby/proj/geoscreens/tools/output/gs_010_with_augs--geoscreens_010-model_faster_rcnn-bb_resnest50_fpn-8b23604566",
    #     "/home/gbiamby/proj/geoscreens/tools/output/gs_010_extra_augs--geoscreens_010-model_faster_rcnn-bb_resnest50_fpn-024f52f6dd",
    # ]
    # for checkpoint in checkpoints:
    #     args.checkpoint_path = Path(checkpoint)
    #     print("Checkpoint: ", args.checkpoint_path)
    #     args.save_dir = save_dir_base / args.checkpoint_path.parent.name / args.checkpoint_path.name
    #     for split in ["train", "val", "test"]:
    #         print("")
    #         print("")
    #         print("=" * 100)
    #         print("=" * 100)
    #         print("Processisng split: ", split)
    #         generate_detections(args, split)
