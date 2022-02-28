import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from IPython.display import display
from termcolor import colored


def load_gt(filename: str = "seg_ground_truth_003.json"):
    """
    Loads ground truth for video segmentation from a json file that was exported from label-studio.
    """
    gt = {}
    anns = json.load(open(Path("/shared/gbiamby/geo/segment") / filename, "r"))
    for video in anns:
        segments, oog_segments = [], []
        video_id = (
            video["data"]["video_url"].replace("/data/local-files/?d=", "").replace(".mp4", "")
        )
        if not ("annotations" in video):
            print("no annotations key")
            continue
        for ann in video["annotations"][0]["result"]:
            segment = ann["value"]
            start_time, end_time = max(segment["start"], 0), max(0, segment["end"])
            if segment["labels"][0] == "in_game":
                segments.append((start_time, end_time))
            else:
                oog_segments.append((start_time, end_time))
        gt[video_id] = {
            "in_game_segs_count": len(segments),
            "oog_segs": oog_segments,
            "in_game_segs": segments,
        }
    return gt


def compare_to_gt(segs: List[Dict], gt: Dict) -> None:
    """
    Compares generated segments (in the DataFrame) to ground truth segments (Dict parameter), and
    stores the result in the "is_correct" column of `segs` DataFrame.
    """
    gt_oog_segs = gt["oog_segs"]
    for seg in segs:
        if seg["state"] == "out_of_game":
            matching_gt_segments = filter(
                lambda s: float(s[1]) - 2.0 < seg["end_sec"] <= float(s[1]) + 2.0,
                gt["oog_segs"],
            )
            seg["is_correct"] = len(list(matching_gt_segments)) >= 1
        elif "in_game_segs" in gt:
            matching_gt_segments = filter(
                lambda s: float(s[1]) - 2.0 < seg["end_sec"] <= float(s[1]) + 2.0,
                gt["in_game_segs"],
            )
            seg["is_correct"] = len(list(matching_gt_segments)) >= 1


def compare_to_ground_truth(segments, seg_gt: Dict[str, Any]):
    for video_id, seg in segments.items():
        print("\n\n", "=" * 120)
        print("video_id: ", video_id)
        compare_to_gt(seg, seg_gt[video_id])
        df_seg = pd.DataFrame(seg)
        display(style_df_redgreen(df_seg, "is_correct"))
        num_in_game_segments = df_seg[df_seg.state == "in_game"].shape[0]
        print(
            "num in_game: ", num_in_game_segments, ", gt: ", seg_gt[video_id]["in_game_segs_count"]
        )
        if num_in_game_segments != seg_gt[video_id]["in_game_segs_count"]:
            print(colored("in_game segments count is: WRONG!!!", color="red", on_color="on_yellow"))
        else:
            print(colored("in_game segments count is: CORRECT!!!!!", color="green"))
        if df_seg.is_correct.sum() != len(df_seg.is_correct):
            print(
                "out_of_game gt: ",
                seg_gt[video_id]["oog_segs"] if "oog_segs" in seg_gt[video_id] else "",
            )
            print(
                "in_game_gt:     ",
                seg_gt[video_id]["in_game_segs"] if "in_game_segs" in seg_gt[video_id] else "",
            )


def style_correct(s, props=""):
    return np.where(s == 1, props, "")


def style_wrong(s, props=""):
    return np.where(s == 0, props, "")


def style_df_redgreen(df: pd.DataFrame, column: str):
    return df.style.apply(
        style_correct, props="color:white;background-color:green", axis=0, subset=["is_correct"]
    ).apply(style_wrong, props="color:white;background-color:red", axis=0, subset=["is_correct"])


# Hardcoded out_of_game segments collected by Grace:
seg_gt = {
    "AF9uezxZDeE": {
        "in_game_segs_count": 5,
        "oog_segs": [(39, 141), (248, 275), (344, 352), (439, 440), (562, 650)],
    },
    "9RQUIk1OwAY": {
        "in_game_segs_count": 5,
        "oog_segs": [(185, 212), (394, 415), (597, 619), (801, 811), (994, 1022)],
    },
    "S5Ne5eoHxsY": {
        "in_game_segs_count": 5,
        "oog_segs": [(186, 212), (394, 405), (588, 608), (790, 809), (992, 1020)],
    },
    "nyHeQWnm8YA": {
        "in_game_segs_count": 5,
        "oog_segs": [(223, 233), (370, 377), (552, 554), (735, 742), (904, 929)],
    },
    "hZWt1PYH3hI": {
        "in_game_segs_count": 5,
        "oog_segs": [(117, 137), (314, 334), (471, 492), (660, 683), (866, 906)],
    },
    "dY1RXh-43q4": {
        "in_game_segs_count": 5,
        "oog_segs": [(158, 171), (287, 297), (418, 455), (577, 622), (740, 777)],
    },
    "83m9ys4kxro": {
        "in_game_segs_count": 5,
        "oog_segs": [(184, 206), (386, 398), (579, 592), (774, 806), (916, 951)],
    },
    "osTwgzWluVs": {
        "in_game_segs_count": 5,
        "oog_segs": [(147, 152), (272, 279), (360, 365), (525, 541), (671, 696)],
    },
    "o8qQAjkaXMM": {
        "in_game_segs_count": 20,
        "oog_segs": [
            (59, 67),
            (99, 107),
            (139, 147),
            (180, 195),
            (228, 262),
            (294, 299),
            (332, 346),
            (374, 377),
            (410, 425),
            (458, 482),
            (514, 528),
            (560, 566),
            (599, 607),
            (639, 647),
            (680, 708),
            (740, 757),
            (790, 793),
            (825.50, 838.50),
            (870.00, 886.75),
        ],
    },
}
