import json
import logging
import os
import sys
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd

from geoscreens.consts import DETECTIONS_PATH
from geoscreens.data import get_all_geoguessr_split_metadata, load_detections
from geoscreens.utils import save_json

from .ground_truth import compare_to_gt, load_gt, seg_gt
from .uirules import ui_to_gamestates_map


def apply_smoothing(
    df_framedets: pd.DataFrame, window_size: int = 5, direction: str = "forward"
) -> None:
    """
    Applies smoothing to the game_state column, storing the results in a new "game_states_smoothed"
    column. Smoothing is only used at points where the game_state changes values, in which case the
    new value only changes if it is the most common element in the buffer (which can be either be
    look-ahead/backwards) of nearby game_states. Preferred direction is forward.
    """
    smoothed = []
    current_state = df_framedets.loc[0]["game_state"]
    direction = direction.replace("backwards", "backward").replace("forwards", "forward")
    prev_state = current_state
    if direction == "backward":
        buffer = [df_framedets.loc[0]["game_state"]] * window_size
        for i, row in df_framedets.iterrows():
            buffer.pop(0)
            buffer.append(row.game_state)
            counter = Counter(buffer)
            current_state = counter.most_common()[0]
            smoothed.append(current_state)
    else:
        buffer = [df_framedets.loc[0]["game_state"]] * window_size
        for i, row in df_framedets.iterrows():
            buffer.pop(0)
            buffer.append(
                df_framedets.loc[min(i + window_size, df_framedets.shape[0] - 1)].game_state
            )
            if prev_state == row.game_state:
                smoothed.append((row.game_state, 0))
            else:
                counter = Counter(buffer)
                current_state = counter.most_common()[0]
                smoothed.append(current_state)
            prev_state = smoothed[-1][0]

    df_framedets["game_state_smoothed"] = [s[0] for s in smoothed]


def add_state_transition(state_transitions, row: pd.Series, from_state: str, to_state: str):
    """
    Helper method to append transition from from one state to another to a list of game state end
    points.
    """
    state_transitions.extend(
        [
            {
                "state": from_state,
                "frame_idx": row.frame_idx,
                "end_frame_idx": row.frame_idx,
                "end_sec": row.seconds,
                "end_time": row.time,
            },
            {
                "state": to_state,
                "frame_idx": row.frame_idx,
                "start_frame_idx": row.frame_idx,
                "end_frame_idx": None,
                "start_sec": row.seconds,
                "end_sec": None,
                "start_time": row.time,
                "end_time": None,
            },
        ]
    )


def get_game_state_endpoints(df_framedets: pd.DataFrame, smoothing=False) -> List[Dict[str, Any]]:
    """
    Given a DataDrame with the detections from a geoguessr video, returns list of dictionaries, each
    representing either the start or end of a contiguous section of the video. The sections tracked
    are either "in_game" or "out_of_game". Out of game can be anything such as not_in_geoguessr,
    between round, end of round -- anything that isn't the user actually in the game playing with
    the street view.
    """
    current_state = "out_of_game"
    state_transitions = []
    state_key = "game_state_smoothed" if smoothing else "game_state"
    for i, row in df_framedets.iterrows():
        if current_state != "in_game" and row[state_key] == "in_game":
            add_state_transition(state_transitions, row, "out_of_game", "in_game")
            current_state = "in_game"
        elif current_state == "in_game":
            if row[state_key] == "in_game":
                continue
            # On second thought, I don't think we want to include OOG sequences mid-game. We'd want
            # to splice them out, so have to handle this at some other level:
            #
            # elif row[state_key] == "out_of_game":
            #     continue
            else:
                add_state_transition(state_transitions, row, "in_game", "out_of_game")
                current_state = row[state_key]
        elif current_state == "end_of_game" and row[state_key] in [
            "out_of_game",
            "end_of_game",
            "unknown",
        ]:
            current_state = "end_of_game"

    return state_transitions


def endpoints_to_segments(segments: List[Dict], game_state: str = None):
    """
    Collapses list of video segment endpoints into list of states. Each state in the return value
    has information about the start and end, and duration of the segment.
    """
    i = 1
    segs = []
    while i + 1 < len(segments):
        seg = segments[i]
        seg.update(segments[i + 1])
        seg["duration_sec"] = seg["end_sec"] - seg["start_sec"]
        seg["duration_hms"] = datetime.utcfromtimestamp(seg["duration_sec"]).strftime("%H:%M:%S:%f")

        if game_state and seg["state"] == game_state:
            segs.append(seg)
        elif not game_state:
            segs.append(seg)
        i += 2
    return segs


def classify_frame(
    dets: pd.Series, ui_to_gamestates_map: OrderedDict[str, Dict[str, List[Any]]]
) -> str:
    """
    Input is a row of a pd.DataFrame. The row contains object detector output for the geoguessr UI
    elements.
    """
    label_set_base = set(dets["labels_set"])
    label_set = set(dets["labels_set"])
    if len(label_set) == 0:
        return "out_of_game"

    for state, match_types in ui_to_gamestates_map.items():
        label_set = label_set_base

        if "do_not_match" in match_types:
            if len(label_set.intersection(match_types["do_not_match"][0])) > 0:
                continue

        if "ignore" in match_types:
            label_set = label_set_base - match_types["ignore"][0]

        for ui_combo in match_types["exact"]:
            if ui_combo == label_set:
                return state

        for ui_combo in match_types["any"]:
            intersection = ui_combo.intersection(label_set)
            if len(ui_combo) == len(intersection):
                return state

    return "unknown"


def format_ui_to_gamestates_map(ui_to_gamestates_map):
    """
    _summary_
    """
    for state, match_types in ui_to_gamestates_map.items():
        for match_type, ui_element_combos in match_types.items():
            if match_type == "any" and state == "in_game":
                # expand the map / guess button combos to include all possible
                # expanded/not-expanded combinations for the in_game_map and the corresponding
                # "guess" button
                map_combos = [
                    ui_combo for ui_combo in ui_element_combos if ui_combo[0] == "in_game_mini_map"
                ]
                for ui_combo in map_combos:
                    guess_button_label = ui_combo[1]
                    ui_element_combos.append(["in_game_mini_map", f"{guess_button_label}_expanded"])
                    ui_element_combos.append(["in_game_map_expanded", guess_button_label])
                    ui_element_combos.append(
                        ["in_game_map_expanded", f"{guess_button_label}_expanded"]
                    )

            # Convert the ui element lists to sets:
            match_types[match_type] = [set(elements) for elements in ui_element_combos]


def compute_segments_qa(args, model: str):
    segments: Dict[str, List[Dict[str, Any]]] = {}
    seg_gt_new = load_gt("seg_ground_truth_009.json")
    seg_gt_new.update(seg_gt)
    print("num videos: ", len(seg_gt_new))
    val_ids = list(seg_gt_new.keys())
    val_ids_no_detection_files = {
        val_id
        for val_id in val_ids
        if not (DETECTIONS_PATH / f"{model}/val/df_frame_dets-video_id_{val_id}.csv").exists()
    }
    print("video_ids missing detection files: ", val_ids_no_detection_files)
    val_ids = list(set(val_ids) - val_ids_no_detection_files)
    print(f"Segmenting {len(val_ids)} videos...")
    for video_id in val_ids:
        print("video_id: ", video_id)
        df_framedets = load_detections(video_id, split="val", model=model)
        df_framedets["game_state"] = df_framedets.apply(
            lambda row: classify_frame(row, ui_to_gamestates_map), axis=1
        )
        apply_smoothing(df_framedets)
        end_points = get_game_state_endpoints(df_framedets, smoothing=True)
        segments[video_id] = endpoints_to_segments(end_points)
    return segments


def _segment_video(args, model: str, video_id: str, df_meta: pd.DataFrame):
    # print("video_id: ", video_id)
    if video_id not in df_meta.index:
        print(f"SKIP {video_id} - unknown split")
        return {"video_id": video_id, "result": True, "msg": "Skipped, no metadata found."}
    split = df_meta.loc[video_id].split
    csv_path = Path(args.save_dir / split / f"df_seg-video_id_{video_id}.csv")
    if csv_path.exists() and not args.force:
        print("SKIP segment, csv_path exists: ", csv_path)
        return {"video_id": video_id, "result": True, "msg": "Skipped, file exists."}
    csv_path.parent.mkdir(exist_ok=True, parents=True)

    # Compute segments
    df_framedets = load_detections(video_id, split=split, model=model)
    df_framedets["game_state"] = df_framedets.apply(
        lambda row: classify_frame(row, ui_to_gamestates_map), axis=1
    )
    apply_smoothing(df_framedets)
    end_points = get_game_state_endpoints(df_framedets, smoothing=True)
    segments = endpoints_to_segments(end_points)

    df_seg = pd.DataFrame(segments)

    # Basic (but not perfect) error check:
    if df_seg is None or df_seg.shape[0] == 0:
        return {
            "video_id": video_id,
            "result": False,
            "msg": "No rounds found.",
            "details": (
                f"(df_seg is None: {df_seg is None}, "
                f"df_seg.shape: {df_seg.shape if df_seg is not None else 'None'}"
            ),
        }
    if len(df_seg[df_seg.state == "in_game"]) % 5 != 0:
        return {
            "video_id": video_id,
            "result": False,
            "msg": "MOD 5 violation.",
            "details": f"(got {len(df_seg[df_seg.state == 'in_game']) } rounds)",
        }

    # Save output
    print(f"Saving output: {csv_path}")
    df_seg.to_csv(csv_path, header=True, index=False)
    df_seg.to_pickle(str(csv_path.with_suffix(".pkl")))
    return {"video_id": video_id, "result": True}


def segment_video(args, model: str, video_id: str, df_meta: pd.DataFrame):
    try:
        return _segment_video(args, model, video_id, df_meta)
    except Exception as ex:
        print(f"Error processing video_id {video_id}")
        print(ex)
        return {"video_id": video_id, "result": False, "msg": str(type(ex)), "details": str(ex)}


def compute_segments(args, model: str, multi_threaded: bool = False):
    """
    Example to inspect results after running::

        segs = load_json("/home/gbiamby/proj/geoscreens/tools/seg_log-20220304-211620.json")["results"]
        df_segs = pd.DataFrame([s for s in segs if s is not None])
        df_segs[~df_segs.result]
        df_segs.groupby("msg").agg(total=("video_id", "count"))
    """
    df_meta = (
        pd.DataFrame(get_all_geoguessr_split_metadata().values())
        .rename(columns={"id": "video_id"})
        .set_index("video_id")
    )[["split"]].copy(deep=True)

    # segments: Dict[str, List[Dict[str, Any]]] = {}
    dets_path = (Path(args.dets_path) / model).resolve()
    dets_files = sorted(dets_path.glob("**/df_frame_dets*.pkl"))
    video_ids = [d.stem.replace("df_frame_dets-video_id_", "") for d in dets_files]
    print(f"Segmenting {len(video_ids)} videos...")

    results = []
    if not multi_threaded:
        for video_id in video_ids:
            results.append(segment_video(args, model, video_id, df_meta))
    else:
        num_workers = 50
        _args = ((args, model, video_id, df_meta) for i, video_id in enumerate(video_ids))

        with Pool(processes=num_workers) as pool:
            results = pool.starmap(segment_video, _args)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_json(f"./seg_log-{timestr}.json", {"results": results})
