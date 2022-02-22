import json
import logging
import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from IPython.display import Image, display
from termcolor import colored
from tqdm.contrib.bells import tqdm

from geoscreens.video_seg.ground_truth import compare_to_gt, load_gt, seg_gt


def parse_tuple(s: Union[str, tuple]) -> tuple:
    """Helper for load_detections_csv, to parse string column into column of Tuples."""
    if isinstance(s, str):
        result = s.replace("(", "[").replace(")", "]")
        result = result.replace("'", '"').strip()
        result = result.replace(",]", "]")
        if result:
            # print(result)
            return tuple(sorted((json.loads(result))))
        else:
            return tuple()
    else:
        return s


def parse_dict(s: str):
    """Helper for load_detections_csv, to parse string column into Dict."""
    if isinstance(s, str):
        return json.loads(s.replace("'", '"'))
    return s


def load_detections_csv(
    video_id: str, split: str = "val", model: str = "geoscreens_009-resnest50_fpn-with_augs"
) -> pd.DataFrame:
    csv_path = Path(
        f"/shared/gbiamby/geo/segment/detections/{model}/{split}/df_frame_dets-video_id_{video_id}.csv",
    )
    df = pd.read_csv(csv_path)
    df.frame_id = df.frame_id.astype(int)
    df.label_ids = df.label_ids.apply(lambda x: parse_dict(x))
    df.labels = df.labels.apply(lambda x: parse_dict(x))
    df.labels_set = df.labels_set.apply(lambda x: parse_tuple(x))
    df.scores = df.scores.apply(lambda x: parse_dict(x))
    df.bboxes = df.bboxes.apply(lambda x: parse_dict(x))

    return df


def load_detections(
    video_id: str,
    split: str = "val",
    model: str = "geoscreens_009-resnest50_fpn-with_augs",
    frame_sample_rate: float = 4.0,
) -> pd.DataFrame:
    """
    NOTE: This assumes the detections are using frame sample rate of 4.0 fps. Specify
    frame_sample_rate if you're using a different setting.
    """
    dets_path = Path(
        f"/shared/gbiamby/geo/segment/detections/{model}/{split}/df_frame_dets-video_id_{video_id}.csv",
    )
    if dets_path.suffix == ".csv":
        df = load_detections_csv(video_id, split=split, model=model)
    else:
        df = pickle.load(open(dets_path, "rb"))

    if "frame_time" not in df.columns:
        df["frame_time"] = df.apply(lambda x: f"{x.frame_id/frame_sample_rate:04}", axis=1)
    if "seconds" not in df.columns:
        df["seconds"] = df.frame_id.apply(lambda frame_id: frame_id / frame_sample_rate)
    if "time" not in df.columns:
        df["time"] = df.frame_id.apply(
            lambda frame_id: datetime.utcfromtimestamp(frame_id / frame_sample_rate).strftime(
                "%H:%M:%S:%f"
            )
        )
    return df


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
                df_framedets.loc[min(int(i) + window_size, df_framedets.shape[0] - 1)].game_state
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
                "frame_id": row.frame_id,
                "end_frame_id": row.frame_id,
                "end_sec": row.seconds,
                "end_time": row.time,
            },
            {
                "state": to_state,
                "frame_id": row.frame_id,
                "start_frame_id": row.frame_id,
                "end_frame_id": None,
                "start_sec": row.seconds,
                "end_sec": None,
                "start_time": row.time,
                "end_time": None,
            },
        ]
    )


def get_game_state_endpoints(df_framedets: pd.DataFrame, smoothing=False) -> List[Dict[str, Any]]:
    """
    Given a DataFrame with the detections from a geoguessr video, returns list of dictionaries, each
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


ui_to_gamestates_map = OrderedDict(
    {
        "pre_game": {
            "any": [
                ["play", "left_menu_dark"],
                ["play", "challenge_btn_orange"],
                ["game_about_to_start_box_white", "left_menu_dark"],
                ["battle_royale_start_menu_w_logo"],
                ["game_about_to_start_box_white", "next_orange_btn"],
                ["setup_round_time_limit_box"],
                ["battle_royale_start_menu_w_logo"],
                ["start_game"],
                ["invite_friends"],
                ["next_orange_btn"],
                ["start_challenge_orange"],
                ["start_game"],
            ],
            "exact": [],
        },
        "between_rounds": {
            "any": [
                ["left_menu_dark", "challenge_high_score_board"],
                ["play_next_round", "points_bar", "did_you_enjoy_this_location"],
                ["play_next_round", "points_bar_two_bars", "did_you_enjoy_this_location"],
                ["btw_rounds_points_bar_blue_wide", "play_next_round"],
                ["battle_royale_btw_rounds_green_box"],
                ["battle_royale_btw_rounds_red_box"],
            ],
            "exact": [
                ["in_game_mini_map", "status_bar", "points_bar", "game_title"],
                ["btw_rounds_points_bar_blue_wide"],
            ],
        },
        "between_round_or_game_ambiguous": {
            "any": [
                ["points_bar_two_bars", "did_you_enjoy_this_location", "status_bar"],
                ["between_rounds_box_white", "did_you_enjoy_this_location", "status_bar"],
                ["points_bar_two_bars", "status_bar"],
                ["points_bar_two_bars", "status_bar", "game_title"],
                ["points_bar_two_bars", "did_you_enjoy_this_location", "status_bar", "game_title"],
                ["points_bar_two_bars", "status_bar", "share_challenge_box_white"],
                ["points_bar_two_bars", "challenge_high_score_board", "status_bar"],
                ["points_bar_two_bars", "status_bar", "game_title", "share_challenge_box_white"],
                [
                    "between_rounds_box_white",
                    "did_you_enjoy_this_location",
                    "status_bar",
                    "game_title",
                ],
                ["points_bar", "left_menu_dark"],
                ["points_bar", "did_you_enjoy_this_location"],
                ["points_bar_two_bars", "did_you_enjoy_this_location"],
                ["status_bar", "points_bar", "game_title"],
                ["left_menu_dark"],
                ["left_menu_dark", "status_bar", "game_title"],
                ["left_menu_dark", "other"],
                ["left_menu_dark", "points_bar"],
                ["left_menu_dark", "points_bar_two_bars"],
            ],
            "exact": [],
            "do_not_match": [
                [
                    "in_game_mini_map",
                    "in_game_map_expanded",
                    "guess",
                    "guess_grey",
                    "make_a_guess",
                    "place_your_pin_grey",
                    "guess_w_icon_only",
                    "guess_expanded",
                    "guess_grey_expanded",
                    "make_a_guess_expanded",
                    "place_your_pin_grey_expanded",
                    "guess_w_icon_only_expanded",
                ],
            ],
        },
        "in_game": {
            "any": [
                # The ones with "in_game_mini_map" in first position will get auto-expanded:
                ["in_game_mini_map", "guess"],
                ["in_game_mini_map", "guess_grey"],
                ["in_game_mini_map", "make_a_guess"],
                ["in_game_mini_map", "place_your_pin_grey"],
                ["in_game_mini_map", "guess_w_icon_only"],
                # Won't get expanded:
                # ["status_bar", "in_game_mini_map"],
                # ["status_bar", "in_game_map_expanded"],
                ["status_bar_purple", "in_game_mini_map"],
                ["status_bar_purple", "in_game_map_expanded"],
                ["game_title", "in_game_map_expanded", "status_bar", "status_bar_white"],
                ["game_title", "in_game_mini_map", "status_bar"],
                ["game_title", "in_game_map_expanded", "status_bar"],
                ["game_title", "guess", "status_bar", "status_bar_white"],
                ["game_title", "guess_grey", "status_bar", "status_bar_white"],
                ["game_title", "make_a_guess", "status_bar", "status_bar_white"],
                ["game_title", "place_your_pin_grey", "status_bar", "status_bar_white"],
                ["game_title", "guess_w_icon_only", "status_bar", "status_bar_white"],
                ["game_title", "guess_expanded", "status_bar", "status_bar_white"],
                ["game_title", "guess_grey_expanded", "status_bar", "status_bar_white"],
                ["game_title", "make_a_guess_expanded", "status_bar", "status_bar_white"],
                ["game_title", "place_your_pin_grey_expanded", "status_bar", "status_bar_white"],
                ["game_title", "guess_w_icon_only_expanded", "status_bar", "status_bar_white"],
            ],
            "exact": [
                ["status_bar_white", "status_bar", "game_title"],
                # ["game_title", "status_bar", "status_bar_white", "url"]
                ["status_bar_white", "other", "status_bar", "game_title"],
                ["game_title", "status_bar_white", "make_a_guess_expanded", "status_bar"],
                ["status_bar_purple"],
                ["game_title", "in_game_mini_map", "status_bar", "status_bar_white"],
                ["game_title", "in_game_map_expanded", "status_bar", "status_bar_white"],
            ],
            # Ignore these during matching elements in the 'any' and 'exact' lists:
            "ignore": [
                [
                    "between_rounds_box_white",
                    "btw_rounds_points_bar_blue_wide",
                    "challenge_high_score_board",
                    "did_you_enjoy_this_location",
                    "high_score_box",
                    "leader_board",
                    "left_menu_dark",
                    "play_next_round",
                    "play",
                    "points_bar_two_bars",
                    "points_bar",
                    "show_full_results",
                    "show_high_score",
                    "try_another_map",
                    "battle_royale_start_menu_w_logo",
                    "url",
                ]
            ],
            "do_not_match": [
                ["google_mini_map", "other"],
            ],
        },
        "between_games": {
            "any": [
                ["left_menu_dark", "challenge_high_score_board"],
                ["points_bar", "show_high_score"],
                ["points_bar_two_bars", "show_high_score"],
                ["try_another_map", "points_bar_two_bars", "show_full_results"],
                ["points_bar_two_bars", "show_full_results"],
                ["high_score_box", "leader_board"],
                ["battle_royale_game_over_dark_box"],
                ["battle_royale_knocked_out_end_game_red"],
                ["btw_rounds_points_bar_blue_wide", "game_breakdown_blue", "play_again"],
            ],
            "exact": [],
        },
        "out_of_game": {
            "any": [
                ["google_mini_map"],
            ],
            "exact": [
                [],
                ["other"],
            ],
        },
        "unknown": {
            "any": [],
            "exact": [],
        },
    }
)


def compute_segments(model: str):
    segments: Dict[str, List[Dict[str, Any]]] = {}
    seg_gt_new = load_gt("seg_ground_truth_009.json")
    seg_gt_new.update(seg_gt)
    print("num videos: ", len(seg_gt_new))
    val_ids = list(seg_gt_new.keys())
    val_ids_no_detection_files = {
        val_id
        for val_id in val_ids
        if not Path(
            f"/shared/gbiamby/geo/segment/detections/{model}/val/df_frame_dets-video_id_{val_id}.csv"
        ).exists()
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
