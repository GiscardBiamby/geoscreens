import json
import logging
import os
import pickle
import shutil
import sys
from copy import deepcopy
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.bells import tqdm

from geoscreens.consts import EXTRACTED_FRAMES_PATH, FRAMES_METADATA_PATH, VIDEO_PATH
from geoscreens.data.metadata import GOOGLE_SHEET_IDS
from geoscreens.utils import get_indices_to_sample, load_json, save_json, timeit_context


def get_frames_generator_decord(config: DictConfig, video_path: Union[str, Path]):
    vr = VideoReader(str(video_path), ctx=cpu())
    sample_indices = get_indices_to_sample(config, len(vr), vr.get_avg_fps())
    print(
        f"num_frames: {len(vr):,}, num_to_sample: {len(sample_indices):,}, fps: {vr.get_avg_fps()}"
    )
    for sample_idx in tqdm(
        range(len(sample_indices)),
        total=len(sample_indices),
        disable=config.get("disable_progress_bar", False),
    ):
        frame_idx = sample_indices[sample_idx]
        if config.fast_debug and sample_idx >= config.debug_max_frames:
            break
        frame = vr[frame_idx]
        seconds = round(frame_idx / vr.get_avg_fps(), 2)
        yield (frame_idx, seconds, frame)


@timeit_context("extract_frames")
def extract_frames(config: DictConfig, video_path: Path, get_frames_fn: Callable):
    frames_path = Path(config.video_frames_path) / video_path.stem
    if frames_path.exists():
        return (video_path, True, None)
    frames_path.mkdir(exist_ok=True, parents=True)
    try:
        print("Saving frames to: ", frames_path)
        for frame_idx, seconds, frame in get_frames_fn(config, video_path):
            frame_out_path = frames_path / f"frame_{frame_idx:08}-{seconds:010.3f}s.jpg"
            if not frame_out_path.exists():
                cv2.imwrite(str(frame_out_path), cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))
        return (video_path, True, None)
    except Exception as ex:
        print(f"Failed: {video_path}, error: {str(ex)}")
        shutil.rmtree(str(frames_path))
        return (video_path, False, str(ex))


def extract_frames_fake(config: DictConfig, video_path: Path, get_frames_fn: Callable):
    try:
        frames_path = Path(config.video_frames_path) / video_path.stem
        frames_path.mkdir(exist_ok=True, parents=True)
        print("Saving frames to: ", frames_path)
        if "pF9OA332DPk" in str(video_path):
            raise Exception("Fake error")
        return (video_path, True, None)
    except Exception as ex:
        return (video_path, False, str(ex))


def save_frames_metadata(config: DictConfig, files):
    frame_info = {}
    if FRAMES_METADATA_PATH.exists():
        frame_info = load_json(FRAMES_METADATA_PATH)
    for file in tqdm(files):
        video_id = file.stem
        vr = VideoReader(str(file), ctx=cpu(0))
        sample_indices = get_indices_to_sample(config, len(vr), vr.get_avg_fps())
        frame_info[video_id] = {
            "video_id": video_id,
            "total_frames": len(vr),
            "video_fps": vr.get_avg_fps(),
            "frame_sample_rate_fps": 4.0,
            "num_frames_sampled": len(sample_indices),
        }
    out_path = FRAMES_METADATA_PATH.parent / f"{FRAMES_METADATA_PATH.stem}_new.json"
    save_json(out_path, frame_info)

    # Save list of files for each video:
    # Append frames_list results for all items in `files` to existing pickle file:
    results = {}
    if FRAMES_METADATA_PATH.with_name("frames_list.pkl").exists():
        results = pickle.load(open(FRAMES_METADATA_PATH.with_name("frames_list.pkl"), "rb"))
    for file in tqdm(files):
        video_id = file.stem
        if (EXTRACTED_FRAMES_PATH / f"{video_id}").exists():
            frames = sorted(os.listdir(EXTRACTED_FRAMES_PATH / f"{video_id}"))
            results[video_id] = [
                {"video_id": video_id, "file_path": f"{video_id}/{f}"} for f in frames
            ]
    pickle.dump(results, open(FRAMES_METADATA_PATH.with_name("frames_list.pkl"), "wb"))


def process_videos_muli_cpu(config: DictConfig):
    id_list = deepcopy(GOOGLE_SHEET_IDS)

    # EXCLUDE files that are already extracted:
    frame_info = load_json(FRAMES_METADATA_PATH)
    id_list = id_list.union(set(frame_info.keys()))

    files = sorted(VIDEO_PATH.glob("*.mp4"))
    files = [f for f in files if f.stem not in id_list][:500]
    print("Num videos: ", len(files))

    num_workers = config.get("num_workers", 16)
    args = ((config, video_path, get_frames_generator_decord) for i, video_path in enumerate(files))
    with Pool(processes=num_workers) as pool:
        result = pool.starmap(extract_frames, args)
        df_results = pd.DataFrame(
            {
                "video_path": [r[0] for r in result],
                "success": [bool(r[1]) for r in result],
                "error": [r[2] for r in result],
            }
        )
        print("")
        print("")
        print("Failed videos:")
        print(df_results[~df_results.astype(bool).success])

    save_frames_metadata(config, files)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.max_rows", None)
    # Suitable default display for floats
    pd.options.display.float_format = "{:,.2f}".format
    config = DictConfig(
        {
            "frame_sample_rate_fps": 4.0,
            "fast_debug": False,
            "debug_max_frames": 30,
            # "video_frames_path": "/home/gbiamby/proj/geoscreens/data/video_frames",
            # "videos_path": "/home/gbiamby/proj/geoscreens/data/videos",
            "video_frames_path": "/shared/gbiamby/geo/video_frames",
            "videos_path": "/shared/g-luo/geoguessr/videos",
            "num_workers": 16,
            "disable_progress_bar": True,
        }
    )
    process_videos_muli_cpu(config)
