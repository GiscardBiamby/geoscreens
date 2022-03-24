import argparse
import json
import operator
import os
import pickle
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import easyocr
import numpy as np
import pandas as pd
import PIL.Image as pil_img
from tqdm.contrib import tenumerate, tmap, tzip
from tqdm.contrib.bells import tqdm, trange

from geoscreens.consts import (
    EXTRACTED_FRAMES_PATH,
    FRAMES_METADATA_PATH,
    LATEST_DETECTION_MODEL_NAME,
    VIDEO_PATH,
)
from geoscreens.data import get_all_geoguessr_split_metadata
from geoscreens.data.metadata import GOOGLE_SHEET_IDS, FramesList
from geoscreens.utils import batchify, load_json, save_json, timeit_context


def last_index(lst, value):
    return len(lst) - operator.indexOf(reversed(lst), value) - 1


def get_video_urls(args, gpu_id: int, df_url_frames: pd.DataFrame):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    reader = easyocr.Reader(["en"], gpu=True)
    urls = defaultdict(list)

    for i, (idx, row) in tenumerate(
        df_url_frames.iterrows(), total=len(df_url_frames), position=gpu_id
    ):
        # if i >= 400:
        #     break
        # print(row)
        video_id = row.video_id
        url_idx = last_index(row.labels, "url")
        # Crop:
        img = pil_img.open(row.file_path)
        url_area = row.bboxes[url_idx]
        url_area = (url_area["xmin"], url_area["ymin"], url_area["xmax"], url_area["ymax"])
        img_cropped = img.crop(url_area)
        result = reader.recognize(np.array(img_cropped))
        urls[video_id].append({**row.to_dict(), "ocr": result})
        # print(result)
    return urls


def main(args):
    df_ingame = pickle.load(open("/shared/gbiamby/geo/segment/in_game_frames_000.pkl", "rb"))
    df_url_frames = df_ingame[df_ingame.labels.apply(lambda l: "url" in l)].copy(deep=True)

    # Divide work evenly across num_gpus:
    df_url_frames["row_num"] = df_url_frames.reset_index().index
    df_url_frames["gpu_id"] = df_url_frames.row_num.apply(lambda x: x % args.num_gpu)
    worker_args = (
        (deepcopy(args), gpu_id, df_url_frames[df_url_frames.gpu_id == gpu_id].copy(deep=True))
        for gpu_id in range(args.num_gpu)
    )
    # print(list(worker_args))

    # Compute
    with Pool(processes=args.num_gpu) as pool:
        urls = pool.starmap(get_video_urls, worker_args)

    print("urls: ", len(urls))
    # Combine results from each thread into a single dict:
    all_urls = {}
    for _urls in urls:
        all_urls.update(_urls)
    print("all_urls: ", len(all_urls))

    # Save
    save_file = Path("/shared/gbiamby/geo/data/urls") / f"url_ocr_raw.pkl"
    print("Saving results to: ", save_file)
    save_file.parent.mkdir(exist_ok=True, parents=True)
    pickle.dump(all_urls, open(save_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=3,
        help="Num gpu threads/processes to use. More is not always better. I found 3 to be better than any value >=5.",
    )
    args = parser.parse_args()
    main(args)
