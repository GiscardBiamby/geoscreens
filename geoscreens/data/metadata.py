import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from geoscreens.consts import DETECTIONS_PATH

from ..consts import VIDEO_PATH


def load_metadata(path: Union[str, Path]):
    """
    Load metadata for a single .mp4, from the .info.json file. Drops some of the really verbose json keys before returning:
        "formats", "thumbnails", "automatic_captions", "http_headers"
    """
    if isinstance(path, str):
        path = Path(path).resolve()
    if path.suffix:
        path = path.with_suffix("")
    info_path = path.with_suffix(".info.json")
    data = json.load(open(info_path, "r"))
    drop_keys = set(["formats", "thumbnails", "automatic_captions", "http_headers"])
    for k in drop_keys.intersection(data.keys()):
        del data[k]
    data["path"] = path

    return data


def get_geoguessr_split_metadata(split: str) -> List[Dict]:
    """
    Grace's geoguessr dataset has train/val/test splits. This meethod returns the metadata for the
    specified split.
    """
    json_path = Path(f"/shared/g-luo/geoguessr/data/data/{split}.json").resolve()
    assert json_path.exists(), str(json_path)
    data = json.load(open(json_path, "r"))
    print(f"Length of metadata for split='{split}': ", len(data))
    for video in data:
        video["split"] = split
    return data


def _clean_attributes(meta):
    # fmt: off
    drop_list = set([
        'view_count', 'average_rating', 'age_limit', 'webpage_url', 'playable_in_embed', 'is_live', 'was_live', 'live_status', 'like_count',
        'dislike_count', 'availability', 'webpage_url_basename', 'extractor', 'extractor_key', 'display_id', 'format_id',
        'format_note', 'source_preference', 'tbr', 'language', 'language_preference', 'ext', 'vcodec', 'acodec',
        'dynamic_range', 'protocol', 'video_ext', 'audio_ext', 'vbr', 'abr', 'format', 'filesize_approx', 'fulltitle', 'epoch', 'path',
        "subtitles", "filesize", "release_timestamp", "release_date", "chapters", "track", "artist", "album", "creator", "alt_title", "tags",
        "ner", "caption", "label", "label_geocoder", "url", "nemo_caption", "nemo_caption_entities",
    ])
    # fmt: on
    for col_name, value in list(meta.items()):
        if col_name in drop_list or "nemo" in col_name:
            del meta[col_name]


def get_all_geoguessr_split_metadata(force_include: Optional[list[str]] = None) -> Dict:
    """
    Arguments:

        force_include: if specified, a list of attributes from the meta data files to include in the
        results.
    """
    train_meta = get_geoguessr_split_metadata("train")
    val_meta = get_geoguessr_split_metadata("val")
    test_meta = get_geoguessr_split_metadata("test")
    combined_meta = [] + train_meta + val_meta + test_meta
    splits_meta = {m["id"]: m for m in combined_meta}
    # fmt: off
    drop_list = set([
        'view_count', 'average_rating', 'age_limit', 'webpage_url', 'playable_in_embed', 'is_live', 'was_live', 'live_status', 'like_count',
        'dislike_count', 'availability', 'webpage_url_basename', 'extractor', 'extractor_key', 'display_id', 'format_id',
        'format_note', 'source_preference', 'tbr', 'language', 'language_preference', 'ext', 'vcodec', 'acodec',
        'dynamic_range', 'protocol', 'video_ext', 'audio_ext', 'vbr', 'abr', 'format', 'filesize_approx', 'fulltitle', 'epoch', 'path',
        "subtitles", "filesize", "release_timestamp", "release_date", "chapters", "track", "artist", "album", "creator", "alt_title", "tags",
        "ner", "caption", "label", "label_geocoder", "url", "nemo_caption", "nemo_caption_entities",
    ])
    # fmt: on
    if force_include:
        drop_list -= set(force_include)
    for video_id, video_meta in list(splits_meta.items()):
        for col_name, _ in list(video_meta.items()):
            # if col_name in drop_list or "nemo" in col_name:
            if col_name in drop_list:
                del video_meta[col_name]
    return splits_meta


def get_all_metadata() -> List[Dict[str, Any]]:
    """
    Loads all three (e.g., train/val/test) metadata files and returns all as one list.
    """
    files = sorted(VIDEO_PATH.glob("**/*.mp4"))
    print("Total video files found: ", len(files))
    all_metadata = []
    for f in tqdm(files):
        all_metadata.append(load_metadata(f))
    train_meta = get_geoguessr_split_metadata("train")
    val_meta = get_geoguessr_split_metadata("val")
    test_meta = get_geoguessr_split_metadata("test")
    splits_meta = {m["id"]: m for m in ([] + train_meta + val_meta + test_meta)}

    for meta in all_metadata:
        if meta["id"] in splits_meta:
            meta.update(splits_meta[meta["id"]])
        else:
            meta["split"] = "None"
            # print(f"WARNING: video_id {meta['id']} is not in any of the splits!")
        _clean_attributes(meta)

    return all_metadata


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


def load_detections_csv(video_id: str, split: str = "val", model: str = "") -> pd.DataFrame:
    csv_path = DETECTIONS_PATH / f"{model}/{split}/df_frame_dets-video_id_{video_id}.csv"
    df = pd.read_csv(csv_path)
    df.frame_id = df.frame_id.astype(int)
    df.frame_idx = df.frame_idx.astype(int)
    df.label_ids = df.label_ids.apply(lambda x: parse_dict(x))
    df.labels = df.labels.apply(lambda x: parse_dict(x))
    df.labels_set = df.labels_set.apply(lambda x: parse_tuple(x))
    df.scores = df.scores.apply(lambda x: parse_dict(x))
    df.bboxes = df.bboxes.apply(lambda x: parse_dict(x))

    return df


def load_detections(
    video_id: str,
    split: str = "val",
    model: str = "",
    frame_sample_rate: float = 4.0,
    prob_thresh: float = 0.7,
) -> pd.DataFrame:
    """
    NOTE: This assumes the detections are using frame sample rate of 4.0 fps. Specify
    frame_sample_rate if you're using a different setting.
    """
    dets_path = DETECTIONS_PATH / f"{model}/{split}/df_frame_dets-video_id_{video_id}.pkl"

    if dets_path.suffix == ".csv":
        df = load_detections_csv(video_id, split=split, model=model)
    else:
        df = pickle.load(open(dets_path, "rb"))

    def filter_dets(row):
        return tuple(set([l for l, s in zip(row.labels, row.scores) if s >= prob_thresh]))

    df.labels_set = df.apply(filter_dets, axis=1)
    return df
