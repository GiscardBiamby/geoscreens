import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

from ..consts import VIDEO_PATH


def load_metadata(path: Path):
    """
    Load metadata for a single .mp4, from the .info.json file. Drops some of the really verbose json keys before returning:
        "formats", "thumbnails", "automatic_captions", "http_headers"
    """
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


def get_all_metadata() -> List[Dict[str, Any]]:
    files = sorted(VIDEO_PATH.glob("**/*.mp4"))
    print("Total video files found: ", len(files))
    all_metadata = []
    for f in tqdm(files):
        all_metadata.append(load_metadata(f))
    train_meta = get_geoguessr_split_metadata("train")
    val_meta = get_geoguessr_split_metadata("val")
    test_meta = get_geoguessr_split_metadata("test")

    splits_meta = {m["id"]: m for m in ([] + train_meta + val_meta + test_meta)}

    # fmt: off
    drop_list = set([
        'view_count', 'average_rating', 'age_limit', 'webpage_url', 'playable_in_embed', 'is_live', 'was_live', 'live_status', 'like_count',
        'dislike_count', 'availability', 'webpage_url_basename', 'extractor', 'extractor_key', 'display_id', 'format_id',
        'format_note', 'source_preference', 'tbr', 'language', 'language_preference', 'ext', 'vcodec', 'acodec',
        'dynamic_range', 'protocol', 'video_ext', 'audio_ext', 'vbr', 'abr', 'format', 'filesize_approx', 'fulltitle', 'epoch', 'path',
        "subtitles", "filesize", "release_timestamp", "release_date", "chapters", "track", "artist", "album", "creator", "alt_title", "tags",
        "ner", "caption", "label", "label_geocoder", "url"
    ])
    # fmt: on

    for meta in all_metadata:
        if meta["id"] in splits_meta:
            meta.update(splits_meta[meta["id"]])
        else:
            meta["split"] = "None"
            # print(f"WARNING: video_id {meta['id']} is not in any of the splits!")
        for col_name, value in list(meta.items()):
            if col_name in drop_list or "nemo" in col_name:
                del meta[col_name]

    return all_metadata
