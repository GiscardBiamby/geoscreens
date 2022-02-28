from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from IPython.display import display
from pycocotools.helpers import CocoClassDistHelper as COCO
from pycocotools.helpers import CocoJsonBuilder
from tqdm.auto import tqdm

from .metadata import get_all_metadata


def get_images_with_metadata(coco: COCO, df_meta: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Gets list of all images in the given COCO object, combined with metadata info
    from df_meta.

    Returns::

        (Example)
            [
                {
                    'file_name': 'screen_samples_auto/DZ9JablpbhQ/frame_00000012.jpg',
                    'height': 720,
                    'id': 0,
                    'width': 1280,
                    'video_id': 'DZ9JablpbhQ',
                    'author': 'Mister Blue Geoguessr',
                    'ann_count': 6
                }, ...
            ]
    """
    img_data = []
    for img in coco.dataset["images"]:
        video_id = Path(img["file_name"]).parent.name
        img_data.append(
            {
                **img,
                "video_id": video_id,
                "author": df_meta.loc[video_id]["uploader"] if video_id in df_meta.index else None,
                "ann_count": sum(
                    [1 for ann in coco.dataset["annotations"] if ann["image_id"] == img["id"]]
                ),
                "meta_split": df_meta.loc[video_id]["split"] if video_id in df_meta.index else None,
            }
        )
    return img_data


def get_metadata_df(all_metadata) -> pd.DataFrame:
    df_meta = pd.DataFrame(all_metadata).set_index("id")
    df_meta = df_meta[~(df_meta.split == "None")].copy(deep=True)

    return cast(pd.DataFrame, df_meta)


def generate_train_val_splits(
    coco_path: Path,
    split_by: str = "author",
    train_pct: float = 0.8,
):
    if isinstance(coco_path, str):
        coco_path = Path(coco_path)
    # Prepare metadata about the images;
    df_meta = cast(pd.DataFrame, get_metadata_df(get_all_metadata()))

    # Get image data from the source COCO dataset:
    coco = COCO(coco_path)
    img_data = get_images_with_metadata(coco, df_meta)
    df_images = pd.DataFrame(img_data)
    df_images.rename(columns={"id": "image_id"}, inplace=True)

    # Generate the grouping (e.g., group by author|video_id|etc first and then all images under a
    # group all go into exactly one split)
    df_groupings = pd.DataFrame(
        df_images.groupby([split_by])
        .agg(img_count=("image_id", "count"), ann_count=("ann_count", "sum"))
        .sort_values(["img_count"], ascending=False)
    )  # noqa
    df_groupings = df_groupings.reset_index()
    df_groupings = df_groupings.set_index(split_by)
    df_groupings = (
        df_groupings.merge(
            pd.DataFrame(df_groupings.img_count.cumsum())
            .reset_index()
            .set_index(split_by)
            .rename(columns={"img_count": "img_count_cum"}),
            left_index=True,
            right_index=True,
        )
        .reset_index()
        .set_index(split_by)
    )

    # Compute which groups go into which split, according to the train_pct setting:
    df_groupings["split"] = df_groupings.apply(
        lambda row: "train" if row.img_count_cum <= int(train_pct * len(df_images)) else "val",
        axis=1,
    )

    display(df_groupings)

    # Generate and save the splits:
    train_images = df_images[
        df_images.author.isin(
            df_groupings[df_groupings.split == "train"].reset_index().author,
        )
    ]
    val_images = df_images[~df_images.image_id.isin(train_images.image_id)]
    generate_coco_split(
        coco_path,
        "train",
        train_images.image_id.values.tolist(),
    )
    generate_coco_split(
        coco_path,
        "val",
        val_images.image_id.values.tolist(),
    )


def generate_coco_split(input_file: Path, split: str, img_ids: List[int]):
    coco = COCO(input_file)
    output_path = input_file.with_name(f"{input_file.stem}_{split}.json")
    print("input_path: ", input_file)
    print("output_path: ", output_path)

    coco_builder = CocoJsonBuilder(
        coco.cats, dest_path=output_path.parent, dest_name=output_path.name
    )
    for idx, img_id in enumerate(img_ids):
        coco_builder.add_image(coco.imgs[img_id], coco.imgToAnns[img_id])
    coco_builder.save()
