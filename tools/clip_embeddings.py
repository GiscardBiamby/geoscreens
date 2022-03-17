import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from copy import deepcopy
from multiprocessing.dummy import Pool
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

batch_size = 64


def embed_text(sentences, model, device):
    tokenized = clip.tokenize(sentences, truncate=True).to(device, non_blocking=True)
    with torch.no_grad():
        text_embeddings = model.encode_text(tokenized)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


def get_text_embeddings(anns, model, device, gpu_id):
    texts, ids = zip(*anns)
    texts, ids = list(texts), list(ids)
    text_embeddings = {}
    for i in tqdm(
        range(0, len(texts), batch_size),
        total=len(range(0, len(texts), batch_size)),
        desc=f"get_text_emb_{gpu_id}",
        position=gpu_id,
    ):
        batch_ids = ids[i : i + batch_size]
        embed = embed_text(texts[i : i + batch_size], model, device).detach().cpu().numpy()
        for j in range(len(batch_ids)):
            text_embeddings[batch_ids[j]] = embed[j]
    return text_embeddings


def embed_image(images, model):
    with torch.no_grad():
        image_embedddings = model.encode_image(images)
        image_embedddings /= image_embedddings.norm(dim=-1, keepdim=True)
    return image_embedddings


def get_valid_images(ids, image_paths, preprocess):
    invalid_ids = []
    images = []
    for i, image_path in enumerate(image_paths):
        try:
            image = Image.open(image_path)
            image = preprocess(image).unsqueeze(0)
            images.append(image)
        except Exception as e:
            print(e, ids[i])
            invalid_ids.append(ids[i])
    return images, set(invalid_ids)


def get_image_embeddings(anns, model, preprocess, device, gpu_id) -> dict[str, np.ndarray]:
    image_paths, ids = zip(*anns)
    image_paths, ids = list(image_paths), list(ids)
    image_embeddings = {}
    for i in tqdm(
        range(0, len(image_paths), batch_size),
        total=len(list(range(0, len(image_paths), batch_size))),
        position=gpu_id,
        desc=f"get_img_emb_{gpu_id}",
    ):
        try:
            batch_ids = ids[i : i + batch_size]
            batch_images, invalid_batch_ids = get_valid_images(
                batch_ids, image_paths[i : i + batch_size], preprocess
            )
            images_tensor = torch.vstack(batch_images).to(device, non_blocking=True)
            embed = embed_image(images_tensor, model).detach().cpu().numpy()
            for j in range(embed.shape[0]):
                if batch_ids[j] not in invalid_batch_ids:
                    image_embeddings[batch_ids[j]] = embed[j]
        except Exception as e:
            print(e)
    return image_embeddings


def load_pretrained(args, device: torch.device, model: nn.Module, checkpoint_path: Path):
    if args.gpu is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}), device: {device}")


def get_text_anns(args):
    """
    Return a list of 2-tuples. First tuple item is sentence text, second is sentence text (or a
    unique id for the sentence if one exists). Return value looks like::

        [("cars are driving on the left", "cars are driving on the left), ...]
    """
    if args.dataset == "geoframes":
        args.data_file_name = f"geoframes_clip_samples_fixed_window_{args.split}"
        data = pd.read_csv(
            f"/shared/gbiamby/geo/captions/clip_samples_fixed_window_{args.split}_openclip.csv"
        )
        texts = [p for p in set(data.sentence.values)]
        ids = texts
    if args.dataset == "gptj_clues":
        args.data_file_name = f"gptj_clues"
        data = json.load(
            open("/shared/g-luo/geoguessr/data/data/guidebook/kb/v3/cleaned_clues.json")
        )
        texts = list(set([t["text"] for t in data["clues"]]))
        ids = texts
    else:
        raise NotImplementedError()
    # De-duplicate by "id", i.e., sentence text:
    id_to_path = {}
    for path, _id in zip(texts, ids):
        id_to_path[_id] = path
    print(f"Loaded dataset {args.dataset}-{args.data_type} with {len(ids)} items")
    return list(zip(list(id_to_path.values()), list(id_to_path.keys())))


def get_image_anns(args):
    """
    Return a list of 2-tuples. First tuple item is full image path, second is a unique identifier
    for the image (which can also just be the image path)

    Return value looks like::

        [("/shared/you/geo/img_01.jpg", "/shared/you/geo/img_01.jpg"), ...]
    """
    if args.dataset in ["flickr", "yfcc", "yfcc100m"]:
        args.data_file_name = "placing2014_reference"
        img_base_path = Path("/shared/group/yfcc100m/placing2014/data/images")
        dataset = json.load(
            open(f"/shared/g-luo/geoguessr/models/clip_zs/{args.data_file_name}.json")
        )
        image_paths = [img_base_path / ann["IMG_ID"] for ann in dataset]
        ids = [ann["hash"] for ann in dataset]
    elif args.dataset == "geoframes":
        args.data_file_name = f"geoframes_clip_samples_fixed_window_{args.split}"
        data = pd.read_csv(
            f"/shared/gbiamby/geo/captions/clip_samples_fixed_window_{args.split}_openclip.csv"
        )
        image_paths = [p for p in set(data.file_path.values)]
        ids = image_paths
    elif args.dataset == "streetview":
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    # De-duplicate by "id":
    id_to_path = {}
    for path, _id in zip(image_paths, ids):
        id_to_path[_id] = path
    print(f"Loaded dataset {args.dataset} with {len(ids)} items")
    return list(zip(list(id_to_path.values()), list(id_to_path.keys())))


def get_embeddings_worker(args, gpu_id: int, anns):
    args.gpu = gpu_id
    device = torch.device(f"cuda:{gpu_id}")
    model, preprocess = clip.load(args.model_name, device=device)
    if "checkpoint" in args and args.checkpoint is not None:
        load_pretrained(args, device, model, args.checkpoint)
    model = model.to(device)

    if args.data_type in ["img", "image"]:
        embeddings = get_image_embeddings(anns, model, preprocess, device, gpu_id)
    else:
        embeddings = get_text_embeddings(anns, model, device, gpu_id)
    return embeddings


def get_embeddings(args):
    if args.data_type in ["img", "image"]:
        anns = get_image_anns(args)
    else:
        anns = get_text_anns(args)
    all_embeddings = {}

    # Divide the work into roughly equal chunks across num_gpus:
    gpu_to_anns = defaultdict(lambda: [])
    for i, ann in enumerate(anns):
        gpu_id = i % args.num_gpu
        gpu_to_anns[gpu_id].append(ann)
    worker_args = ((deepcopy(args), gpu_id, anns) for gpu_id, ids in gpu_to_anns.items())

    # Compute
    with Pool(processes=args.num_gpu) as pool:
        embeddings = pool.starmap(get_embeddings_worker, worker_args)

    # Combine results from each thread into a single dict:
    for emb in embeddings:
        all_embeddings.update(emb)

    print("Total embeddings: ", len(all_embeddings))
    save_file = (
        Path("/shared/gbiamby/geo/models/clip_ft")
        / args.model_name.lower().replace("/", "")
        / f"{args.data_file_name}_{args.data_type}.pkl"
    )
    print("Saving results to: ", save_file)
    save_file.parent.mkdir(exist_ok=True, parents=True)
    pickle.dump(all_embeddings, open(save_file, "wb"))


def main(args):
    get_embeddings(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr",
        help="Name of dataset to generate embeddings for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="(Optional) Split (train/val/test) within the dataset, if applicable, to generate emb's for.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="image",
        help="'image' or 'text'",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=1,
        help="Num gpu threads to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/gbiamby/proj/im2gps_kb/lib/open_clip/logs/lr=1e-06_wd=0.1_agg=True_model=ViT-B/32_batchsize=96_workers=4_date=2022-03-15-04-33-55/checkpoints/epoch_2.pt"
        ),
        help="Path to model checkpoint. If None then loads off the shelf pretrained CLIP",
    )
    args = parser.parse_args()
    main(args)
