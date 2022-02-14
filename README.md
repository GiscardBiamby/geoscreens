# geoscreens

A dataset to detect UI elements in geoguessr videos!

## Installation

### Setup Python Environment

This uses Anaconda to create the package, and installs requirements from requirements.txt. The environment name is defined in the `./manifest` file.

```bash
cd scripts
./create_env.sh
```

## Setup Data

Copy images (\~26K total) from `/shared/gbiamby/geo/screenshots/screen_samples_auto/`.

Alternatively you can set `img_dir` in the dataset_config section of your config file(s) to point to the above path and skip the copying step.

## Train a Detector

```bash
cd ./tools
python train_geo.py --config_file ../configs/mmdet_faster_rcnn.resnest.yaml
```
