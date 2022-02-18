#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest
conda activate "${PYTHON_ENV_NAME}"

TARGET_DATASET_VERSION="011"

pushd ../tools || exit
# python generate_pseudo_labels.py ls_to_coco \
#     --target_ds_version "${TARGET_DATASET_VERSION}" \
#     --ls_project_id 74
# rm -f "../datasets/geoscreens_${TARGET_DATASET_VERSION}/dataset_cache.pkl"
NUM_CLASSES=$(jq '.categories | max_by(.value) | .id' ../datasets/geoscreens_${TARGET_DATASET_VERSION}/geoscreens_${TARGET_DATASET_VERSION}.json)
NUM_CLASSES=$(($NUM_CLASSES + 1))
echo "NUM_CLASSES: ${NUM_CLASSES}"

CUDA_VISIBLE_DEVICES="4,5,6,7" python train_geo.py \
    training.params.gpus=4 \
    training.experiment_name="latest_model_geoscreens_{TARGET_DATASET_VERSION}" \
    dataset_config.dataset_name="geoscreens_${TARGET_DATASET_VERSION}" \
    dataset_config.num_classes="${NUM_CLASSES}" \
    training.num_workers=32 \
    training.experiment_name="gs_${TARGET_DATASET_VERSION}_extra_augs" \
    training.batch_size=8

popd || exit
