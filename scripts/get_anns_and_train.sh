#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest
conda activate "${PYTHON_ENV_NAME}"

TARGET_DATASET_VERSION="010"

pushd ../tools || exit
python generate_pseudo_labels.py get_anns \
    --target_version "${TARGET_DATASET_VERSION}" \
    --ls_project_id 59
rm -f "datasets/geoscreens_${TARGET_DATASET_VERSION}/dataset_cache.pkl"
NUM_CLASSES=$(jq '.categories | max_by(.value) | .id' ../datasets/geoscreens_${TARGET_DATASET_VERSION}/geoscreens_${TARGET_DATASET_VERSION}.json)
echo "NUM_CLASSES: ${NUM_CLASSES}"

CUDA_VISIBLE_DEVICES="0,1,2,3" python train_geo.py \
    training.params.gpus=4 \
    training.experiment_name="latest_model_geoscreens_{TARGET_DATASET_VERSION}" \
    dataset_config.dataset_name="geoscreens_${TARGET_DATASET_VERSION}" \
    dataset_config.num_classes="71" \
    training.num_workers=32 \
    training.experiment_name="custom_augs" \
    training.batch_size=8

popd || exit
