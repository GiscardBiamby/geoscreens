#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest
conda activate "${PYTHON_ENV_NAME}"

TARGET_DATASET_VERSION="012"

pushd ../tools || exit
# python generate_pseudo_labels.py ls_to_coco \
#     --target_ds_version "${TARGET_DATASET_VERSION}" \
#     --ls_project_id 84

# Clear dataset cache:
# rm -f "../datasets/geoscreens_${TARGET_DATASET_VERSION}/dataset_cache.pkl"
# rm -f "../datasets/geoscreens_${TARGET_DATASET_VERSION}/dataset_cache_train.pkl"
# rm -f "../datasets/geoscreens_${TARGET_DATASET_VERSION}/dataset_cache_valid.pkl"

# Set num_classes:
NUM_CLASSES=$(jq '.categories | max_by(.value) | .id' ../datasets/geoscreens_${TARGET_DATASET_VERSION}/geoscreens_${TARGET_DATASET_VERSION}.json)
NUM_CLASSES=$(($NUM_CLASSES + 1))
echo "NUM_CLASSES: ${NUM_CLASSES}"

# CUDA_VISIBLE_DEVICES="5,6,7,8,9" python train_geo.py \
CUDA_VISIBLE_DEVICES="4,5,6,7" python train_geo.py \
    training.params.gpus=4 \
    dataset_config.dataset_name="geoscreens_${TARGET_DATASET_VERSION}" \
    dataset_config.num_classes="${NUM_CLASSES}" \
    training.num_workers=32 \
    training.experiment_name="gsmoreanch02b_${TARGET_DATASET_VERSION}" \
    training.batch_size=8 \
    training.params.max_epochs=33 \
    training.wandb.enabled=true \
    scheduler.MultiStepLR.params.milestones=[18,25,32] \
    model_config.mmdet.faster_rcnn.cfg_options.model--rpn_head--anchor_generator--strides=[4,6,23,52,63] \
    model_config.mmdet.faster_rcnn.cfg_options.model--rpn_head--anchor_generator--ratios=[0.49,0.32,0.20,0.17,0.08,0.12,0.20,1.21,1.38,0.78,0.38,0.57,1.32,1.24,5.09]
popd || exit
