#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest
conda activate "${PYTHON_ENV_NAME}"

TARGET_DATASET_VERSION="013"

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

# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python train_geo.py \
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python train_geo.py \
    training.params.gpus=4 \
    dataset_config.dataset_name="geoscreens_${TARGET_DATASET_VERSION}" \
    dataset_config.num_classes="${NUM_CLASSES}" \
    training.num_workers=32 \
    training.experiment_name="gs_urls_gamma0.2_03a_${TARGET_DATASET_VERSION}" \
    training.batch_size=8 \
    training.params.max_epochs=38 \
    training.wandb.enabled=true \
    scheduler.MultiStepLR.params.milestones=[18,25,32] \
    scheduler.MultiStepLR.params.gamma=0.2 \
    model_config.mmdet.faster_rcnn.cfg_options.model--rpn_head--anchor_generator--strides=[4,6,23,52,63]

popd || exit
