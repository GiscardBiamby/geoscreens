#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest
conda activate "${PYTHON_ENV_NAME}"

TARGET_LS_VERSION="013"

pushd ../tools || exit

python generate_pseudo_labels.py label_pipeline \
    --target_ls_version "${TARGET_LS_VERSION}" \
    --ls_project_id 84 \
    --compute_preds

popd || exit
