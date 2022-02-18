#!/bin/bash

# If you don't use anaconda or miniconda you can replace the relevant environment creation and
# activation lines with pyenv or whatever system you use to manage python environments.
source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest

conda activate "${ENV_NAME}"

pushd ../
# export LOCAL_FILES_SERVING_ENABLED=true
# export LOCAL_FILES_DOCUMENT_ROOT=/shared/gbiamby/geo/screenshots
# export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/shared/gbiamby/geo/screenshots

export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=/shared/gbiamby/geo/videos
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/shared/gbiamby/geo/videos

label-studio start --port 6008
popd || exit

# Don't forget to launch the web server to serve images from $LOCAL_FILES_DOCUMENT_ROOT
