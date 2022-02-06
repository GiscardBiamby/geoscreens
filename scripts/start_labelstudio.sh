#!/bin/bash

# If you don't use anaconda or miniconda you can replace the relevant environment creation and
# activation lines with pyenv or whatever system you use to manage python environments.
source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest

conda activate "${ENV_NAME}"

pushd ../

label-studio start --port 6008

popd
