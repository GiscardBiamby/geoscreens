#!/bin/bash

# If you don't use anaconda or miniconda you can replace the relevant environment creation and
# activation lines with pyenv or whatever system you use to manage python environments.

# shellcheck source=/home/gbiamby/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
# shellcheck source=../manifest
source "../manifest"

ENV_NAME=$PYTHON_ENV_NAME
echo "ENV_NAME: ${ENV_NAME}"

## Remove env if exists:
conda deactivate && conda env remove --name "${ENV_NAME}"
rm -rf "/home/${USER}/anaconda3/envs/${ENV_NAME}"

# Create env:
conda create --name "${ENV_NAME}" python=="${PYTHON_VERSION}" -y

conda activate "${ENV_NAME}"
echo "Current environment: "
conda info --envs | grep "*"

##
## Base dependencies
echo "Installing requirements..."
pip install --upgrade pip -c ../constraints.txt
pip install -r ../requirements.txt -c ../constraints.txt

# Make the python environment available for running jupyter kernels:
python -m ipykernel install --user --name="${ENV_NAME}"
# Install jupyter extensions
jupyter contrib nbextension install --user

pushd ..
pip install -e . -c constraints.txt
popd

# ## Object Detection Framework(s):
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html -c ../constraints.txt --upgrade
pip install mmdet==2.16.0 -c ../constraints.txt --upgrade
pip install yolov5-icevision --upgrade -c ../constraints.txt

# Install external library code:
pushd ../lib/ || return

# Install icevision (https://github.com/airctic/icevision):
git clone git@github.com:GiscardBiamby/icevision.git
pushd icevision || return
git checkout geo
pip install -e .[all,dev] -c ../../constraints.txt
popd || return

# Install customized version of pycocotools (https://github.com/GiscardBiamby/cocobetter):
git clone git@github.com:GiscardBiamby/cocobetter.git
pushd ./cocobetter/PythonAPI || return
pip install -e . -c ../../../constraints.txt
popd || return

popd || return

# We are done, show the python environment:
conda list
echo "Done!"
