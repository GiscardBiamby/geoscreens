#!/bin/bash

# If you don't use anaconda or miniconda you can replace the relevant environment creation and
# activation lines with pyenv or whatever system you use to manage python environments.
source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest

ENV_NAME=$PYTHON_ENV_NAME
echo "ENV_NAME: ${ENV_NAME}"

## Remove env if exists:
conda deactivate && conda env remove --name "${ENV_NAME}"
rm -rf "/home/gbiamby/anaconda3/envs/${ENV_NAME}"

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

pip install -e . -c ../constraints.txt

# ## Object Detection Framework(s):
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/1.10.1/index.html -c ../constraints.txt
pip install mmdet -c ../constraints.txt
# pip install icevision[all]

# Install external library code:
pushd ../lib/

# Install icevision (https://github.com/airctic/icevision):
git clone git@github.com:GiscardBiamby/icevision.git
pushd icevision
pip install -e .[all,dev] -c ../../constraints.txt
popd

# Install customized version of pycocotools (https://github.com/GiscardBiamby/cocobetter):
git clone git@github.com:GiscardBiamby/cocobetter.git
pushd ./cocobetter/PythonAPI
pip install -e .
popd

popd

# We are done, show the python environment:
conda list
echo "Done!"
