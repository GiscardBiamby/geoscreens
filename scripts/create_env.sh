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

# ## Object Detection Framework(s):
# pip install icevision[all]
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/1.10.1/index.html
# pip install mmdet

pip install -e . -c ../constraints.txt

conda list

echo "Done!"
