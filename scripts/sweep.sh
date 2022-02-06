#/bin/bash

# If you don't use anaconda or miniconda you can replace the relevant environment creation and
# activation lines with pyenv or whatever system you use to manage python environments.
source ~/anaconda3/etc/profile.d/conda.sh
source ../manifest

ENV_NAME=$PYTHON_ENV_NAME
echo "ENV_NAME: ${ENV_NAME}"
conda activate "${ENV_NAME}"
echo "Current environment: "
conda info --envs | grep "*"

for lr in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6; do
    echo $lr
    python train_geo.py --lr "${lr}"
done
