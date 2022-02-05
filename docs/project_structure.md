# Project Structure

```bash
.
└── .vscode
    ├── extensions.json
    └── settings.json
├── docs
├── lib
├── LICENSE
├── notebooks
├── PROJECT_NAME
│   ├── __init__.py
│   └── main.py
├── scripts
│   └── create_env.sh
├── .envrc
├── constraints.txt
├── PROJECT_NAME.code-workspace
├── pyproject.toml
├── manifest
├── README.md
├── requirements.txt
├── setup.cfg
├── setup.py
```

All your project code should go into `./<PROJECT_NAME>`. Create python sub-modules in there as necessary.

The create_env.sh script will install your code into your python environment and you can then use your code as a normal python module that you can import. No need to do any weird stuff with sys.path.append("./...") anywhere in your code. For example you can import functions from your code into a jupyter notebook in `./notebooks/`:

```python
from PROJECT_NAME import main
```

## Including Code From External Libraries

Most of the time you would simply add dependencies to `./requirements.txt`, but sometimes you want to clone an external repo and modify it (e.g., customize the training code or models from some machine learning framework/repo).

In those custom cases, clone the external libraries into `./lib`. You can then add a step to `./create_env.sh` to install the external library as a locally editable package in your python environment. Here is an example with Pytorch Image Models ([https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)):

```bash
cd ./lib
git clone git@github.com:rwightman/pytorch-image-models.git
cd ./pytorch-image-models
pip install -e .
```

Your folder structure will look like:

```bash
PROJECT_NAME/
└── .vscode
└── docs
└── lib
    └── pytorch-image-models
        ├── convert
        ├── docs
        ├── .github
        ├── notebooks
        ├── results
        └── timm
        └── ...
├── LICENSE
...

```

asdf
