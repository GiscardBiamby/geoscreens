# README

## Installation

### Update Project Name

1. Rename `./PROJECT_NAME.code-workspace`
2. Rename `./PROJECT_NAME` folder
3. Edit `./manifest` file with new project name: `PYTHON_ENV_NAME=PROJECT_NAME`
4. Edit `name="PROJECT_NAME"` and `url="https://github.com/USERNAME/PROJECT_NAME/"` line in in `./setup.py`

### Setup Python Environment

This uses Anaconda to create the package, and installs requirements from requirements.txt. The environment name is defined in the `./manifest` file.

```bash
cd scripts
./create_env.sh
```
