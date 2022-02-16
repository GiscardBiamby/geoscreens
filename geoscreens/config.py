import os
import uuid
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_only

from geoscreens.consts import PROJECT_ROOT


# For strict typing purposes, prefer OmegaConf.structured() when creating structured configs
@dataclass
class MyConfig:
    port: int = 80
    host: str = "localhost"


def resolve_dir(env_variable, default="data"):
    # default_dir = os.path.join(resolve_cache_dir(), default)
    default_dir = PROJECT_ROOT / "datasets"
    dir_path = os.getenv(env_variable, default_dir)
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    return str(dir_path)


def _resolve_path(config: Union[dict, DictConfig], key: str, default=None):
    if hasattr(config, "get"):
        path = Path(config.get(key, default))
    else:
        path = Path(config[key])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config[key] = str(path.resolve())
    return config


def build_config(args: Namespace) -> DictConfig:
    config_path = _resolve_path(vars(args), "config_file")["config_file"]
    base_config = OmegaConf.load(PROJECT_ROOT / "configs/default.yaml")
    config = OmegaConf.load(config_path)
    cli_conf = OmegaConf.from_cli(args.overrides)
    config = cast(DictConfig, OmegaConf.merge(base_config, config, cli_conf))
    config.config_file = config_path
    _resolve_path(config.dataset_config, "data_root", "./datasets")
    _resolve_path(config.dataset_config, "img_dir", "./datasets/images")

    uid = str(uuid.uuid4()).replace("-", "")[:10]
    exp_name = "-".join(
        [
            f"{config.dataset_config.dataset_name}",
            f"model_{config.model_config.name}",
            f"bb_{config.model_config.backbone}",
            f"{uid}",
        ]
    )
    config.training.experiment_name += "--" + exp_name

    # TODO: Add logic to reuse an existing folder if we are resuming training.
    save_dir = Path(f"{config.env.save_dir}/{config.training.experiment_name}").resolve()
    config.env.save_dir = str(save_dir)

    # Resolve the config here itself after full creation so that spawned workers don't face any
    # issues
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, True)
    OmegaConf.set_readonly(config, True)
    print(OmegaConf.to_yaml(config))
    save_config(config)
    return cast(DictConfig, config)


@rank_zero_only
def save_config(config: DictConfig):
    Path(config.env.save_dir).mkdir(exist_ok=True, parents=True)
    with open(Path(config.env.save_dir) / "config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp)


def _register_resolvers():
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("resolve_dir", resolve_dir)


_register_resolvers()
