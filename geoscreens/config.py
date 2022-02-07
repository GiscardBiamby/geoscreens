import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

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


def build_config(args: Namespace) -> DictConfig:
    base_config = OmegaConf.load(PROJECT_ROOT / "configs/default.yaml")
    config = OmegaConf.load(args.config_file)
    cli_conf = OmegaConf.from_cli(args.overrides)
    config = OmegaConf.merge(base_config, config, cli_conf)

    data_root = Path(config.dataset_config.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    config.dataset_config.data_root = str(data_root.resolve())

    # Resolve the config here itself after full creation so that spawned workers don't face any
    # issues
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, True)
    print(OmegaConf.to_yaml(config))
    # sys.exit()
    return cast(DictConfig, config)


def _register_resolvers():
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("resolve_dir", resolve_dir)


_register_resolvers()
