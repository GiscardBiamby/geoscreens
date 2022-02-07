from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class MyConfig:
    port: int = 80
    host: str = "localhost"


# For strict typing purposes, prefer OmegaConf.structured() when creating structured configs
