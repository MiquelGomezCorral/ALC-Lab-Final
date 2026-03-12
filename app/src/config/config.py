"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
# import os
import dataclasses
from dataclasses import dataclass
from argparse import Namespace

@dataclass 
class Configuration:
    """Configuration class for the project."""

    exp_name: str = "base_name"
    seed:     int = 42

    gym_id:          str = None
    learning_rate: float = 2.5e-4
    total_timesteps: int = 25_000

    torch_deterministic: bool = True
    cuda:                bool = True

    track_run:         bool = False
    wandb_project_name: str = "RL"
    wandb_entity:       str = None

    def __post_init__(self):
        ...
