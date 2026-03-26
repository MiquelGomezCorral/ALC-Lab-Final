"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass

@dataclass 
class Configuration:
    """Configuration class for the project."""
    DATA_PATH: str = os.path.join("..", "data")
    MODEL_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")


    videos_path: str = os.path.join(DATA_PATH, "EXIST 2026 Videos Dataset", "training", "videos")
    videos_data: str = os.path.join(DATA_PATH, "EXIST 2026 Videos Dataset", "training", "EXIST2026_training.json")
    transcriptions_path: str = os.path.join(DATA_PATH, "EXIST 2026 Videos Dataset", "training", "transcriptions.json")




    exp_name: str = "base_name"
    seed:     int = 42
    batch_size: int = 16

    def __post_init__(self):
        ...
