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


    # full_dataset_path: str = os.path.join(DATA_PATH, "EXIST 2026 Videos Dataset", "training")
    full_dataset_path: str = os.path.join(DATA_PATH, "EXIST 2026 Videos Dataset", "test")
    videos_path: str = os.path.join(full_dataset_path, "videos")
    # videos_data: str = os.path.join(full_dataset_path, "EXIST2026_training.json")
    videos_data: str = os.path.join(full_dataset_path, "EXIST2026_test_clean.json")
    metadata_path: str = os.path.join(full_dataset_path, "metadata.json")
    transcriptions_path: str = os.path.join(full_dataset_path, "transcriptions.json")
    ocr_path: str = os.path.join(full_dataset_path, "ocr_results.json")
    extra_sounds_path: str = os.path.join(full_dataset_path, "extra_sounds.json")




    exp_name: str = "base_name"
    seed:     int = 42
    batch_size: int = 16

    def __post_init__(self):
        ...
