from maikol_utils.file_utils import load_json

from src.config import Configuration


def train_test_mostro(CONFIG: Configuration):
    metadata = load_json(CONFIG.metadata_path)


    