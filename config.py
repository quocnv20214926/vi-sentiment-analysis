import os
import torch
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
RESULTS_OUTPUT_DIR = OUTPUT_DIR / "results"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "train.xlsx"
VAL_DATA_PATH = DATA_DIR / "val.xlsx"
TEST_DATA_PATH = DATA_DIR / "test.xlsx"

# Model configuration
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Label mapping
LABEL2ID = {
    'sadness': 0,
    'joy': 1,
    'love': 2,
    'anger': 3,
    'fear': 4,
    'surprise': 5
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Random seed
SEED = 42

def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
