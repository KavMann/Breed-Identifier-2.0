from pathlib import Path
import os

import torch

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data_split"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CLASSES_JSON = PROJECT_DIR / "models" / "classes.json"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_checkpoint.pth"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pth"
FINAL_MODEL = MODEL_DIR / "dog_breed_classifier.pth"

TRAIN_HISTORY_CSV = LOG_DIR / "training_history.csv"

IMAGE_SIZE = 384
RESIZE_SIZE = 420

NUM_CLASSES = 120

BATCH_SIZE = 48
NUM_WORKERS = 8

CLASSIFIER_EPOCHS = 5
PARTIAL_FINE_TUNE_EPOCHS = 8
FINE_TUNE_EPOCHS = 20

PARTIAL_FINE_TUNE_BLOCKS = 3

CLASSIFIER_LEARNING_RATE = 1e-3
PARTIAL_FINE_TUNE_BACKBONE_LEARNING_RATE = 3e-5
PARTIAL_FINE_TUNE_CLASSIFIER_LEARNING_RATE = 1e-4
FINE_TUNE_BACKBONE_LEARNING_RATE = 1e-5
FINE_TUNE_CLASSIFIER_LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-4
GRADIENT_CLIP_NORM = 1.0
RANDOM_SEED = 42

REQUESTED_DEVICE = os.getenv("DOGBREED_DEVICE", "").strip().lower()

if REQUESTED_DEVICE == "cpu":
    DEVICE = torch.device("cpu")
elif REQUESTED_DEVICE == "cuda" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

USE_MIXED_PRECISION = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
PERSISTENT_WORKERS = NUM_WORKERS > 0
USE_TEST_TIME_AUGMENTATION = (
    os.getenv("DOGBREED_TTA", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DOG_REJECTION_ENABLED = (
    os.getenv("DOGBREED_DOG_REJECTION", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DOG_REJECTION_THRESHOLD = float(
    os.getenv("DOGBREED_DOG_THRESHOLD", "0.25")
)
GRADCAM_ENABLED = (
    os.getenv("DOGBREED_GRADCAM", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
