# type: ignore[attr-defined]
"""A baseline ResNet50 RetinaNet model"""
from .config import RetinaNetConfig, DefaultConfig
from .dataset import sample_frames
from .inference import import_predictions, run_predictions
from .model import RetinaNet, save_pretrained, export

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
