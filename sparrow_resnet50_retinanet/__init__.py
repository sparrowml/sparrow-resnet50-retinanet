# type: ignore[attr-defined]
"""A baseline ResNet50 RetinaNet model"""
from .config import Config
from .dataset import sample_frames, version_annotations
from .evaluate import evaluate_predictions
from .inference import import_predictions, run_predictions
from .model import RetinaNet, save_pretrained, export_model
from .train import train_model, save_checkpoint

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
