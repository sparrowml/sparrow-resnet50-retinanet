# type: ignore[attr-defined]
"""A baseline ResNet50 RetinaNet model"""
import warnings

from .config import Config
from .dataset import sample_frames, version_annotations
from .evaluate import evaluate_predictions
from .inference import import_predictions, run_predictions
from .model import RetinaNet, export, save_pretrained
from .train import save_checkpoint, train_model

warnings.filterwarnings("ignore", category=UserWarning)
