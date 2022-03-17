# type: ignore[attr-defined]
"""A baseline ResNet50 RetinaNet model"""
from .config import RetinaNetConfig, DefaultConfig
from .dataset import sample_frames
from .model import RetinaNet, save_pretrained, export
