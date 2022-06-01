"""Config settings."""
from dataclasses import dataclass
from pathlib import Path

from .labels import labels

DATA_DIRECTORY = Path("/code/data")
DATASET_DIRECTORY = DATA_DIRECTORY / "dataset"
MODELS_DIRECTORY = DATA_DIRECTORY / "models"


@dataclass
class Config:
    """Config class."""

    # Paths
    raw_videos_directory: Path = Path("/data/speedtrap/videos")
    images_directory: Path = DATASET_DIRECTORY / "images"
    annotations_directory: Path = DATASET_DIRECTORY / "annotations"
    predictions_directory: Path = DATASET_DIRECTORY / "predictions"

    pretrained_model_path: Path = MODELS_DIRECTORY / "pretrained.pth"
    trained_model_path: Path =  MODELS_DIRECTORY / "model.pth"
    onnx_model_path: Path = MODELS_DIRECTORY / "model.onnx"

    # Dataset
    batch_size: int = 4
    n_classes: int = 91
    n_workers: int = 4
    min_size: int = 800
    darwin_dataset_slug: str = "retinanet-detections"
    labels: tuple[str, ...] = labels

    # Training
    max_epochs: int = 100
    gpus: int = 1
    learning_rate: float = 0.00025

    # Model
    trainable_backbone_layers: int = 0
