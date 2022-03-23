from dataclasses import dataclass
from pathlib import Path

from .labels import labels


@dataclass
class Config:
    # Paths
    data_directory: Path = Path("/code/data")
    raw_videos_directory: Path = Path("/data/speedtrap/videos")
    images_directory: Path = Path("/code/data/dataset/images")
    annotations_directory: Path = Path("/code/data/dataset/annotations")
    predictions_directory: Path = Path("/code/data/dataset/predictions")

    pretrained_model_path: Path = Path("/code/data/models/pretrained.pth")
    trained_model_path: Path = Path("/code/data/models/model.pth")

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
