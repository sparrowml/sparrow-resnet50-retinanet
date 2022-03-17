from typing import List, Optional

import os
from dataclasses import dataclass
from pathlib import Path

from .labels import labels


@dataclass
class RetinaNetConfig:
    # Paths
    data_directory: Path = Path(os.getenv("DATA_DIR", "/code/data"))
    _raw_videos_directory: Optional[str] = None
    _dataset_directory: Optional[str] = None
    _images_directory: Optional[str] = None
    _annotations_directory: Optional[str] = None
    _pretrained_model_path: Optional[str] = None
    _trained_model_path: Optional[str] = None

    # Dataset
    batch_size: int = 8
    n_classes: int = 91
    n_workers: int = 4
    min_size: int = 800

    # Training
    max_epochs: int = 100
    gpus: int = 1
    learning_rate: float = 0.00025

    # Model
    pretrained: bool = False

    @property
    def raw_videos_directory(self) -> Path:
        if self._raw_videos_directory:
            return Path(self._raw_videos_directory)
        return Path("/data/speedtrap/videos")

    @property
    def dataset_directory(self) -> Path:
        if self._dataset_directory:
            return Path(self._dataset_directory)
        return self.data_directory / "dataset"

    @property
    def images_directory(self) -> Path:
        if self._images_directory:
            return Path(self._images_directory)
        return self.dataset_directory / "images"

    @property
    def annotations_directory(self) -> Path:
        if self._annotations_directory:
            return Path(self._annotations_directory)
        return self.dataset_directory / "annotations"

    @property
    def pretrained_model_path(self) -> Path:
        if self._pretrained_model_path:
            return Path(self._pretrained_model_path)
        return self.data_directory / "models/pretrained.pth"

    @property
    def trained_model_path(self) -> Path:
        if self._trained_model_path:
            return Path(self._trained_model_path)
        return self.data_directory / "models/model.pth"

    @property
    def labels(self) -> List[str]:
        return labels


DefaultConfig = RetinaNetConfig()
