import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RetinaNetConfig:
    # Paths
    data_directory: str = os.getenv("DATA_DIR", "/code/data")
    _dataset_directory: Optional[str] = None
    _images_directory: Optional[str] = None
    _annotations_directory: Optional[str] = None
    _pretrained_model_path: Optional[str] = None
    _trained_model_path: Optional[str] = None

    # Dataset
    batch_size: int = 24
    n_classes: int = 1
    n_workers: int = 4
    image_size: Tuple[int, int] = 512, 512
    min_size: int = 512

    # Training
    max_epochs: int = 100
    gpus: int = 1
    learning_rate: float = 0.00025

    @property
    def dataset_directory(self) -> str:
        if self._dataset_directory:
            return self._dataset_directory
        return os.path.join(self.data_directory, "dataset")

    @property
    def images_directory(self) -> str:
        if self._images_directory:
            return self._images_directory
        return os.path.join(self.dataset_directory, "images")

    @property
    def annotations_directory(self) -> str:
        if self._annotations_directory:
            return self._annotations_directory
        return os.path.join(self.dataset_directory, "annotations")

    @property
    def pretrained_model_path(self) -> str:
        if self._pretrained_model_path:
            return self._pretrained_model_path
        return os.path.join(self.data_directory, "models/pretrained.pth")

    @property
    def trained_model_path(self) -> str:
        if self._trained_model_path:
            return self._trained_model_path
        return os.path.join(self.data_directory, "models/model.pth")


DefaultConfig = RetinaNetConfig()
