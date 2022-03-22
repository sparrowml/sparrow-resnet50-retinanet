import os

from .config import DefaultConfig, RetinaNetConfig


def test_overriding_full_path():
    path = "foo/models/bar.txt"
    config = RetinaNetConfig(_pretrained_model_path=path)
    assert str(config.pretrained_model_path) == path


def test_default_data_directory_is_absolute_path():
    assert os.path.isabs(DefaultConfig.data_directory)
