import os

import torch

from .config import DefaultConfig
from .dataset import get_image_ids, get_sample_dicts, Holdout, RetinaNetDataset


def test_train_set_is_roughly_80_percent():
    all_samples = get_image_ids()
    train_samples = get_image_ids(Holdout.TRAIN)
    assert 0.75 < len(train_samples) / len(all_samples) < 0.85, "Bad sampling"


def test_dev_set_is_roughly_10_percent():
    all_samples = get_image_ids()
    dev_samples = get_image_ids(Holdout.DEV)
    assert 0.05 < len(dev_samples) / len(all_samples) < 0.15, "Bad sampling"


def test_test_set_is_roughly_10_percent():
    all_samples = get_image_ids()
    test_samples = get_image_ids(Holdout.TEST)
    assert 0.05 < len(test_samples) / len(all_samples) < 0.15, "Bad sampling"


def test_sample_dicts_point_to_real_images():
    all_samples = get_sample_dicts(sample_size=10)
    assert os.path.exists(all_samples[0]["image_path"]), "Image doesn't exist"


def test_dataset_returns_tensor_dict():
    dataset = RetinaNetDataset(sample_size=10)
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["boxes"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
