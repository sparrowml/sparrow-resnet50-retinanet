import os

import torch

from .dataset import Holdout, RetinaNetDataset, get_holdout_slugs, get_sample_dicts


def test_train_set_is_roughly_80_percent():
    all_samples = get_holdout_slugs()
    train_samples = get_holdout_slugs(Holdout.train)
    assert 0.75 < len(train_samples) / len(all_samples) < 0.85, "Bad sampling"


def test_dev_set_is_roughly_10_percent():
    all_samples = get_holdout_slugs()
    dev_samples = get_holdout_slugs(Holdout.dev)
    assert 0.05 < len(dev_samples) / len(all_samples) < 0.15, "Bad sampling"


def test_test_set_is_roughly_10_percent():
    all_samples = get_holdout_slugs()
    test_samples = get_holdout_slugs(Holdout.test)
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
