from typing import Any, Optional

import enum
import os
import random
from operator import itemgetter
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sparrow_datums import FrameAugmentedBoxes
from tqdm import tqdm

from .config import Config


class Holdout(enum.Enum):
    train = "train"
    dev = "dev"
    test = "test"


def sample_frames() -> None:
    video_paths = list(Config.raw_videos_directory.glob("*.mp4"))
    for raw_video in tqdm(video_paths):
        slug, _ = os.path.splitext(raw_video.name)
        reader = imageio.get_reader(raw_video)
        fps, duration = itemgetter("fps", "duration")(reader.get_meta_data())
        total_frames = int(fps * duration)
        for frame_index in range(0, total_frames, round(fps)):
            image_name = f"{slug}_{frame_index:05d}.jpg"
            image_path = Config.images_directory / image_name
            try:
                image = reader.get_data(frame_index)
            except (IndexError, OSError):
                break
            imageio.imwrite(image_path, image)


def version_annotations(darwin_annotations_directory: str) -> None:
    for darwin_path in Path(darwin_annotations_directory).glob("*.json"):
        boxes = FrameAugmentedBoxes.from_darwin_file(darwin_path, Config.labels)
        slug, _ = os.path.splitext(darwin_path.name)
        annotation_filename = f"{slug}.json.gz"
        annotation_path = Config.annotations_directory / annotation_filename
        boxes.to_file(annotation_path)


def get_holdout_slugs(
    holdout: Optional[Holdout] = None,
) -> list[str]:
    image_ids = [
        p.name.split(".")[0] for p in Config.annotations_directory.glob("*.json.gz")
    ]
    if holdout == Holdout.train:
        image_ids = set(filter(lambda id: hash(id) % 10 < 8, image_ids))
    if holdout == Holdout.dev:
        image_ids = set(filter(lambda id: hash(id) % 10 == 8, image_ids))
    if holdout == Holdout.test:
        image_ids = set(filter(lambda id: hash(id) % 10 == 9, image_ids))
    return image_ids


def get_sample_dicts(
    holdout: Optional[Holdout] = None,
    sample_size: Optional[Holdout] = None,  # For testing
) -> list[dict[str, Any]]:
    slugs = get_holdout_slugs(holdout)
    if sample_size is not None:
        random.shuffle(slugs)
        slugs = slugs[:sample_size]

    samples = []

    for slug in slugs:
        image_path = Config.images_directory / f"{slug}.jpg"
        annotation_path = Config.annotations_directory / f"{slug}.json.gz"
        boxes: FrameAugmentedBoxes = (
            FrameAugmentedBoxes.from_file(annotation_path).to_absolute().to_tlbr()
        )
        samples.append(
            {
                "image_path": str(image_path),
                "boxes": boxes.array[:, :4],
                "labels": boxes.labels,
            }
        )
    return samples


class RetinaNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        holdout: Optional[Holdout] = None,
        sample_size: Optional[Holdout] = None,  # For testing
    ) -> None:
        self.samples = get_sample_dicts(holdout, sample_size)
        self.transform = T.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"])
        x = self.transform(img)
        boxes = sample["boxes"].astype("float32")
        return {
            "image": x,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(sample["labels"]),
        }
