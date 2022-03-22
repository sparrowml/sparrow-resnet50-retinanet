from typing import Optional

import os
import random
import tempfile

import imageio
from darwin import Client, importer
from sparrow_datums import FrameAugmentedBoxes
from tqdm import tqdm

from .config import Config
from .model import RetinaNet


def run_predictions(
    model_path: str = str(Config.pretrained_model_path),
    n_frames: Optional[int] = None,
    score_threshold: float = 0.5,
) -> None:
    model = RetinaNet().eval().cuda()
    model.load(model_path)
    image_paths = list(Config.images_directory.glob("*.jpg"))
    random.shuffle(image_paths)
    if n_frames is not None:
        image_paths = image_paths[:n_frames]
    for image_path in tqdm(image_paths):
        slug, _ = os.path.splitext(image_path.name)
        img = imageio.imread(image_path)
        boxes: FrameAugmentedBoxes = model(img)
        boxes = boxes[boxes.scores > score_threshold]
        json_filename = f"{slug}.json.gz"
        boxes.to_file(Config.predictions_directory / json_filename)


def import_predictions() -> None:
    client = Client.local()
    slug = Config.darwin_dataset_slug
    dataset = next(d for d in client.list_remote_datasets() if d.slug == slug)
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_paths = []
        for prediction_path in Config.predictions_directory.glob("*.json.gz"):
            slug = prediction_path.name.split(".")[0]
            boxes: FrameAugmentedBoxes = FrameAugmentedBoxes.from_file(prediction_path)
            annotation_path = os.path.join(tmpdir, f"{slug}.json")
            annotation_paths.append(annotation_path)
            boxes.to_darwin_annotation_file(
                annotation_path, f"{slug}.jpg", label_names=Config.labels
            )
        importer.import_annotations(
            dataset,
            importer.get_importer("darwin"),
            annotation_paths,
            append=True,
            class_prompt=False,
        )
