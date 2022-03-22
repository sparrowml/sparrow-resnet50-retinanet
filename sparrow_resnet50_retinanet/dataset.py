import os
from operator import itemgetter
from pathlib import Path

import imageio
from sparrow_datums import FrameAugmentedBoxes
from tqdm import tqdm

from .config import DefaultConfig


def sample_frames(
    raw_videos_directory: str = str(DefaultConfig.raw_videos_directory),
    frames_directory: str = "/data/darwin/sparrow-computing/retinanet-detections/images",
) -> None:
    raw_videos_path = Path(raw_videos_directory)
    frames_path = Path(frames_directory)
    video_paths = list(raw_videos_path.glob("*.mp4"))
    for raw_video in tqdm(video_paths):
        slug, _ = os.path.splitext(raw_video.name)
        reader = imageio.get_reader(raw_video)
        fps, duration = itemgetter("fps", "duration")(reader.get_meta_data())
        total_frames = int(fps * duration)
        for frame_index in range(0, total_frames, round(fps)):
            image_path = frames_path / f"{slug}_{frame_index:05d}.jpg"
            try:
                image = reader.get_data(frame_index)
            except (IndexError, OSError):
                break
            imageio.imwrite(image_path, image)


def version_annotations(darwin_annotations_directory: str) -> None:
    for darwin_path in Path(darwin_annotations_directory).glob("*.json"):
        boxes = FrameAugmentedBoxes.from_darwin_file(darwin_path, DefaultConfig.labels)
        slug, _ = os.path.splitext(darwin_path.name)
        annotation_filename = f"{slug}.json.gz"
        annotation_path = DefaultConfig.annotations_directory / annotation_filename
        boxes.to_file(annotation_path)
