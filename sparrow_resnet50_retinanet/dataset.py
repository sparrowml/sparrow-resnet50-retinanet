import os
from operator import itemgetter
from pathlib import Path

import imageio
from tqdm import tqdm

from .config import DefaultConfig


def sample_frames(
    raw_videos_directory: str = str(DefaultConfig.raw_videos_directory),
    frames_directory: str = "/data/darwin/sparrow-computing/retinanet-detections/images",
):
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
