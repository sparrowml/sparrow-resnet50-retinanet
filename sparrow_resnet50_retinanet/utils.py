from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from sparrow_datums import FrameAugmentedBoxes, PType
from sparrow_tracky import MODA, compute_moda_by_class


def to_numpy(x: torch.Tensor) -> npt.NDArray[np.float64]:
    return x.detach().cpu().numpy()


def result_to_boxes(
    result: dict[str, torch.Tensor],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> FrameAugmentedBoxes:
    box_data = to_numpy(result["boxes"]).astype("float64")
    labels = to_numpy(result["labels"]).astype("float64")
    if "scores" in result:
        scores = to_numpy(result["scores"]).astype("float64")
    else:
        scores = np.ones(len(labels))
    return FrameAugmentedBoxes(
        np.concatenate([box_data, scores[:, None], labels[:, None]], -1),
        ptype=PType.absolute_tlwh,
        image_width=image_width,
        image_height=image_height,
    )


def batch_moda(
    results: list[dict[str, torch.Tensor]],
    batch: list[dict[str, torch.Tensor]],
) -> MODA:
    moda = MODA()
    for result, sample in zip(results, batch):
        predicted_boxes = result_to_boxes(result)
        ground_truth_boxes = result_to_boxes(sample)
        moda_dict = compute_moda_by_class(predicted_boxes, ground_truth_boxes)
        moda += sum(moda_dict.values())
    return moda
