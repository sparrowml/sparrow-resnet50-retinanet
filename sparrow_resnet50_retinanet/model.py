from typing import Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sparrow_datums import FrameAugmentedBoxes, PType
from torchvision import models

from .config import Config
from .utils import result_to_boxes


class RetinaNet(torch.nn.Module):
    """RetinaNet detector."""

    def __init__(
        self,
        n_classes: int = Config.n_classes,
        min_size: int = Config.min_size,
        trainable_backbone_layers: int = Config.trainable_backbone_layers,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model = models.detection.retinanet_resnet50_fpn(
            progress=False,
            pretrained=False,
            num_classes=n_classes,
            min_size=min_size,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    def forward(
        self, x: list[torch.Tensor], y: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> Union[
        dict[str, torch.Tensor], list[dict[str, torch.Tensor]], FrameAugmentedBoxes
    ]:
        """
        Forward pass for training and inference

        Parameters
        ----------
        x
            A list of image tensors with shape (3, n_rows, n_cols) with
            unnormalized values in [0, 1].
        y
            An optional list of targets with an x1, x2, y1, y2 "boxes" tensor
            and a class index "labels" tensor.

        Returns
        -------
        result(s)
            If inference, this will be a list of dicts with predicted tensors
            for "boxes", "scores" and "labels" in each one. If training, this
            will be a dict with loss tensors for "classification" and
            "bbox_regression".
        """
        if isinstance(x, np.ndarray):
            return self.forward_numpy(x)
        return self.model.forward(x, y)

    def forward_numpy(self, x: np.ndarray) -> FrameAugmentedBoxes:
        image_height, image_width = x.shape[:2]
        x = T.ToTensor()(Image.fromarray(x))
        if torch.cuda.is_available():
            x = x.cuda()
        result = self.forward([x])[0]
        return result_to_boxes(
            result,
            image_width=image_width,
            image_height=image_height,
        )

    def load(self, model_path: str, skip_classes: bool = False) -> None:
        weights = torch.load(model_path)
        if skip_classes:
            del weights["model.head.classification_head.cls_logits.weight"]
            del weights["model.head.classification_head.cls_logits.bias"]
        strict = not skip_classes
        return self.load_state_dict(weights, strict=strict)


def save_pretrained(
    pretrained_model_path: str = str(Config.pretrained_model_path),
) -> None:
    model = RetinaNet(pretrained=True, n_classes=91)
    torch.save(model.state_dict(), pretrained_model_path)


def export(
    output_path: str,
    input_shape: tuple[int, int, int] = (3, 512, 512),
) -> None:
    x = torch.randn(*input_shape)
    model = RetinaNet().eval()
    print(f"Input shape: {input_shape}")
    torch.onnx.export(
        model,
        [x],
        output_path,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        opset_version=11,
    )
