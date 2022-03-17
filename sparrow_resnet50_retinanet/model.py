from typing import List, Optional, Tuple, Union

import torch
from torchvision import models

from .config import DefaultConfig
from .types import TensorDict


class RetinaNet(torch.nn.Module):
    def __init__(
        self,
        pretrained: bool = DefaultConfig.pretrained,
        n_classes: int = DefaultConfig.n_classes,
        min_size: int = DefaultConfig.min_size,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model = models.detection.retinanet_resnet50_fpn(
            progress=False,
            pretrained=pretrained,
            num_classes=n_classes,
            min_size=min_size,
        )

    def forward(
        self, x: List[torch.Tensor], y: Optional[List[TensorDict]] = None
    ) -> Union[TensorDict, List[TensorDict]]:
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
        return self.model.forward(x, y)

    def load(self, model_path: str, skip_classes: bool = False) -> None:
        weights = torch.load(model_path)
        if skip_classes:
            del weights["model.head.classification_head.cls_logits.weight"]
            del weights["model.head.classification_head.cls_logits.bias"]
        strict = not skip_classes
        return self.load_state_dict(weights, strict=strict)


def save_pretrained(
    pretrained_model_path: str = str(DefaultConfig.pretrained_model_path),
) -> None:
    model = RetinaNet(pretrained=True, n_classes=91)
    torch.save(model.state_dict(), pretrained_model_path)


def export(
    output_path: str,
    input_shape: Tuple[int, int, int] = (3, 512, 512),
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
