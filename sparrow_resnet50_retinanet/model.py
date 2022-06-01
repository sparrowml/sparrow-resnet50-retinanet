"""RetinaNet model."""
from typing import Any, Callable, Optional, Union, overload, no_type_check

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
import torchvision.ops.boxes as box_ops
from PIL import Image
from torchvision import models
from sparrow_datums import FrameAugmentedBoxes

from .config import Config
from .utils import result_to_boxes


class RetinaNet(torch.nn.Module):
    """RetinaNet detector."""

    def __init__(
        self,
        n_classes: int = Config.n_classes,
        min_size: int = Config.min_size,
        trainable_backbone_layers: int = Config.trainable_backbone_layers,
        optimized: bool = False,
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
        self.optimized = optimized
        if self.optimized:
            self.model.transform.normalize = lambda x: x
            self.model.transform.resize = lambda x, y: (x, y)
            self.model.transform.batch_images = lambda x, size_divisible: (
                x[0].unsqueeze(0)
            )
            self.model.postprocess_detections = self.postprocess_detections
            self.model.transform.postprocess = lambda x, y, z: x

    @overload
    def forward(self, x: npt.NDArray[np.float64]) -> FrameAugmentedBoxes:
        ...

    @overload
    def forward(
        self, x: list[torch.Tensor], y: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        ...

    @overload
    def forward(self, x: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        ...

    @overload
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(
        self,
        x: Union[torch.Tensor, npt.NDArray[np.float64], list[torch.Tensor]],
        y: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> Union[
        dict[str, torch.Tensor], list[dict[str, torch.Tensor]], FrameAugmentedBoxes
    ]:
        """
        Perform forward pass for training and inference.

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
        if self.optimized:
            x = [x[0]]
        result = self.model.forward(x, y)
        if self.optimized:
            return result[None]
        return result

    def forward_numpy(self, x: npt.NDArray[np.float64]) -> FrameAugmentedBoxes:
        """Forward inference for NumPy."""
        image_height, image_width = x.shape[:2]
        torch_x = T.ToTensor()(Image.fromarray(x))
        if torch.cuda.is_available():
            torch_x = torch_x.cuda()
        results: list[dict[str, torch.Tensor]] = self.forward([torch_x])
        return result_to_boxes(
            results[0],
            image_width=image_width,
            image_height=image_height,
        )

    def postprocess_detections(
        self,
        head_outputs: dict[str, list[torch.Tensor]],
        anchors: list[list[torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Optimized post-processing for detections."""
        class_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        boxes: list[torch.Tensor] = []
        classes: list[torch.Tensor] = [logits[0] for logits in class_logits]
        for level_bbox, level_anchors in zip(bbox_regression, anchors[0]):
            boxes.append(
                self.model.box_coder.decode_single(level_bbox[0], level_anchors)
            )
        boxes = torch.cat(boxes, dim=0)
        scores = torch.sigmoid(torch.cat(classes, dim=0)).max(-1)
        height, width = image_shapes[0]
        return torch.cat(
            [
                box_ops.clip_boxes_to_image(boxes, (height, width))
                / torch.tensor([width, height, width, height]),
                scores.values[:, None],  # scores
                scores.indices.to(torch.float32)[:, None],  # labels
            ],
            dim=-1,
        )

    def load(self, model_path: str, skip_classes: bool = False) -> None:
        """Load weights for model."""
        weights = torch.load(model_path)
        if skip_classes:
            del weights["model.head.classification_head.cls_logits.weight"]
            del weights["model.head.classification_head.cls_logits.bias"]
        strict = not skip_classes
        self.load_state_dict(weights, strict=strict)


def save_pretrained(
    pretrained_model_path: str = str(Config.pretrained_model_path),
) -> None:
    """Save the pretrained model."""
    model = RetinaNet(pretrained=True, n_classes=91)
    torch.save(model.state_dict(), pretrained_model_path)


def export_model(input_shape: tuple[int, int, int, int] = (1, 3, 512, 512)) -> None:
    """Export the model."""
    x = torch.randn(*input_shape)
    model = RetinaNet(optimized=True).eval()
    model.load(Config.trained_model_path)
    print(f"Input shape: {input_shape}")
    torch.onnx.export(
        model,
        x,
        Config.onnx_model_path,
        input_names=["input"],
        output_names=["augmented_boxes"],
        opset_version=11,
    )
    print("Version ONNX model:")
    print("dvc add /code/data/models/model.onnx")
