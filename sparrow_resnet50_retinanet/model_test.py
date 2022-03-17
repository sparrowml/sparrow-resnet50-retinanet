import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch

from .config import DefaultConfig
from .model import RetinaNet, export


def test_retinanet_inference_works():
    model = RetinaNet().eval()
    (result,) = model([torch.randn((3, 32, 32))])
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert isinstance(labels, torch.Tensor)


@pytest.mark.skipif(os.getenv("FAST") == "1", reason="Skip slow tests")
def test_retinanet_compiles_to_valid_onnx():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.onnx")
        export(path, input_shape=(3, 32, 32))
        inference_session = ort.InferenceSession(path)
        x = np.random.randn(3, 32, 32).astype("float32")
        boxes, scores, labels = inference_session.run(
            ["boxes", "scores", "labels"], {"input": x}
        )
        assert isinstance(boxes, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(labels, np.ndarray)


def test_retinanet_can_load_pretrained_model():
    model = RetinaNet()
    status = model.load(DefaultConfig.pretrained_model_path, skip_classes=True)
    assert "missing_keys" in str(status)
