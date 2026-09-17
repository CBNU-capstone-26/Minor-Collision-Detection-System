import torch

from model.hitandrun_model import HitAndRun3DCNN


def test_s3d_model_contract():
    model = HitAndRun3DCNN(num_classes=2, pretrained=False)
    assert hasattr(model, "features")
    assert hasattr(model, "inception5b")
    assert model(torch.zeros(1, 3, 30, 224, 224)).shape == (1, 2)
