import torch

from app.models.dxa_multitask import build_model


def test_model_output_shapes():
    model = build_model(pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        out = model(x)

    assert out.anatomy_logits.shape == (2, 3)
    assert out.spine_logits.shape == (2, 3)
    assert out.hip_logits.shape == (2, 2)
    assert out.quality_logits.shape == (2, 1)
    assert out.features.shape[0] == 2


def test_cam_layer_exists():
    model = build_model(pretrained=False)
    assert model.cam_target_layer() is model.backbone.layer4[-1]
