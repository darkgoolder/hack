import torch
from hack.app.training.losses import masked_bce_with_logits


def test_masked_bce_ignores_missing_labels():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([0.0, 1.0])
    mask = torch.tensor([1.0, 0.0])

    loss = masked_bce_with_logits(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0]), torch.tensor([0.0])
    )
    assert torch.allclose(loss, expected)
