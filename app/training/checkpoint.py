from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    config: dict,
    class_weights: dict,
    val_metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": config,
            "class_weights": class_weights,
            "val_metrics": val_metrics,
        },
        path,
    )
