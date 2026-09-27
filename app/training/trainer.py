from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from app.data.dicom_dataset import DXAQualityDataset
from app.models.dxa_multitask import build_model
from app.training.checkpoint import save_checkpoint
from app.training.class_weights import compute_weights
from app.training.losses import multitask_loss
from app.training.metrics import anatomy_metrics, summarize_quality_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key in ("image", "anatomy", "targets", "target_mask"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def create_loaders(
    train_csv: str | Path,
    val_csv: str | Path,
    *,
    batch_size: int,
    image_size: int,
    num_workers: int,
    root: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = DXAQualityDataset(
        train_csv, image_size=image_size, training=True, root=root,
    )
    val_ds = DXAQualityDataset(
        val_csv, image_size=image_size, training=False, root=root,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, drop_last=False, **common),
    )


def _autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _use_grad_scaler(device: torch.device) -> bool:
    return device.type == "cuda" and not torch.cuda.is_bf16_supported()


def _optimizer(model, head_lr: float, backbone_lr: float, weight_decay: float):
    backbone, heads = [], []
    for name, p in model.named_parameters():
        (backbone if name.startswith("backbone.") else heads).append(p)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": heads, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def _weights_on_device(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    keys = [
        "spine_pos_weight", "hip_pos_weight",
        "quality_pos_weight", "anatomy_class_weight",
    ]
    return {key: raw[key].to(device) for key in keys}


def _loss(model, batch, weights):
    output = model(batch["image"])
    loss_dict = multitask_loss(
        output,
        batch["targets"],
        batch["target_mask"],
        batch["anatomy"],
        spine_pos_weight=weights["spine_pos_weight"],
        hip_pos_weight=weights["hip_pos_weight"],
        quality_pos_weight=weights["quality_pos_weight"],
        anatomy_class_weight=weights["anatomy_class_weight"],
    )
    return output, loss_dict


def train_one_epoch(model, loader, optimizer, weights, device, scaler=None):
    model.train()
    sums = {k: 0.0 for k in ("loss", "anatomy_loss", "spine_loss", "hip_loss", "quality_loss")}
    batches = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device):
            _, losses = _loss(model, batch, weights)
            loss = losses["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for key in sums:
            sums[key] += float(losses[key].item())
        batches += 1

    return {key: value / max(batches, 1) for key, value in sums.items()}


@torch.no_grad()
def validate(model, loader, weights, device):
    model.eval()
    sums = {k: 0.0 for k in ("loss", "anatomy_loss", "spine_loss", "hip_loss", "quality_loss")}
    batches = 0

    anatomy_true, anatomy_logits = [], []
    targets, masks, probabilities = [], [], []

    for batch in loader:
        batch = move_batch(batch, device)
        with _autocast(device):
            output, losses = _loss(model, batch, weights)

        for key in sums:
            sums[key] += float(losses[key].item())
        batches += 1

        anatomy_true.append(batch["anatomy"].cpu().numpy())
        anatomy_logits.append(output.anatomy_logits.float().cpu().numpy())
        targets.append(batch["targets"].cpu().numpy())
        masks.append(batch["target_mask"].cpu().numpy())

        logits = torch.cat(
            [output.spine_logits, output.hip_logits, output.quality_logits], dim=1
        )
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())

    y_anatomy = np.concatenate(anatomy_true)
    a_logits = np.concatenate(anatomy_logits)
    y_targets = np.concatenate(targets)
    y_masks = np.concatenate(masks)
    y_prob = np.concatenate(probabilities)

    return {
        "losses": {key: value / max(batches, 1) for key, value in sums.items()},
        "anatomy": anatomy_metrics(y_anatomy, a_logits),
        "quality": summarize_quality_metrics(y_targets, y_masks, y_prob, y_anatomy),
    }


def fit(
    train_csv: str | Path,
    val_csv: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 25,
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 0,
    seed: int = 42,
    pretrained: bool = True,
    freeze_backbone_epochs: int = 3,
    head_lr: float = 5e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    patience: int = 7,
    root: str | Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_loaders(
        train_csv, val_csv,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        root=root,
    )

    model = build_model(pretrained=pretrained).to(device)
    if freeze_backbone_epochs > 0:
        model.freeze_backbone()

    optimizer = _optimizer(model, head_lr, backbone_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )

    raw_weights = compute_weights(train_csv)
    weights = _weights_on_device(raw_weights, device)
    scaler = torch.amp.GradScaler("cuda") if _use_grad_scaler(device) else None

    config = {
        "train_csv": str(train_csv), "val_csv": str(val_csv),
        "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
        "num_workers": num_workers, "seed": seed, "pretrained": pretrained,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "head_lr": head_lr, "backbone_lr": backbone_lr,
        "weight_decay": weight_decay, "patience": patience,
        "device": str(device),
    }
    serializable_weights = {
        key: value.cpu().tolist() if torch.is_tensor(value) else value
        for key, value in raw_weights.items()
        if key != "counts"
    }
    serializable_weights["counts"] = raw_weights["counts"]

    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "class_weights.json").write_text(
        json.dumps(serializable_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            model.unfreeze_backbone()

        started = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, weights, device, scaler)
        val_metrics = validate(model, val_loader, weights, device)
        val_loss = val_metrics["losses"]["loss"]
        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_heads": optimizer.param_groups[1]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        save_checkpoint(
            output_dir / "last.pt",
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_loss=min(best_val_loss, val_loss),
            config=config, class_weights=serializable_weights,
            val_metrics=val_metrics,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                output_dir / "best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_loss=best_val_loss,
                config=config, class_weights=serializable_weights,
                val_metrics=val_metrics,
            )
        else:
            stale += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_loss:.4f} | "
            f"anatomy_F1={val_metrics['anatomy']['macro_f1']:.4f} | "
            f"detail_F1={val_metrics['quality']['detail_macro_f1']}"
        )

        if stale >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "device": str(device),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
