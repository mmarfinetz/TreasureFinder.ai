"""Generic training script for segmentation models.

Provides optional focal loss or class-weighted cross entropy.
Logs IoU/F1 per epoch and saves the best model based on validation IoU.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.segmentation import iou, f1_score


class FocalLoss(nn.Module):
    """Focal loss for dense classification."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_p, targets, weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        p = torch.exp(-ce_loss)
        loss = ((1 - p) ** self.gamma) * ce_loss
        return loss.mean()


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
) -> None:
    """Train a segmentation model.

    Args:
        model: segmentation model producing logits of shape (N, C, H, W).
        train_loader: dataloader for training data.
        val_loader: dataloader for validation data.
        optimizer: optimizer instance.
        device: computation device.
        epochs: number of epochs.
        loss_type: 'ce', 'weighted_ce', or 'focal'.
        class_weights: optional weights for classes when using weighted loss.
    """

    model.to(device)
    if loss_type == "focal":
        criterion = FocalLoss(weight=class_weights)
    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_iou, val_f1 = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                iou_per_class, _ = iou(preds, masks, outputs.shape[1])
                f1_per_class, _ = f1_score(preds, masks, outputs.shape[1])
                val_iou.append(iou_per_class)
                val_f1.append(f1_per_class)

        mean_iou = torch.stack(val_iou).mean().item()
        mean_f1 = torch.stack(val_f1).mean().item()
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - IoU: {mean_iou:.4f} - F1: {mean_f1:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
