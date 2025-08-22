import torch
from torch import Tensor
from typing import Optional, Tuple

def _confusion_matrix(pred: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None) -> Tensor:
    """Compute confusion matrix for segmentation.

    Args:
        pred: predictions of shape (N, H, W) with class indices.
        target: ground truth of shape (N, H, W) with class indices.
        num_classes: number of classes.
        ignore_index: optional index to ignore in the calculation.

    Returns:
        Tensor of shape (num_classes, num_classes) with counts.
    """
    if pred.ndim > 2:
        pred = pred.view(-1)
    if target.ndim > 2:
        target = target.view(-1)

    mask = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        mask &= target != ignore_index

    indices = num_classes * target[mask].to(torch.int64) + pred[mask]
    cm = torch.bincount(indices, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)

def iou(pred: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None) -> Tuple[Tensor, float]:
    """Compute per-class and mean Intersection over Union.

    Args:
        pred: predictions as class indices (N, H, W) or logits (N, C, H, W).
        target: ground truth class indices (N, H, W).
        num_classes: number of classes.
        ignore_index: optional class to ignore.

    Returns:
        Tuple of (per_class_iou, mean_iou).
    """
    if pred.ndim == target.ndim + 1:
        pred = pred.argmax(dim=1)

    cm = _confusion_matrix(pred, target, num_classes, ignore_index)
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    union = tp + fp + fn
    iou_per_class = tp.float() / (union.float() + 1e-7)
    mean_iou = iou_per_class.mean().item()
    return iou_per_class, mean_iou

def f1_score(pred: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None) -> Tuple[Tensor, float]:
    """Compute per-class and mean F1 score for segmentation."""
    if pred.ndim == target.ndim + 1:
        pred = pred.argmax(dim=1)

    cm = _confusion_matrix(pred, target, num_classes, ignore_index)
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
    mean_f1 = f1.mean().item()
    return f1, mean_f1

def per_class_recall(pred: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None) -> Tensor:
    """Compute recall for each class."""
    if pred.ndim == target.ndim + 1:
        pred = pred.argmax(dim=1)

    cm = _confusion_matrix(pred, target, num_classes, ignore_index)
    tp = cm.diag().float()
    fn = cm.sum(1).float() - tp
    recall = tp / (tp + fn + 1e-7)
    return recall
