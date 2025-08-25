#!/usr/bin/env python3
"""
Enhanced CNN Training Script for Satellite Imagery
Improvements over train_from_annotations.py:
- Advanced model architectures with regularization
- Data augmentation pipeline
- Learning rate scheduling
- Class imbalance handling
- Better normalization and preprocessing
- Early stopping and model checkpointing
"""

import argparse
import os
import sys
import csv
import time
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced CNN training for satellite imagery")
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "frontend", "training_data"),
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--datasets",
        default="DTM,Open_Positive",
        help="Comma-separated dataset folder names",
    )
    parser.add_argument("--train-csv", default="train_annotations.csv")
    parser.add_argument("--val-csv", default="valid_annotations.csv")
    parser.add_argument("--img-size", type=int, default=128, help="Increased default size")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100, help="More epochs with early stopping")
    parser.add_argument("--lr", type=float, default=3e-4, help="Lower initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--model-type", choices=["resnet", "efficientnet", "custom"], default="custom")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup augmentation alpha")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument(
        "--bands",
        default="",
        help="Comma-separated band folder names to stack as channels",
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Include base RGB as extra channels",
    )
    return parser.parse_args()


def read_annotations(dataset_dir: str, csv_name: str, validate_paths: bool = True) -> List[Tuple[str, Tuple[int, int, int, int], str]]:
    """Read annotations with improved error handling"""
    csv_path = os.path.join(dataset_dir, csv_name)
    samples: List[Tuple[str, Tuple[int, int, int, int], str]] = []
    
    if not os.path.exists(csv_path):
        if validate_paths:
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
        return samples
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row or len(row) < 6:
                continue
            
            img_rel = row[0]
            try:
                xmin = int(float(row[1]))
                ymin = int(float(row[2]))
                xmax = int(float(row[3]))
                ymax = int(float(row[4]))
            except Exception:
                continue
                
            label = row[5].strip() if len(row) > 5 else "unknown"
            img_path = os.path.join(dataset_dir, img_rel)
            
            if validate_paths and not os.path.exists(img_path):
                print(f"Warning: Missing file at row {row_idx+1}: {img_path}")
                continue
            
            samples.append((img_path, (xmin, ymin, xmax, ymax), label))
    
    return samples


class EnhancedBBoxDataset(Dataset):
    """Enhanced dataset with augmentation and normalization"""
    
    def __init__(
        self,
        samples: List[Tuple[str, Tuple[int, int, int, int], str]],
        class_to_idx: Dict[str, int],
        img_size: int,
        data_root: str,
        band_names: List[str],
        include_rgb: bool,
        augment: bool = False,
        is_training: bool = True,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.img_size = img_size
        self.data_root = data_root
        self.band_names = band_names
        self.include_rgb = include_rgb
        self.augment = augment and is_training
        
        # Calculate normalization statistics from a subset of data
        self.mean, self.std = self._calculate_stats()
        
        # Base transforms
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
        
        # Training augmentations
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.augment_transform = None
        
        self.base_transform = transforms.Compose(base_transforms)
        
    def _calculate_stats(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate dataset statistics for normalization"""
        # Sample a subset of data to calculate mean and std
        sample_indices = np.random.choice(len(self.samples), min(num_samples, len(self.samples)), replace=False)
        
        pixel_sum = torch.zeros(3 if not self.band_names else len(self.band_names) + (3 if self.include_rgb else 0))
        pixel_sq_sum = torch.zeros_like(pixel_sum)
        pixel_count = 0
        
        for idx in sample_indices:
            try:
                img_tensor, _ = self._load_sample(idx, apply_augment=False)
                pixel_sum += img_tensor.sum(dim=(1, 2))
                pixel_sq_sum += (img_tensor ** 2).sum(dim=(1, 2))
                pixel_count += img_tensor.shape[1] * img_tensor.shape[2]
            except Exception:
                continue
        
        if pixel_count > 0:
            mean = pixel_sum / pixel_count
            std = torch.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
            # Prevent division by zero
            std = torch.clamp(std, min=1e-6)
        else:
            # Default values if calculation fails
            mean = torch.tensor([0.5] * pixel_sum.shape[0])
            std = torch.tensor([0.5] * pixel_sum.shape[0])
        
        return mean, std
    
    def _load_sample(self, idx: int, apply_augment: bool = True):
        """Load and preprocess a single sample"""
        img_path, (xmin, ymin, xmax, ymax), label = self.samples[idx]
        
        try:
            base_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Create placeholder image if loading fails
            base_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        
        # Crop to bounding box
        w, h = base_img.size
        xmin_c = max(0, min(int(xmin), w - 1))
        ymin_c = max(0, min(int(ymin), h - 1))
        xmax_c = max(xmin_c + 1, min(int(xmax), w))
        ymax_c = max(ymin_c + 1, min(int(ymax), h))
        
        cropped_img = base_img.crop((xmin_c, ymin_c, xmax_c, ymax_c))
        
        # Apply augmentations if training
        if apply_augment and self.augment_transform:
            cropped_img = self.augment_transform(cropped_img)
        
        # Build channel stack
        tensors: List[torch.Tensor] = []
        
        # Multi-band stack
        if self.band_names:
            rel_idx = img_path.rfind(os.sep + "images" + os.sep)
            rel_tail = img_path[rel_idx + 1:] if rel_idx != -1 else os.path.basename(img_path)
            
            for band in self.band_names:
                band_img_path = os.path.join(self.data_root, band, rel_tail)
                if os.path.exists(band_img_path):
                    try:
                        band_img = Image.open(band_img_path).convert("L")
                        band_img = band_img.crop((xmin_c, ymin_c, xmax_c, ymax_c))
                    except Exception:
                        band_img = cropped_img.convert("L")
                else:
                    band_img = Image.new("L", cropped_img.size, color=0)
                
                band_tensor = self.base_transform(band_img)
                tensors.append(band_tensor)
        
        # Include RGB channels
        if self.include_rgb or not self.band_names:
            rgb_tensor = self.base_transform(cropped_img)
            tensors.append(rgb_tensor)
        
        # Concatenate channels
        img_tensor = torch.cat(tensors, dim=0)
        
        # Normalize
        img_tensor = (img_tensor - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        
        target = self.class_to_idx[label]
        return img_tensor, target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        return self._load_sample(idx, apply_augment=True)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = F.relu(out)
        
        return out


class AttentionModule(nn.Module):
    """Channel and spatial attention module"""
    
    def __init__(self, channels: int):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.ReLU(),
            nn.Linear(channels // 8, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))
        channel_att = (avg_out + max_out).view(batch_size, channels, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class EnhancedCNN(nn.Module):
    """Enhanced CNN with residual connections, attention, and regularization"""
    
    def __init__(self, num_classes: int, in_channels: int, dropout: float = 0.3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, dropout=dropout),
            ResidualBlock(64, 64, dropout=dropout)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=dropout),
            ResidualBlock(128, 128, dropout=dropout)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout=dropout),
            ResidualBlock(256, 256, dropout=dropout)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, dropout=dropout),
            ResidualBlock(512, 512, dropout=dropout)
        )
        
        # Attention modules
        self.attention1 = AttentionModule(128)
        self.attention2 = AttentionModule(256)
        self.attention3 = AttentionModule(512)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer classifier with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention1(x)
        x = self.layer3(x)
        x = self.attention2(x)
        x = self.layer4(x)
        x = self.attention3(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_epoch(model, loader, criterion, optimizer, device, args, epoch):
    """Train for one epoch with mixup and gradient clipping"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Mixup augmentation
        if args.mixup_alpha > 0 and np.random.random() > 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, targets, args.mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            if args.label_smoothing > 0:
                # Label smoothing
                num_classes = outputs.size(1)
                smooth_targets = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
                smooth_targets = smooth_targets * (1 - args.label_smoothing) + args.label_smoothing / num_classes
                loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * smooth_targets, dim=1))
            else:
                loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        
        # For mixup, use original targets for accuracy calculation
        if args.mixup_alpha > 0 and 'lam' in locals():
            correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
        else:
            correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")
    
    return running_loss / total, correct / total


def validate_epoch(model, loader, criterion, device):
    """Validate model performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy tracking
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Track per-class accuracy
            for target, pred in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                if target not in class_total:
                    class_total[target] = 0
                    class_correct[target] = 0
                class_total[target] += 1
                if target == pred:
                    class_correct[target] += 1
    
    # Calculate per-class accuracies
    per_class_acc = {cls: class_correct[cls] / class_total[cls] 
                     for cls in class_total if class_total[cls] > 0}
    
    return running_loss / total, correct / total, per_class_acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    band_names = [b.strip() for b in args.bands.split(",") if b.strip()]
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    from train_from_annotations import collect_dataset
    train_samples, val_samples, class_to_idx = collect_dataset(
        args.data_root, datasets, args.train_csv, args.val_csv, validate_paths=True
    )
    
    if len(train_samples) == 0:
        print("No training samples found.")
        sys.exit(1)
    
    print(f"Classes ({len(class_to_idx)}): {sorted(class_to_idx.keys())}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")
    
    # Calculate class weights for imbalanced data
    train_labels = [class_to_idx[label] for _, _, label in train_samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create datasets
    train_ds = EnhancedBBoxDataset(
        train_samples, class_to_idx, args.img_size, args.data_root,
        band_names, args.include_rgb, augment=args.augment, is_training=True
    )
    val_ds = EnhancedBBoxDataset(
        val_samples, class_to_idx, args.img_size, args.data_root,
        band_names, args.include_rgb, augment=False, is_training=False
    ) if val_samples else None
    
    # Create weighted sampler for balanced training
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    # Determine input channels
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    num_classes = len(class_to_idx)
    
    # Create model
    if args.model_type == "custom":
        model = EnhancedCNN(num_classes=num_classes, in_channels=in_channels, dropout=0.3)
    else:
        # Could add ResNet or EfficientNet variants here
        model = EnhancedCNN(num_classes=num_classes, in_channels=in_channels, dropout=0.3)
    
    model = model.to(device)
    
    # Loss function - use Focal Loss for imbalanced data
    criterion = FocalLoss(alpha=1.0, gamma=2.0) if len(class_weights) > 2 else nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args, epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        if val_loader:
            val_loss, val_acc, per_class_acc = validate_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            print(f"Per-class Val Acc: {per_class_acc}")
            
            # Save checkpoint if best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                    'class_to_idx': class_to_idx,
                    'img_size': args.img_size,
                    'num_classes': num_classes,
                    'in_channels': in_channels,
                    'history': history,
                }, checkpoint_path)
                print(f"✓ Saved best model checkpoint (val_acc: {best_val_acc:.3f})")
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
    
    # Save final model
    out_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.model_out or os.path.join(out_dir, f"enhanced_cnn_{ts}.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'img_size': args.img_size,
        'num_classes': num_classes,
        'in_channels': in_channels,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'history': history,
        'args': vars(args),
    }, out_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_path}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()