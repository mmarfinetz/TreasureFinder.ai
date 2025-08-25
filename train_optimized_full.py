#!/usr/bin/env python3
"""
Optimized full training with all improvements for 55-65% validation accuracy
"""

import os
import sys
import csv
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Dict, Tuple


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
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
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss


class ResBlock(nn.Module):
    """Residual block with batch norm and dropout"""
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class ImprovedCNN(nn.Module):
    """Improved CNN architecture with residual connections"""
    def __init__(self, num_classes=3, in_channels=8, dropout=0.4):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.block1 = ResBlock(64, dropout=0.2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.block2 = ResBlock(128, dropout=0.3)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.block3 = ResBlock(256, dropout=dropout)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.block1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.block2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.block3(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MultiChannelDataset(Dataset):
    """Dataset with multi-channel support and augmentation"""
    def __init__(self, samples, class_to_idx, img_size=96, data_root=None, 
                 band_names=None, is_training=True):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.img_size = img_size
        self.data_root = data_root
        self.band_names = band_names or []
        self.is_training = is_training
        
        # Augmentation for training
        if is_training:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            ])
        else:
            self.augment = None
        
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, (xmin, ymin, xmax, ymax), label = self.samples[idx]
        
        # Load base image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        
        # Crop with padding
        w, h = img.size
        pad = 5
        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(w, xmax + pad)
        ymax = min(h, ymax + pad)
        
        cropped = img.crop((xmin, ymin, xmax, ymax))
        cropped = self.resize(cropped)
        
        # Apply augmentation if training
        if self.augment:
            cropped = self.augment(cropped)
        
        tensors = []
        
        # Add band channels
        if self.band_names and self.data_root:
            for band in self.band_names:
                band_path = img_path.replace("/DTM/", f"/{band}/")
                try:
                    if os.path.exists(band_path):
                        band_img = Image.open(band_path).convert("L")
                        band_crop = band_img.crop((xmin, ymin, xmax, ymax))
                        band_crop = self.resize(band_crop)
                        band_tensor = self.to_tensor(band_crop)
                        tensors.append(band_tensor)
                    else:
                        tensors.append(torch.zeros(1, self.img_size, self.img_size))
                except:
                    tensors.append(torch.zeros(1, self.img_size, self.img_size))
        
        # Add RGB channels
        rgb_tensor = self.to_tensor(cropped)
        tensors.append(rgb_tensor)
        
        # Concatenate all channels
        img_tensor = torch.cat(tensors, dim=0)
        
        # Normalize
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor, self.class_to_idx[label]


def collect_all_samples(data_root, datasets, csv_name):
    """Collect samples from all datasets"""
    all_samples = []
    for dataset in datasets:
        dataset_dir = os.path.join(data_root, dataset)
        csv_path = os.path.join(dataset_dir, csv_name)
        
        if not os.path.exists(csv_path):
            continue
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    try:
                        img_path = os.path.join(dataset_dir, row[0])
                        bbox = (int(float(row[1])), int(float(row[2])),
                               int(float(row[3])), int(float(row[4])))
                        label = row[5].strip()
                        all_samples.append((img_path, bbox, label))
                    except:
                        continue
    
    return all_samples


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixup augmentation
        if use_mixup and np.random.random() > 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return running_loss / total, correct / total


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("OPTIMIZED CNN TRAINING FOR SATELLITE IMAGERY")
    print("Target: 55-65% validation accuracy (improved from 40%)")
    print("="*70 + "\n")
    
    # Configuration
    data_root = "frontend/training_data"
    datasets = ["DTM", "Hillshade", "Slope", "Sky_View_Factor", "Local_dominance"]
    band_names = ["DTM", "Hillshade", "Slope", "Sky_View_Factor", "Local_dominance"]
    
    batch_size = 64
    epochs = 20
    initial_lr = 0.001
    weight_decay = 1e-4
    img_size = 96  # Smaller for faster training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data from all datasets...")
    train_samples = collect_all_samples(data_root, datasets, "train_annotations.csv")
    val_samples = collect_all_samples(data_root, datasets, "valid_annotations.csv")
    
    # Build class mapping
    all_labels = set(s[2] for s in train_samples + val_samples)
    class_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    num_classes = len(class_to_idx)
    
    print(f"Classes ({num_classes}): {list(class_to_idx.keys())}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Calculate class weights
    train_labels = [class_to_idx[s[2]] for s in train_samples]
    class_weights = compute_class_weight('balanced',
                                        classes=np.unique(train_labels),
                                        y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Create datasets
    train_dataset = MultiChannelDataset(train_samples, class_to_idx, img_size,
                                       data_root, band_names, is_training=True)
    val_dataset = MultiChannelDataset(val_samples, class_to_idx, img_size,
                                     data_root, band_names, is_training=False)
    
    # Weighted sampling for balanced batches
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)
    
    # Determine input channels
    sample_x, _ = train_dataset[0]
    in_channels = sample_x.shape[0]
    print(f"Input channels: {in_channels} (bands + RGB)")
    
    # Create model
    model = ImprovedCNN(num_classes=num_classes, in_channels=in_channels, dropout=0.4)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    print("\n" + "="*70)
    print("IMPROVEMENTS APPLIED:")
    print("✓ Focal Loss - handles class imbalance")
    print("✓ Multi-channel input - using all 5 bands + RGB")
    print("✓ Data augmentation - flip, rotation, color jitter")
    print("✓ Batch normalization - stabilizes training")
    print("✓ Dropout (0.2-0.4) - reduces overfitting")
    print("✓ Residual connections - better gradient flow")
    print("✓ Weight decay - L2 regularization")
    print("✓ Weighted sampling - balanced batches")
    print("✓ Mixup augmentation - better generalization")
    print("✓ Learning rate scheduling - adaptive learning")
    print("✓ Gradient clipping - stable training")
    print("="*70 + "\n")
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting training...\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
            }, 'best_model.pt')
            print(f"✓ Saved best model (val_acc: {val_acc*100:.2f}%)")
        
        print()
        
        # Early stopping if we reach target
        if val_acc >= 0.65:
            print("✅ Reached target accuracy! Stopping early.")
            break
    
    # Final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Final Train Accuracy: {train_acc*100:.2f}%")
    
    # Calculate improvement
    baseline = 0.40  # Original 40% accuracy
    improvement = best_val_acc - baseline
    print(f"\nImprovement: {baseline*100:.0f}% → {best_val_acc*100:.2f}% ")
    print(f"Gain: +{improvement*100:.2f}% absolute improvement")
    
    if best_val_acc >= 0.55:
        print("\n✅ SUCCESS! Achieved target of 55-65% validation accuracy!")
    else:
        print(f"\n📊 Current best: {best_val_acc*100:.2f}%")
    
    # Save final model
    os.makedirs("saved_models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_path = f"saved_models/optimized_model_{timestamp}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'best_val_acc': best_val_acc,
        'history': history,
        'config': {
            'img_size': img_size,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'bands': band_names
        }
    }, final_path)
    
    print(f"\nModel saved to: {final_path}")
    
    # Save training history
    with open('training_history_optimized.json', 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'improvement': improvement,
            'history': history,
            'improvements_applied': [
                'Focal Loss for class imbalance',
                'Multi-channel input (5 bands + RGB)',
                'Data augmentation',
                'Batch normalization',
                'Dropout regularization',
                'Residual connections',
                'Weight decay (L2)',
                'Weighted sampling',
                'Mixup augmentation',
                'Learning rate scheduling',
                'Gradient clipping'
            ]
        }, f, indent=2)
    
    print("Training history saved to: training_history_optimized.json")


if __name__ == "__main__":
    main()