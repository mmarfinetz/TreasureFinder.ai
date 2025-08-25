#!/usr/bin/env python3
"""
Quick demonstration training script with enhanced features for improved accuracy
Targets 55-65% validation accuracy instead of 40%
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ImprovedCNN(nn.Module):
    """Improved CNN with BatchNorm, Dropout, and better architecture"""
    def __init__(self, num_classes: int, in_channels: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        
        # Convolutional blocks with BatchNorm and increased channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout2d(0.4)
        )
        
        # Fully connected layers with dropout
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AugmentedDataset(Dataset):
    """Dataset with strong augmentation for better generalization"""
    def __init__(self, samples, class_to_idx, img_size=128, is_training=True, 
                 band_names=None, data_root=None, include_rgb=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.img_size = img_size
        self.is_training = is_training
        self.band_names = band_names or []
        self.data_root = data_root
        self.include_rgb = include_rgb
        
        # Define augmentations
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, (xmin, ymin, xmax, ymax), label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        
        # Crop to bounding box with some padding
        w, h = img.size
        pad = 10 if self.is_training else 0
        xmin_c = max(0, xmin - pad)
        ymin_c = max(0, ymin - pad)
        xmax_c = min(w, xmax + pad)
        ymax_c = min(h, ymax + pad)
        
        cropped = img.crop((xmin_c, ymin_c, xmax_c, ymax_c))
        
        # Stack multiple bands if specified
        tensors = []
        
        if self.band_names and self.data_root:
            # Process each band
            for band in self.band_names:
                band_path = img_path.replace("/DTM/", f"/{band}/")
                try:
                    if os.path.exists(band_path):
                        band_img = Image.open(band_path).convert("L")
                        band_crop = band_img.crop((xmin_c, ymin_c, xmax_c, ymax_c))
                        band_crop = band_crop.resize((self.img_size, self.img_size))
                        band_tensor = transforms.ToTensor()(band_crop)
                        tensors.append(band_tensor)
                except:
                    # Add zeros if band loading fails
                    tensors.append(torch.zeros(1, self.img_size, self.img_size))
        
        # Add RGB channels if needed
        if self.include_rgb or not self.band_names:
            cropped_tensor = self.transform(cropped)
            if tensors:
                # Concatenate band tensors with RGB
                final_tensor = torch.cat(tensors + [cropped_tensor], dim=0)
            else:
                final_tensor = cropped_tensor
        else:
            # Only use band tensors
            final_tensor = torch.cat(tensors, dim=0) if tensors else torch.zeros(1, self.img_size, self.img_size)
        
        target = self.class_to_idx[label]
        return final_tensor, target


def collect_samples(data_root, datasets, csv_name):
    """Collect all samples from multiple datasets"""
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
                    img_path = os.path.join(dataset_dir, row[0])
                    try:
                        bbox = (int(float(row[1])), int(float(row[2])), 
                               int(float(row[3])), int(float(row[4])))
                        label = row[5].strip()
                        all_samples.append((img_path, bbox, label))
                    except:
                        continue
    
    return all_samples


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_model():
    """Main training function with all improvements"""
    print("=" * 70)
    print("ENHANCED SATELLITE IMAGERY CNN TRAINING")
    print("Target: Improve validation accuracy from 40% to 55-65%")
    print("=" * 70)
    
    # Configuration
    data_root = "frontend/training_data"
    datasets = ["DTM", "Hillshade", "Slope", "Sky_View_Factor", "Local_dominance"]
    band_names = ["DTM", "Hillshade", "Slope", "Sky_View_Factor", "Local_dominance"]
    batch_size = 32
    epochs = 15  # Quick demo with fewer epochs
    initial_lr = 0.001
    weight_decay = 1e-4
    img_size = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_samples = collect_samples(data_root, datasets, "train_annotations.csv")
    val_samples = collect_samples(data_root, datasets, "valid_annotations.csv")
    
    # Build class mapping
    all_labels = set([label for _, _, label in train_samples + val_samples])
    class_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    num_classes = len(class_to_idx)
    
    print(f"Classes ({num_classes}): {list(class_to_idx.keys())}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Calculate class weights for imbalanced data
    train_labels = [class_to_idx[label] for _, _, label in train_samples]
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(train_labels), 
                                        y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Create datasets with augmentation
    train_dataset = AugmentedDataset(train_samples, class_to_idx, img_size, 
                                    is_training=True, band_names=band_names,
                                    data_root=data_root, include_rgb=True)
    val_dataset = AugmentedDataset(val_samples, class_to_idx, img_size, 
                                  is_training=False, band_names=band_names,
                                  data_root=data_root, include_rgb=True)
    
    # Create weighted sampler for balanced training
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    
    # Determine input channels
    sample_x, _ = train_dataset[0]
    in_channels = sample_x.shape[0]
    print(f"Input channels: {in_channels}")
    
    # Create model
    model = ImprovedCNN(num_classes=num_classes, in_channels=in_channels, dropout_rate=0.4)
    model = model.to(device)
    
    # Loss function - Focal Loss for class imbalance
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING WITH ENHANCEMENTS:")
    print("- Focal Loss for class imbalance")
    print("- Data augmentation (rotation, flip, color jitter)")
    print("- Batch normalization and dropout (0.4)")
    print("- Weighted sampling for balanced batches")
    print("- Learning rate scheduling with warm restarts")
    print("- L2 regularization (weight decay)")
    print("- Mixup augmentation")
    print("=" * 70 + "\n")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup augmentation 50% of the time
            if np.random.random() > 0.5 and epoch > 2:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Add label smoothing
            if epoch > 3:
                smooth_targets = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
                smooth_targets = smooth_targets * 0.9 + 0.1 / num_classes
                loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * smooth_targets, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.*train_correct/train_total:.2f}%")
        
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
                
                # Per-class accuracy
                for t, p in zip(targets, predicted):
                    class_total[t.item()] += 1
                    if t == p:
                        class_correct[t.item()] += 1
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate per-class accuracy
        class_names = {v: k for k, v in class_to_idx.items()}
        per_class_acc = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                per_class_acc[class_names[i]] = acc
        
        # Print results
        print(f"\n  EPOCH {epoch+1} RESULTS:")
        print(f"  Training   - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
        print(f"  Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
        print(f"  Per-class accuracy: {per_class_acc}")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
            }, 'best_model_checkpoint.pt')
            print(f"  ✓ Saved best model (val_acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 7:
                print(f"\n  Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    # Final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final training accuracy: {train_acc*100:.2f}%")
    print(f"Final validation accuracy: {val_acc*100:.2f}%")
    
    if best_val_acc >= 0.55:
        print("\n✅ SUCCESS! Achieved target of 55-65% validation accuracy!")
    else:
        print(f"\n⚠️  Current accuracy {best_val_acc*100:.2f}% - below target of 55%")
    
    # Save final model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_path = f"saved_models/enhanced_model_{timestamp}.pt"
    os.makedirs("saved_models", exist_ok=True)
    
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
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("Training history saved to: training_history.json")
    
    return best_val_acc, history


if __name__ == "__main__":
    train_model()