#!/usr/bin/env python3
"""
Hyperparameter Tuning Script using Optuna
Automatically finds optimal hyperparameters for the CNN model
"""

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
import os
import sys
import json
from typing import Dict, Any
import argparse

# Import training functions from enhanced script
sys.path.append(os.path.dirname(__file__))
from train_enhanced import (
    EnhancedCNN, EnhancedBBoxDataset, FocalLoss,
    train_epoch, validate_epoch, EarlyStopping
)
from train_from_annotations import collect_dataset


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD']),
        'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'exponential']),
        'mixup_alpha': trial.suggest_float('mixup_alpha', 0.0, 0.4),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 4.0) if args.num_classes > 2 else 2.0,
        'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 5.0),
    }
    
    # Architecture parameters
    architecture_params = {
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'num_residual_blocks': trial.suggest_int('num_residual_blocks', 1, 4),
        'initial_channels': trial.suggest_categorical('initial_channels', [32, 64, 128]),
    }
    
    # Data augmentation parameters
    if args.augment:
        aug_params = {
            'rotation_degrees': trial.suggest_int('rotation_degrees', 0, 30),
            'color_jitter_strength': trial.suggest_float('color_jitter_strength', 0.0, 0.3),
            'random_erasing_prob': trial.suggest_float('random_erasing_prob', 0.0, 0.3),
        }
    else:
        aug_params = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    band_names = [b.strip() for b in args.bands.split(",") if b.strip()]
    
    train_samples, val_samples, class_to_idx = collect_dataset(
        args.data_root, datasets, args.train_csv, args.val_csv, validate_paths=True
    )
    
    if len(train_samples) == 0:
        return float('inf')
    
    # Create datasets with suggested parameters
    train_ds = EnhancedBBoxDataset(
        train_samples, class_to_idx, args.img_size, args.data_root,
        band_names, args.include_rgb, augment=args.augment, is_training=True
    )
    
    val_ds = EnhancedBBoxDataset(
        val_samples, class_to_idx, args.img_size, args.data_root,
        band_names, args.include_rgb, augment=False, is_training=False
    ) if val_samples else None
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = None
    if val_ds:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    # Create model with suggested architecture
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    num_classes = len(class_to_idx)
    
    model = EnhancedCNN(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=params['dropout']
    ).to(device)
    
    # Loss function
    if num_classes > 2:
        criterion = FocalLoss(alpha=1.0, gamma=params['focal_gamma'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else:  # SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'], momentum=0.9)
    
    # Learning rate scheduler
    if params['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_trials_epochs)
    elif params['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    else:  # exponential
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Training loop
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=5)  # Reduced patience for trials
    
    for epoch in range(args.n_trials_epochs):
        # Create args object for train_epoch
        trial_args = argparse.Namespace(
            mixup_alpha=params['mixup_alpha'],
            label_smoothing=params['label_smoothing'],
            gradient_clip=params['gradient_clip']
        )
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, trial_args, epoch)
        
        # Validate
        if val_loader:
            val_loss, val_acc, _ = validate_epoch(model, val_loader, criterion, device)
            
            # Update best accuracy
            best_val_acc = max(best_val_acc, val_acc)
            
            # Report intermediate value for pruning
            trial.report(val_acc, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if early_stopping(val_loss):
                break
            
            # Update scheduler
            if params['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        else:
            # If no validation set, use training accuracy
            best_val_acc = train_acc
            trial.report(train_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return best_val_acc


def tune_hyperparameters(args: argparse.Namespace):
    """Main hyperparameter tuning function"""
    
    # Count number of classes for loss function selection
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    band_names = [b.strip() for b in args.bands.split(",") if b.strip()]
    _, _, class_to_idx = collect_dataset(
        args.data_root, datasets, args.train_csv, args.val_csv, validate_paths=True
    )
    args.num_classes = len(class_to_idx)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1  # Use 1 job to avoid GPU conflicts
    )
    
    # Print results
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Validation Accuracy: {best_trial.value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "n_trials": len(study.trials),
        "n_completed_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
    }
    
    output_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Generate training command with best parameters
    print("\n" + "="*60)
    print("RECOMMENDED TRAINING COMMAND")
    print("="*60)
    
    cmd = f"""python train_enhanced.py \\
    --data-root {args.data_root} \\
    --datasets {args.datasets} \\
    --img-size {args.img_size} \\
    --batch-size {best_trial.params['batch_size']} \\
    --epochs {args.final_epochs} \\
    --lr {best_trial.params['lr']:.6f} \\
    --weight-decay {best_trial.params['weight_decay']:.6f} \\
    --mixup-alpha {best_trial.params.get('mixup_alpha', 0.2):.2f} \\
    --label-smoothing {best_trial.params.get('label_smoothing', 0.1):.2f} \\
    --gradient-clip {best_trial.params.get('gradient_clip', 1.0):.1f}"""
    
    if args.augment:
        cmd += " \\\n    --augment"
    if args.bands:
        cmd += f" \\\n    --bands {args.bands}"
    if args.include_rgb:
        cmd += " \\\n    --include-rgb"
    
    print(cmd)
    
    return study


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for satellite CNN")
    
    # Data arguments
    parser.add_argument("--data-root", required=True, help="Root directory for datasets")
    parser.add_argument("--datasets", default="DTM,Open_Positive", help="Dataset folders")
    parser.add_argument("--train-csv", default="train_annotations.csv")
    parser.add_argument("--val-csv", default="valid_annotations.csv")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--bands", default="", help="Band folder names")
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    
    # Tuning arguments
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n-trials-epochs", type=int, default=20, help="Epochs per trial")
    parser.add_argument("--final-epochs", type=int, default=100, help="Epochs for final training")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="hyperparameter_results")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for Optuna installation
    try:
        import optuna
        print(f"Optuna version: {optuna.__version__}")
    except ImportError:
        print("Error: Optuna not installed. Install with: pip install optuna")
        sys.exit(1)
    
    # Run tuning
    tune_hyperparameters(args)