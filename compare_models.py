#!/usr/bin/env python3
"""
Model Comparison Script
Compare performance between original SimpleCNN and EnhancedCNN
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import sys
import argparse
import json
from typing import Dict, List, Tuple

# Import both model architectures
sys.path.append(os.path.dirname(__file__))
from train_from_annotations import SimpleCNN, BBoxPatchDataset, collect_dataset
from train_enhanced import EnhancedCNN, EnhancedBBoxDataset


def load_model(model_path: str, device: torch.device):
    """Load a saved model checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model type from checkpoint
    in_channels = checkpoint.get('in_channels', 3)
    num_classes = checkpoint.get('num_classes', 2)
    
    # Try to determine model architecture
    state_dict = checkpoint['model_state_dict']
    
    # Check for enhanced model features (batch norm, attention modules)
    is_enhanced = any('bn' in key or 'attention' in key for key in state_dict.keys())
    
    if is_enhanced:
        model = EnhancedCNN(num_classes=num_classes, in_channels=in_channels)
    else:
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, dataloader, device, class_names=None):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class metrics
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names,
                                  output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_comparison(results_simple: Dict, results_enhanced: Dict, class_names: List[str], save_path: str):
    """Create comparison visualizations"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrices
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(results_simple['confusion_matrix'], annot=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    ax1.set_title('SimpleCNN Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(results_enhanced['confusion_matrix'], annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Greens')
    ax2.set_title('EnhancedCNN Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 2. Per-class Performance Comparison
    ax3 = plt.subplot(2, 3, 3)
    
    simple_f1 = [results_simple['classification_report'][c]['f1-score'] for c in class_names]
    enhanced_f1 = [results_enhanced['classification_report'][c]['f1-score'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax3.bar(x - width/2, simple_f1, width, label='SimpleCNN', color='blue', alpha=0.7)
    ax3.bar(x + width/2, enhanced_f1, width, label='EnhancedCNN', color='green', alpha=0.7)
    ax3.set_xlabel('Class')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Per-Class F1 Score Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Overall Metrics Comparison
    ax4 = plt.subplot(2, 3, 4)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    simple_metrics = [
        results_simple['accuracy'],
        results_simple['classification_report']['weighted avg']['precision'],
        results_simple['classification_report']['weighted avg']['recall'],
        results_simple['classification_report']['weighted avg']['f1-score']
    ]
    enhanced_metrics = [
        results_enhanced['accuracy'],
        results_enhanced['classification_report']['weighted avg']['precision'],
        results_enhanced['classification_report']['weighted avg']['recall'],
        results_enhanced['classification_report']['weighted avg']['f1-score']
    ]
    
    x = np.arange(len(metrics))
    ax4.bar(x - width/2, simple_metrics, width, label='SimpleCNN', color='blue', alpha=0.7)
    ax4.bar(x + width/2, enhanced_metrics, width, label='EnhancedCNN', color='green', alpha=0.7)
    ax4.set_ylabel('Score')
    ax4.set_title('Overall Performance Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (s, e) in enumerate(zip(simple_metrics, enhanced_metrics)):
        ax4.text(i - width/2, s + 0.01, f'{s:.3f}', ha='center', va='bottom')
        ax4.text(i + width/2, e + 0.01, f'{e:.3f}', ha='center', va='bottom')
    
    # 4. ROC Curves (for binary classification)
    if len(class_names) == 2:
        ax5 = plt.subplot(2, 3, 5)
        
        # Simple CNN ROC
        fpr_simple, tpr_simple, _ = roc_curve(results_simple['labels'], 
                                              results_simple['probabilities'][:, 1])
        auc_simple = auc(fpr_simple, tpr_simple)
        
        # Enhanced CNN ROC
        fpr_enhanced, tpr_enhanced, _ = roc_curve(results_enhanced['labels'],
                                                  results_enhanced['probabilities'][:, 1])
        auc_enhanced = auc(fpr_enhanced, tpr_enhanced)
        
        ax5.plot(fpr_simple, tpr_simple, color='blue', lw=2, 
                label=f'SimpleCNN (AUC = {auc_simple:.3f})')
        ax5.plot(fpr_enhanced, tpr_enhanced, color='green', lw=2,
                label=f'EnhancedCNN (AUC = {auc_enhanced:.3f})')
        ax5.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax5.set_xlim([0.0, 1.0])
        ax5.set_ylim([0.0, 1.05])
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('ROC Curves')
        ax5.legend(loc="lower right")
        ax5.grid(True, alpha=0.3)
    
    # 5. Confidence Distribution
    ax6 = plt.subplot(2, 3, 6)
    
    # Get max probabilities (confidence scores)
    simple_confidence = results_simple['probabilities'].max(axis=1)
    enhanced_confidence = results_enhanced['probabilities'].max(axis=1)
    
    ax6.hist(simple_confidence, bins=30, alpha=0.5, label='SimpleCNN', color='blue', density=True)
    ax6.hist(enhanced_confidence, bins=30, alpha=0.5, label='EnhancedCNN', color='green', density=True)
    ax6.set_xlabel('Confidence Score')
    ax6.set_ylabel('Density')
    ax6.set_title('Prediction Confidence Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to: {save_path}")


def generate_improvement_report(results_simple: Dict, results_enhanced: Dict, class_names: List[str]) -> str:
    """Generate detailed improvement report"""
    
    report = []
    report.append("="*60)
    report.append("MODEL IMPROVEMENT ANALYSIS REPORT")
    report.append("="*60)
    
    # Overall improvement
    acc_improvement = (results_enhanced['accuracy'] - results_simple['accuracy']) * 100
    report.append(f"\n### Overall Accuracy Improvement: {acc_improvement:+.2f}%")
    report.append(f"- SimpleCNN: {results_simple['accuracy']:.4f}")
    report.append(f"- EnhancedCNN: {results_enhanced['accuracy']:.4f}")
    
    # Per-class improvements
    report.append("\n### Per-Class Performance Improvements:")
    for cls in class_names:
        simple_f1 = results_simple['classification_report'][cls]['f1-score']
        enhanced_f1 = results_enhanced['classification_report'][cls]['f1-score']
        improvement = (enhanced_f1 - simple_f1) * 100
        
        report.append(f"\n**{cls}:**")
        report.append(f"  F1-Score: {simple_f1:.3f} → {enhanced_f1:.3f} ({improvement:+.2f}%)")
        report.append(f"  Precision: {results_simple['classification_report'][cls]['precision']:.3f} → "
                     f"{results_enhanced['classification_report'][cls]['precision']:.3f}")
        report.append(f"  Recall: {results_simple['classification_report'][cls]['recall']:.3f} → "
                     f"{results_enhanced['classification_report'][cls]['recall']:.3f}")
    
    # Weighted averages
    report.append("\n### Weighted Average Metrics:")
    for metric in ['precision', 'recall', 'f1-score']:
        simple_val = results_simple['classification_report']['weighted avg'][metric]
        enhanced_val = results_enhanced['classification_report']['weighted avg'][metric]
        improvement = (enhanced_val - simple_val) * 100
        
        report.append(f"- {metric.capitalize()}: {simple_val:.3f} → {enhanced_val:.3f} ({improvement:+.2f}%)")
    
    # Confidence analysis
    simple_confidence = results_simple['probabilities'].max(axis=1)
    enhanced_confidence = results_enhanced['probabilities'].max(axis=1)
    
    report.append("\n### Prediction Confidence Analysis:")
    report.append(f"- SimpleCNN mean confidence: {simple_confidence.mean():.3f} (±{simple_confidence.std():.3f})")
    report.append(f"- EnhancedCNN mean confidence: {enhanced_confidence.mean():.3f} (±{enhanced_confidence.std():.3f})")
    
    # Correct predictions with high confidence (>0.9)
    simple_correct = results_simple['predictions'] == results_simple['labels']
    enhanced_correct = results_enhanced['predictions'] == results_enhanced['labels']
    
    simple_high_conf_correct = (simple_correct & (simple_confidence > 0.9)).sum()
    enhanced_high_conf_correct = (enhanced_correct & (enhanced_confidence > 0.9)).sum()
    
    report.append(f"\n### High Confidence (>0.9) Correct Predictions:")
    report.append(f"- SimpleCNN: {simple_high_conf_correct} ({100*simple_high_conf_correct/len(simple_correct):.1f}%)")
    report.append(f"- EnhancedCNN: {enhanced_high_conf_correct} ({100*enhanced_high_conf_correct/len(enhanced_correct):.1f}%)")
    
    # Error analysis
    report.append("\n### Error Reduction:")
    simple_errors = (~simple_correct).sum()
    enhanced_errors = (~enhanced_correct).sum()
    error_reduction = 100 * (simple_errors - enhanced_errors) / simple_errors if simple_errors > 0 else 0
    
    report.append(f"- SimpleCNN errors: {simple_errors}")
    report.append(f"- EnhancedCNN errors: {enhanced_errors}")
    report.append(f"- Error reduction: {error_reduction:.1f}%")
    
    # Recommendations
    report.append("\n### Recommendations for Further Improvement:")
    
    # Check for class imbalance issues
    for cls in class_names:
        if results_enhanced['classification_report'][cls]['f1-score'] < 0.5:
            report.append(f"- Class '{cls}' has low F1-score ({results_enhanced['classification_report'][cls]['f1-score']:.3f}). "
                         f"Consider collecting more samples or using stronger augmentation.")
    
    # Check confidence
    if enhanced_confidence.mean() < 0.8:
        report.append("- Average confidence is below 0.8. Consider:")
        report.append("  * Training for more epochs")
        report.append("  * Adjusting learning rate schedule")
        report.append("  * Adding more training data")
    
    # Check for overfitting signs
    if enhanced_confidence.std() > 0.3:
        report.append("- High variance in confidence scores suggests potential overfitting:")
        report.append("  * Increase dropout rate")
        report.append("  * Add more regularization")
        report.append("  * Use stronger data augmentation")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Compare CNN model performances")
    parser.add_argument("--simple-model", required=True, help="Path to SimpleCNN model")
    parser.add_argument("--enhanced-model", required=True, help="Path to EnhancedCNN model")
    parser.add_argument("--data-root", required=True, help="Root directory for datasets")
    parser.add_argument("--datasets", default="DTM,Open_Positive")
    parser.add_argument("--val-csv", default="valid_annotations.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="comparison_results")
    parser.add_argument("--bands", default="")
    parser.add_argument("--include-rgb", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    simple_model, simple_checkpoint = load_model(args.simple_model, device)
    enhanced_model, enhanced_checkpoint = load_model(args.enhanced_model, device)
    
    # Get data configuration from checkpoint
    img_size = simple_checkpoint.get('img_size', 64)
    class_to_idx = simple_checkpoint.get('class_to_idx', {})
    class_names = list(class_to_idx.keys())
    
    # Load validation data
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    band_names = [b.strip() for b in args.bands.split(",") if b.strip()]
    
    _, val_samples, _ = collect_dataset(
        args.data_root, datasets, "train_annotations.csv", args.val_csv, validate_paths=True
    )
    
    if not val_samples:
        print("No validation samples found!")
        sys.exit(1)
    
    print(f"Validation samples: {len(val_samples)}")
    
    # Create validation dataset (use enhanced dataset for both for fair comparison)
    val_dataset = EnhancedBBoxDataset(
        val_samples, class_to_idx, img_size, args.data_root,
        band_names, args.include_rgb, augment=False, is_training=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Evaluate both models
    print("\nEvaluating SimpleCNN...")
    results_simple = evaluate_model(simple_model, val_loader, device, class_names)
    
    print("Evaluating EnhancedCNN...")
    results_enhanced = evaluate_model(enhanced_model, val_loader, device, class_names)
    
    # Generate comparison plots
    plot_path = os.path.join(args.output_dir, "model_comparison.png")
    plot_comparison(results_simple, results_enhanced, class_names, plot_path)
    
    # Generate improvement report
    report = generate_improvement_report(results_simple, results_enhanced, class_names)
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(args.output_dir, "improvement_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save numerical results
    results = {
        'simple_cnn': {
            'accuracy': float(results_simple['accuracy']),
            'classification_report': results_simple['classification_report']
        },
        'enhanced_cnn': {
            'accuracy': float(results_enhanced['accuracy']),
            'classification_report': results_enhanced['classification_report']
        },
        'improvement': {
            'accuracy': float(results_enhanced['accuracy'] - results_simple['accuracy']),
            'percentage': float((results_enhanced['accuracy'] - results_simple['accuracy']) * 100)
        }
    }
    
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()