#!/usr/bin/env python3
import argparse
import os
import sys
import csv
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN from CSV annotations in frontend/training_data")
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "frontend", "training_data"),
        help="Root directory containing dataset folders (DTM, Open_Positive, etc.)",
    )
    parser.add_argument(
        "--datasets",
        default="DTM,Open_Positive",
        help="Comma-separated dataset folder names under data-root to include",
    )
    parser.add_argument(
        "--train-csv",
        default="train_annotations.csv",
        help="Training CSV filename inside each dataset folder",
    )
    parser.add_argument(
        "--val-csv",
        default="valid_annotations.csv",
        help="Validation CSV filename inside each dataset folder",
    )
    parser.add_argument("--img-size", type=int, default=64, help="Crop resize size for patches")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended on macOS)")
    parser.add_argument("--model-out", default=None, help="Output path for saved model; default saved_models/<timestamp>.pt")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset and exit")
    parser.add_argument("--skip-validation", action="store_true", help="Skip file existence validation (not recommended)")
    parser.add_argument(
        "--bands",
        default="",
        help=(
            "Comma-separated band folder names (under data-root) to stack as channels. "
            "Example: DTM,Hillshade,Slope,Sky_View_Factor,Local_dominance"
        ),
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Include base RGB from the sample's image as 3 extra channels (after band stack)",
    )
    return parser.parse_args()


def read_annotations(dataset_dir: str, csv_name: str, validate_paths: bool = True) -> List[Tuple[str, Tuple[int, int, int, int], str]]:
    csv_path = os.path.join(dataset_dir, csv_name)
    samples: List[Tuple[str, Tuple[int, int, int, int], str]] = []
    missing_files: List[str] = []
    
    if not os.path.exists(csv_path):
        if validate_paths:
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
        return samples
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row or len(row) < 6:
                # Support rows without header: path,xmin,ymin,xmax,ymax,label
                # Some CSVs appear as 6 columns; if 5 columns, last is label
                pass
            # Expected format: images/<file>.tif,xmin,ymin,xmax,ymax,label
            img_rel = row[0]
            try:
                xmin = int(float(row[1])); ymin = int(float(row[2])); xmax = int(float(row[3])); ymax = int(float(row[4]))
            except Exception:
                # Skip malformed lines
                continue
            label = row[5].strip() if len(row) > 5 else "unknown"
            img_path = os.path.join(dataset_dir, img_rel)
            
            # Validate file existence in production mode
            if validate_paths and not os.path.exists(img_path):
                missing_files.append(f"Row {row_idx+1}: {img_path}")
            
            samples.append((img_path, (xmin, ymin, xmax, ymax), label))
    
    if validate_paths and missing_files:
        error_msg = f"\nMissing {len(missing_files)} files referenced in {csv_path}:\n"
        error_msg += "\n".join(missing_files[:10])  # Show first 10
        if len(missing_files) > 10:
            error_msg += f"\n... and {len(missing_files) - 10} more"
        raise FileNotFoundError(error_msg)
    
    return samples


def collect_dataset(data_root: str, datasets: List[str], train_csv: str, val_csv: str, validate_paths: bool = True):
    train_samples: List[Tuple[str, Tuple[int, int, int, int], str]] = []
    val_samples: List[Tuple[str, Tuple[int, int, int, int], str]] = []
    
    # Enforce production mode validation
    production_mode = os.environ.get('PRODUCTION_MODE', '').lower() == 'true'
    if production_mode:
        validate_paths = True
        print("[PRODUCTION MODE] Enforcing strict path validation")

    for name in datasets:
        ds_dir = os.path.join(data_root, name)
        if not os.path.exists(ds_dir):
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")
        
        train_samples.extend(read_annotations(ds_dir, train_csv, validate_paths))
        val_samples.extend(read_annotations(ds_dir, val_csv, validate_paths))

    # Build label mapping from union of labels present
    labels_sorted = sorted({lbl for _, _, lbl in train_samples + val_samples})
    class_to_idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels_sorted)}
    return train_samples, val_samples, class_to_idx


class BBoxPatchDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, Tuple[int, int, int, int], str]],
        class_to_idx: Dict[str, int],
        img_size: int,
        data_root: str,
        band_names: List[str],
        include_rgb: bool,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.img_size = img_size
        self.data_root = data_root
        self.band_names = band_names
        self.include_rgb = include_rgb
        # Use simple resize + ToTensor per single image; we'll concatenate channels manually
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, (xmin, ymin, xmax, ymax), label = self.samples[idx]
        try:
            base_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}. Ensure --data-root points to correct directory with real .tif files.")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path}: {e}")

        # Crop helper
        def crop_box(img: Image.Image) -> Image.Image:
            w, h = img.size
            xmin_c = max(0, min(int(xmin), w - 1))
            ymin_c = max(0, min(int(ymin), h - 1))
            xmax_c = max(0, min(int(xmax), w))
            ymax_c = max(0, min(int(ymax), h))
            if xmax_c <= xmin_c or ymax_c <= ymin_c:
                return img
            return img.crop((xmin_c, ymin_c, xmax_c, ymax_c))

        # Build channel stack
        tensors: List[torch.Tensor] = []

        # Multi-band grayscale stack from specified band folders
        if self.band_names:
            # Recover relative path under dataset folder: 'images/<file>'
            rel_idx = img_path.rfind(os.sep + "images" + os.sep)
            rel_tail = img_path[rel_idx + 1 :] if rel_idx != -1 else os.path.basename(img_path)
            for band in self.band_names:
                band_img_path = os.path.join(self.data_root, band, rel_tail)
                if os.path.exists(band_img_path):
                    try:
                        band_img = Image.open(band_img_path).convert("L")
                    except Exception:
                        # Fallback to base image grayscale if band unreadable
                        band_img = base_img.convert("L")
                else:
                    # Missing band image: fill with zeros using base image size
                    band_img = Image.new("L", base_img.size, color=0)
                band_patch = crop_box(band_img)
                band_patch = self.resize(band_patch)
                band_tensor = self.to_tensor(band_patch)  # (1,H,W)
                tensors.append(band_tensor)

        # Optional include base RGB as channels
        if self.include_rgb or not self.band_names:
            rgb_patch = crop_box(base_img)
            rgb_patch = self.resize(rgb_patch)
            rgb_tensor = self.to_tensor(rgb_patch)  # (3,H,W)
            tensors.append(rgb_tensor)

        # Concatenate along channel dim
        patch = torch.cat(tensors, dim=0)
        target = self.class_to_idx[label]
        return patch, target


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    band_names = [b.strip() for b in args.bands.split(",") if b.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Data root: {args.data_root}")
    print(f"Datasets: {datasets}")
    if band_names:
        print(f"Bands: {band_names} | include_rgb={args.include_rgb}")

    # Validate data root exists
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root directory not found: {args.data_root}")
    
    validate_paths = not args.skip_validation
    train_samples, val_samples, class_to_idx = collect_dataset(
        args.data_root, datasets, args.train_csv, args.val_csv, validate_paths
    )

    if len(train_samples) == 0:
        print("No training samples found. Check paths and CSVs.")
        sys.exit(1)

    print(f"Classes ({len(class_to_idx)}): {sorted(class_to_idx.keys())}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    if args.dry_run:
        # Comprehensive validation in dry-run mode
        import random
        print("\n[DRY RUN] Validating data integrity...")
        
        # Check all training samples
        missing_train = []
        for i, (p, *_rest) in enumerate(train_samples):
            if not os.path.exists(p):
                missing_train.append(p)
        
        # Check all validation samples
        missing_val = []
        for i, (p, *_rest) in enumerate(val_samples):
            if not os.path.exists(p):
                missing_val.append(p)
        
        # Random sample validation (N=50)
        sample_size = min(50, len(train_samples))
        random_samples = random.sample(train_samples, sample_size) if train_samples else []
        valid_samples = sum(1 for (p, *_) in random_samples if os.path.exists(p))
        
        print(f"\nValidation Results:")
        print(f"  Total train samples: {len(train_samples)}")
        print(f"  Missing train files: {len(missing_train)}")
        print(f"  Total val samples: {len(val_samples)}")
        print(f"  Missing val files: {len(missing_val)}")
        print(f"\nRandom sample check ({sample_size} samples):")
        print(f"  Valid: {valid_samples}/{sample_size} ({100*valid_samples/sample_size:.1f}%)")
        
        if missing_train or missing_val:
            print("\n[ERROR] Missing files detected!")
            if missing_train:
                print(f"  First 5 missing train files:")
                for f in missing_train[:5]:
                    print(f"    - {f}")
            if missing_val:
                print(f"  First 5 missing val files:")
                for f in missing_val[:5]:
                    print(f"    - {f}")
            
            # In production mode, fail hard
            if os.environ.get('PRODUCTION_MODE', '').lower() == 'true':
                sys.exit(1)
        else:
            print("\n[SUCCESS] All referenced files exist!")
        
        return

    num_classes = len(class_to_idx)
    train_ds = BBoxPatchDataset(train_samples, class_to_idx, args.img_size, args.data_root, band_names, args.include_rgb)
    val_ds = BBoxPatchDataset(val_samples, class_to_idx, args.img_size, args.data_root, band_names, args.include_rgb) if val_samples else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    # Determine input channels from a sample
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]
    model = SimpleCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        if val_loader is not None:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    v_correct += (preds == targets).sum().item()
                    v_total += targets.size(0)
            val_acc = v_correct / max(v_total, 1)
        else:
            val_acc = 0.0

        print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} acc={train_acc:.3f} val_acc={val_acc:.3f}")
        best_val_acc = max(best_val_acc, val_acc)

    # Save model
    out_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.model_out or os.path.join(out_dir, f"bbox_cnn_{ts}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "img_size": args.img_size,
            "num_classes": num_classes,
            "best_val_acc": best_val_acc,
        },
        out_path,
    )
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()


