#!/usr/bin/env python3
import argparse
import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    p = argparse.ArgumentParser(description='Convert a DOFA HF state_dict to a serialized torch.nn.Module file for production use.')
    p.add_argument('--state', required=True, help='Path to HF-style checkpoint (.pth or .pt) containing a state_dict')
    p.add_argument('--out', required=True, help='Output path for serialized Module (e.g., /abs/path/weights/dofa.pth)')
    p.add_argument('--backbone', default='base', help='DOFA backbone entrypoint (e.g., vit_base_dofa or dofa_base)')
    p.add_argument('--in-channels', type=int, default=8, help='Input channel count expected by your pipeline (default: 8)')
    p.add_argument('--num-classes', type=int, default=2, help='Segmentation classes (default: 2)')
    p.add_argument('--hub-repo', default='DofA/DOFA', help='Torch hub repo for DOFA backbone (used to construct model)')
    args = p.parse_args()

    # Normalize backbone token
    bb = args.backbone
    # Normalize common shorthands
    key = args.backbone.lower()
    if key in ('tiny',):
        bb = 'dofa_tiny'
    elif key in ('small',):
        bb = 'dofa_small'
    elif key in ('base',):
        bb = 'dofa_base'

    # Lazy import to ensure repo root on path
    from models.dofa_segmenter import DOFASegmenter

    print(f"Loading state from {args.state}")
    obj = torch.load(args.state, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        sd = obj['state_dict']
    elif isinstance(obj, dict):
        sd = obj
    else:
        print('Expected a dict or {"state_dict": dict} in the checkpoint file')
        sys.exit(2)

    print(f"Building DOFA model: backbone={bb}, in_channels={args.in_channels}, num_classes={args.num_classes}")
    try:
        model = DOFASegmenter(
            backbone=bb,
            num_classes=int(args.num_classes),
            in_channels=int(args.in_channels),
            hub_repo=args.hub_repo,
            pretrained=False,
        ).eval()
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
        print(f"Missing dependency: {missing}. Install it in your environment, e.g.:\n  pip install timm torch torchvision\nThen rerun this conversion.")
        raise

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded state_dict with strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
    if any(k.endswith('.weight') for k in sd.keys()):
        first_weight = next((v for k, v in sd.items() if hasattr(v, 'shape')), None)
        if first_weight is not None and hasattr(first_weight, 'shape') and len(first_weight.shape) == 4:
            in_ch = int(first_weight.shape[1])
            if in_ch != args.in_channels:
                print(f"WARNING: checkpoint appears to expect in_channels={in_ch}, but you requested {args.in_channels}")

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model, args.out)
    print(f"Saved serialized torch.nn.Module to {args.out}")

if __name__ == '__main__':
    main()
