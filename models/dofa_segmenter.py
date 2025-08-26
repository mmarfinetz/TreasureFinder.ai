import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import math


class UNetHead(nn.Module):
    """A lightweight U-Net style decoder used as the segmentation head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, original_size: Union[tuple, list]) -> torch.Tensor:
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(self.up1(x)))
        x = F.relu(self.conv3(self.up2(x)))
        x = self.classifier(x)
        x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)
        return x


class DOFASegmenter(nn.Module):
    """Segmentation model built on a DOFA backbone with a U-Net head.

    The model is loaded via PyTorch Hub. A dynamic adapter projects the input
    tensor to the expected number of channels for the backbone, allowing the
    network to accept arbitrary channel counts (e.g., RGB, LiDAR, spectral).
    """

    def __init__(
        self,
        backbone: str = "dofa_tiny",
        num_classes: int = 2,
        in_channels: int = 3,
        hub_repo: str = "DofA/DOFA",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        # Load DOFA backbone from PyTorch Hub
        # Trust the repo to avoid GitHub API fork validation issues
        try:
            self.backbone = torch.hub.load(hub_repo, backbone, pretrained=pretrained, trust_repo=True)
        except TypeError:
            # Older torch versions without trust_repo support
            self.backbone = torch.hub.load(hub_repo, backbone, pretrained=pretrained)
        self.backbone.eval()

        # DOFA OFA backbones accept arbitrary channel counts via dynamic patch embed.
        # Keep channels as-is to match the provided wavelength list.
        self.adapter = nn.Identity()

        # Default wavelength list (micrometers) for common 8-channel Sentinel-2 ordering:
        # [B4 (Red), B3 (Green), B2 (Blue), B8 (NIR), B5 (RE1), B6 (RE2), B11 (SWIR1), B12 (SWIR2)]
        self.default_wave_list = self._build_default_wave_list(self.in_channels)

        # Infer target input size expected by backbone positional embeddings
        try:
            pos = getattr(self.backbone, 'pos_embed', None)
            num_patches = int(pos.shape[1] - 1) if (pos is not None and hasattr(pos, 'shape')) else 196
            grid = int(math.sqrt(max(1, num_patches)))
            patch_size = int(getattr(getattr(self.backbone, 'patch_embed', None), 'kernel_size', 16))
            self.target_size = int(grid * patch_size)
        except Exception:
            self.target_size = 224

        # Determine feature dimensions from backbone. Prefer embedding dim from pos_embed.
        out_channels = 768
        try:
            pe = getattr(self.backbone, 'pos_embed', None)
            if pe is not None and hasattr(pe, 'shape'):
                out_channels = int(pe.shape[-1])
        except Exception:
            out_channels = getattr(self.backbone, "out_channels", getattr(self.backbone, "num_features", 768))
        self.seg_head = UNetHead(out_channels, num_classes)

    @staticmethod
    def _build_default_wave_list(in_channels: int):
        # Values in micrometers (um). Scaled inside DOFA by * 1000 to ~nm.
        if in_channels == 8:
            return [0.665, 0.560, 0.490, 0.842, 0.705, 0.740, 1.610, 2.190]
        # Generic fallback: evenly spaced spectral positions in [0.45, 2.20] µm
        lo, hi = 0.45, 2.20
        if in_channels <= 1:
            return [0.55]
        step = (hi - lo) / max(1, in_channels - 1)
        return [lo + i * step for i in range(in_channels)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw segmentation logits."""
        original_size = x.shape[-2:]
        x = self.adapter(x)
        # Resize input to match backbone's positional embedding grid if needed
        try:
            ts = int(getattr(self, 'target_size', 224))
            if int(original_size[0]) != ts or int(original_size[1]) != ts:
                x = F.interpolate(x, size=(ts, ts), mode="bilinear", align_corners=False)
        except Exception:
            pass
        # Compute spatial features by running the DOFA backbone blocks and reshaping tokens
        wave_list = getattr(self, 'default_wave_list', None)
        if wave_list is None or not isinstance(wave_list, (list, tuple)):
            in_ch = getattr(self, 'in_channels', None)
            if in_ch is None:
                in_ch = int(x.shape[1]) if hasattr(x, 'shape') and len(x.shape) >= 2 else 3
            wave_list = self._build_default_wave_list(int(in_ch))
        features = None
        try:
            # 1) Patch embed with wavelength list
            waves = torch.tensor(wave_list, device=x.device).float()
            patch_embed = getattr(self.backbone, 'patch_embed', None)
            pos_embed = getattr(self.backbone, 'pos_embed', None)
            blocks = getattr(self.backbone, 'blocks', None)
            cls_token = getattr(self.backbone, 'cls_token', None)
            if patch_embed is None or pos_embed is None or blocks is None or cls_token is None:
                raise AttributeError('Backbone missing required attributes for spatial feature extraction')

            tokens, _ = patch_embed(x, waves)
            # Add positional embeddings to patch tokens (skip cls position)
            tokens = tokens + pos_embed[:, 1:, :]
            # Prepend CLS token (as in backbone)
            cls_tok = cls_token + pos_embed[:, :1, :]
            x_tokens = torch.cat((cls_tok.expand(tokens.shape[0], -1, -1), tokens), dim=1)
            # Run transformer blocks
            for blk in blocks:
                x_tokens = blk(x_tokens)
            # Remove CLS and reshape to spatial grid
            patch_tokens = x_tokens[:, 1:, :]
            # Infer grid size from num patches
            num_patches = patch_tokens.shape[1]
            grid = int(math.sqrt(int(num_patches)))
            features = patch_tokens.transpose(1, 2).contiguous().view(x.shape[0], -1, grid, grid)
        except Exception:
            # Fallback: try standard forward; may not produce spatial features
            try:
                features = self.backbone(x, wave_list=wave_list)
            except TypeError:
                features = self.backbone(x)
            if isinstance(features, dict):
                features = features.get("out") or features.get("features") or list(features.values())[0]
            if isinstance(features, (list, tuple)):
                features = features[0]
            # If still not spatial, expand to 1x1 map
            if isinstance(features, torch.Tensor) and features.ndim == 2:
                features = features.unsqueeze(-1).unsqueeze(-1)
        if isinstance(features, dict):  # handle torchvision-style outputs
            features = features.get("out") or features.get("features") or list(features.values())[0]
        if isinstance(features, (list, tuple)):
            features = features[0]
        logits = self.seg_head(features, original_size)
        return logits

    def segment_anomalies(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on ``image_tensor`` and return per-pixel class masks."""
        self.eval()
        with torch.no_grad():
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            logits = self.forward(image_tensor)
            masks = logits.argmax(dim=1)
        if masks.shape[0] == 1:
            return masks[0]
        return masks
