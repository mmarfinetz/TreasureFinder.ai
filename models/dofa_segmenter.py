import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


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
        # Load DOFA backbone from PyTorch Hub
        self.backbone = torch.hub.load(hub_repo, backbone, pretrained=pretrained)
        self.backbone.eval()

        # Build dynamic input adapter
        backbone_in = getattr(self.backbone, "input_channels", 3)
        if hasattr(self.backbone, "build_adapter"):
            self.adapter = self.backbone.build_adapter(in_channels)
        elif hasattr(self.backbone, "adapter"):
            self.adapter = self.backbone.adapter(in_channels)
        elif in_channels != backbone_in:
            self.adapter = nn.Conv2d(in_channels, backbone_in, kernel_size=1)
        else:
            self.adapter = nn.Identity()

        # Determine feature dimensions from backbone
        out_channels = getattr(self.backbone, "out_channels", getattr(self.backbone, "num_features", 256))
        self.seg_head = UNetHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw segmentation logits."""
        original_size = x.shape[-2:]
        x = self.adapter(x)
        features = self.backbone(x)
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
