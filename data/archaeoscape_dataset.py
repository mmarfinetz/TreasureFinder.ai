"""ArchaeoScape dataset utilities.

This module provides a :class:`torch.utils.data.Dataset` implementation for
handling geospatial archaeological data consisting of RGB orthophotos, LiDAR
normalised Digital Terrain Models (nDTMs) and polygon annotations of sites.

The dataset creates aligned multi-channel patches and corresponding binary
segmentation masks. It also offers helper functions for creating training and
validation splits together with associated :class:`torch.utils.data.DataLoader`
objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import windows
from rasterio.features import rasterize
import torch
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - geopandas is optional in many environments
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry
except Exception:  # pragma: no cover - fall back to fiona/shapely
    gpd = None
    BaseGeometry = object


@dataclass
class PatchIndex:
    """Structure describing a single patch within a tile."""

    tile: str
    window: windows.Window


class ArchaeoScapeDataset(Dataset):
    """PyTorch dataset for the ArchaeoScape multi-modal dataset.

    Parameters
    ----------
    root: str or :class:`pathlib.Path`
        Root directory containing ``orthophotos``, ``ndtm`` and ``annotations``
        sub-directories.
    tiles: sequence of str
        Identifiers of tiles to be used (stem of file names).
    crop_size: int, default ``256``
        Width and height of generated patches in pixels.
    transforms: callable, optional
        Optional transformation applied to ``(image, mask)`` pairs.
    """

    def __init__(
        self,
        root: Path | str,
        tiles: Sequence[str],
        crop_size: int = 256,
        transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.tiles = list(tiles)
        self.crop_size = int(crop_size)
        self.transforms = transforms

        self.ortho_dir = self.root / "orthophotos"
        self.ndtm_dir = self.root / "ndtm"
        self.ann_dir = self.root / "annotations"

        self._polygons: Dict[str, List[Tuple[BaseGeometry, int]]] = {}
        self._index: List[PatchIndex] = []

        self._prepare_index()

    # ------------------------------------------------------------------
    def _prepare_index(self) -> None:
        """Pre-compute crop windows and polygon annotations for each tile."""

        for tile in self.tiles:
            ortho_path = self.ortho_dir / f"{tile}.tif"
            if not ortho_path.exists():
                raise FileNotFoundError(f"Orthophoto for tile '{tile}' not found: {ortho_path}")

            # Load polygons once per tile
            ann_path = self.ann_dir / f"{tile}.geojson"
            if not ann_path.exists():
                ann_path = self.ann_dir / f"{tile}.shp"
            if ann_path.exists() and gpd is not None:
                gdf = gpd.read_file(ann_path)
                self._polygons[tile] = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
            else:
                self._polygons[tile] = []

            with rasterio.open(ortho_path) as src:
                width, height = src.width, src.height

            for top in range(0, height - self.crop_size + 1, self.crop_size):
                for left in range(0, width - self.crop_size + 1, self.crop_size):
                    win = windows.Window(left, top, self.crop_size, self.crop_size)
                    self._index.append(PatchIndex(tile, win))

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 - short description inherited
        return len(self._index)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, mask)`` pair for a given index."""

        patch = self._index[idx]
        tile = patch.tile
        window = patch.window
        ortho_path = self.ortho_dir / f"{tile}.tif"
        ndtm_path = self.ndtm_dir / f"{tile}.tif"

        with rasterio.open(ortho_path) as ortho_src:
            ortho = ortho_src.read([1, 2, 3], window=window).astype(np.float32) / 255.0
            transform = windows.transform(window, ortho_src.transform)

        with rasterio.open(ndtm_path) as ndtm_src:
            ndtm = ndtm_src.read(1, window=window).astype(np.float32)

        image = np.concatenate([ortho, ndtm[None, ...]], axis=0)

        if self._polygons[tile]:
            mask = rasterize(
                self._polygons[tile],
                out_shape=(self.crop_size, self.crop_size),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
        else:
            mask = np.zeros((self.crop_size, self.crop_size), dtype="uint8")

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).long()

        if self.transforms:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Helper functions for creating dataset and dataloaders

def _collect_tiles(root: Path) -> List[str]:
    """Return sorted tile identifiers available in ``root``."""

    ortho_dir = root / "orthophotos"
    return sorted(p.stem for p in ortho_dir.glob("*.tif"))


def split_tiles(tiles: Sequence[str], val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split tile identifiers into training and validation sets."""

    rng = np.random.default_rng(seed)
    tiles = list(tiles)
    rng.shuffle(tiles)
    val_count = max(1, int(len(tiles) * val_fraction)) if tiles else 0
    return tiles[val_count:], tiles[:val_count]


def create_datasets(
    root: Path | str,
    crop_size: int = 256,
    val_fraction: float = 0.2,
    seed: int = 42,
    transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[ArchaeoScapeDataset, ArchaeoScapeDataset]:
    """Create training and validation :class:`ArchaeoScapeDataset` instances."""

    root = Path(root)
    tiles = _collect_tiles(root)
    train_tiles, val_tiles = split_tiles(tiles, val_fraction, seed)
    train_ds = ArchaeoScapeDataset(root, train_tiles, crop_size, transforms)
    val_ds = ArchaeoScapeDataset(root, val_tiles, crop_size, transforms)
    return train_ds, val_ds


def create_dataloaders(
    root: Path | str,
    crop_size: int = 256,
    batch_size: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader objects for training and validation splits."""

    train_ds, val_ds = create_datasets(root, crop_size, val_fraction, seed, transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


__all__ = [
    "ArchaeoScapeDataset",
    "create_datasets",
    "create_dataloaders",
    "split_tiles",
]
