"""Tiling utilities for running DOFA segmenter over large scenes.

This module provides helpers to run a segmentation model on large images by
splitting them into overlapping tiles, merging the predictions, and
vectorising the results to geospatial formats.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

try:  # Optional dependency used for file outputs
    import geopandas as gpd  # type: ignore
    _GEOPANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - geopandas is optional
    gpd = None  # type: ignore
    _GEOPANDAS_AVAILABLE = False


Tile = Tuple[np.ndarray, Tuple[int, int]]


def generate_tiles(image: np.ndarray, tile_size: int, overlap: float) -> Iterator[Tile]:
    """Yield overlapping tiles from a larger image.

    Args:
        image: Array of shape ``(C, H, W)`` or ``(H, W)``.
        tile_size: Size of the square tiles.
        overlap: Fractional overlap ``[0, 1)`` between tiles.

    Yields:
        Tuples of ``(tile, (x, y))`` where ``(x, y)`` is the origin of the tile
        in the original image.
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    _, height, width = image.shape
    stride = max(1, int(tile_size * (1 - overlap)))

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            tile = image[:, y : y + tile_size, x : x + tile_size]
            pad_h = tile_size - tile.shape[1]
            pad_w = tile_size - tile.shape[2]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            yield tile, (x, y)


class SegmenterProtocol:
    """Lightweight protocol for the DOFA segmenter.

    Any object implementing ``predict`` that accepts a tile array and returns a
    probability map ``(num_classes, H, W)`` can be used with the tiling utils.
    """

    num_classes: int

    def predict(self, tile: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol stub
        raise NotImplementedError


def run_tiled_inference(
    segmenter: SegmenterProtocol,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run tiled inference with a segmentation model.

    Args:
        segmenter: Model implementing :class:`SegmenterProtocol`.
        image: Input array ``(C, H, W)`` or ``(H, W)``.
        tile_size: Tile size for inference.
        overlap: Fractional overlap between tiles ``[0,1)``.

    Returns:
        ``(labels, confidence, probabilities)`` where ``labels`` is the
        per-pixel class ID, ``confidence`` the corresponding probability and
        ``probabilities`` the full per-class probability map.
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    channels, height, width = image.shape

    num_classes = getattr(segmenter, "num_classes", None)
    if num_classes is None:
        raise AttributeError("segmenter must define 'num_classes'")

    prob_accum = np.zeros((num_classes, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for tile, (x, y) in generate_tiles(image, tile_size, overlap):
        pred = segmenter.predict(tile)
        h = min(tile_size, height - y)
        w = min(tile_size, width - x)
        prob_accum[:, y : y + h, x : x + w] += pred[:, :h, :w]
        weight[y : y + h, x : x + w] += 1

    weight = np.clip(weight, 1, None)
    probabilities = prob_accum / weight
    labels = probabilities.argmax(axis=0).astype(np.int32)
    confidence = probabilities.max(axis=0)
    return labels, confidence, probabilities


def _connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Find 4‑connected components in a boolean mask."""
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                comp: List[Tuple[int, int]] = []
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append(comp)
    return components


def mask_to_polygons(
    label_mask: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    pixel_size: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    min_area: float = 0.0,
) -> List[Dict[str, Any]]:
    """Convert a label mask into vector polygons.

    Args:
        label_mask: Array of integer class IDs ``(H, W)``.
        confidence: Optional array of confidence scores ``(H, W)``.
        class_names: Optional list mapping class IDs to names.
        pixel_size: Size of a pixel in output units.
        origin: Coordinates of the ``(0,0)`` pixel (``x, y``).
        min_area: Minimum polygon area to keep.

    Returns:
        List of feature dictionaries with geometry and attributes.
    """
    h, w = label_mask.shape
    features: List[Dict[str, Any]] = []

    for class_id in np.unique(label_mask):
        if class_id == 0:
            continue  # treat 0 as background
        mask = label_mask == class_id
        for component in _connected_components(mask):
            boxes = [
                box(
                    origin[0] + c * pixel_size,
                    origin[1] + r * pixel_size,
                    origin[0] + (c + 1) * pixel_size,
                    origin[1] + (r + 1) * pixel_size,
                )
                for r, c in component
            ]
            geom = unary_union(boxes)
            if geom.is_empty or geom.area < min_area:
                continue

            if confidence is not None:
                rr, cc = zip(*component)
                conf = float(confidence[rr, cc].mean())
            else:
                conf = None

            feature: Dict[str, Any] = {
                "geometry": geom,
                "class_id": int(class_id),
                "confidence": conf,
            }
            if class_names and class_id < len(class_names):
                feature["class_name"] = class_names[class_id]
            features.append(feature)
    return features


def export_polygons(
    polygons: Sequence[Dict[str, Any]],
    geojson_path: Optional[str] = None,
    shapefile_path: Optional[str] = None,
    crs: str = "EPSG:4326",
) -> None:
    """Export polygon features to GeoJSON and/or Shapefile.

    Args:
        polygons: Iterable of feature dictionaries returned by
            :func:`mask_to_polygons`.
        geojson_path: Output path for GeoJSON (optional).
        shapefile_path: Output path for ESRI Shapefile (optional, requires
            GeoPandas).
        crs: Coordinate reference system for outputs.
    """
    if not geojson_path and not shapefile_path:
        return

    records = []
    for feat in polygons:
        rec = {k: v for k, v in feat.items() if k != "geometry"}
        rec["geometry"] = feat["geometry"]
        records.append(rec)

    if _GEOPANDAS_AVAILABLE:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        if geojson_path:
            gdf.to_file(geojson_path, driver="GeoJSON")
        if shapefile_path:
            gdf.to_file(shapefile_path)
        return

    if geojson_path:  # Fallback simple GeoJSON writer
        import json
        from shapely.geometry import mapping

        features = []
        for rec in records:
            geom = mapping(rec.pop("geometry"))
            features.append({"type": "Feature", "geometry": geom, "properties": rec})
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    if shapefile_path:
        raise RuntimeError("GeoPandas is required to write Shapefiles")


def segment_and_export(
    segmenter: SegmenterProtocol,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.2,
    class_names: Optional[Sequence[str]] = None,
    pixel_size: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    min_area: float = 0.0,
    geojson_path: Optional[str] = None,
    shapefile_path: Optional[str] = None,
    crs: str = "EPSG:4326",
) -> List[Dict[str, Any]]:
    """High level convenience function running segmentation and exporting.

    Returns the list of polygon features.
    """
    labels, confidence, _ = run_tiled_inference(segmenter, image, tile_size, overlap)
    polygons = mask_to_polygons(
        labels,
        confidence=confidence,
        class_names=class_names,
        pixel_size=pixel_size,
        origin=origin,
        min_area=min_area,
    )
    export_polygons(polygons, geojson_path, shapefile_path, crs=crs)
    return polygons


__all__ = [
    "generate_tiles",
    "run_tiled_inference",
    "mask_to_polygons",
    "export_polygons",
    "segment_and_export",
]
