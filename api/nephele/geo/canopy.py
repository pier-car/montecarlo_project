"""Urban canopy ingestion: building footprints to roughness/obstacle fields.

Building footprints (typically loaded in QGIS and exported as GeoJSON or a
vector layer) are rasterised onto a regular grid to provide ground/obstacle
height used by the dispersion engine's reflecting boundaries and to derive an
aerodynamic roughness length.

GDAL/OGR are optional: if unavailable, GeoJSON is parsed with the standard
library so the module remains importable in minimal environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class UrbanCanopy:
    """Rasterised obstacle-height field over a regular horizontal grid."""

    origin: tuple[float, float]
    cell_size: float
    height: np.ndarray  # shape (ny, nx), obstacle/ground height [m]

    def __post_init__(self) -> None:
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be > 0")
        if self.height.ndim != 2:
            raise ValueError("height must be a 2-D array")

    def ground_height(self, x: float, y: float) -> float:
        """Obstacle height at world coordinate (x, y); 0 outside the grid."""
        ny, nx = self.height.shape
        ix = int((x - self.origin[0]) / self.cell_size)
        iy = int((y - self.origin[1]) / self.cell_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            return float(self.height[iy, ix])
        return 0.0

    def roughness_length(self) -> float:
        """Macroscopic aerodynamic roughness z0 ~ 0.1 * mean building height."""
        mean_h = float(self.height[self.height > 0].mean()) if np.any(
            self.height > 0
        ) else 0.0
        return max(0.03, 0.1 * mean_h)


def _read_geojson(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("type") != "FeatureCollection":
        raise ValueError("expected a GeoJSON FeatureCollection")
    return list(data.get("features", []))


def _polygon_bbox(coords: list) -> tuple[float, float, float, float]:
    xs = [pt[0] for ring in coords for pt in ring]
    ys = [pt[1] for ring in coords for pt in ring]
    return min(xs), min(ys), max(xs), max(ys)


def load_building_footprints(
    path: str | Path,
    cell_size: float,
    default_height: float = 12.0,
    height_property: str = "height",
) -> UrbanCanopy:
    """Rasterise building-footprint polygons into an :class:`UrbanCanopy`.

    Parameters
    ----------
    path:
        GeoJSON file of (Multi)Polygon building footprints.
    cell_size:
        Raster cell size in the same units as the geometry (metres).
    default_height:
        Height assigned when a feature lacks the height property.
    height_property:
        Name of the per-feature property holding building height.
    """
    path = Path(path)
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    if not path.exists():
        raise FileNotFoundError(f"footprint file not found: {path}")

    features = _read_geojson(path)
    if not features:
        raise ValueError("no features found in footprint file")

    polygons: list[tuple[list, float]] = []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        height = float(feat.get("properties", {}).get(height_property, default_height))
        if gtype == "Polygon":
            rings = geom["coordinates"]
            polygons.append((rings, height))
        elif gtype == "MultiPolygon":
            for poly in geom["coordinates"]:
                polygons.append((poly, height))
        else:
            continue
        bx0, by0, bx1, by1 = _polygon_bbox(polygons[-1][0])
        minx, miny = min(minx, bx0), min(miny, by0)
        maxx, maxy = max(maxx, bx1), max(maxy, by1)

    if not polygons:
        raise ValueError("no polygon geometries found")

    nx = max(1, int(np.ceil((maxx - minx) / cell_size)))
    ny = max(1, int(np.ceil((maxy - miny) / cell_size)))
    grid = np.zeros((ny, nx), dtype=float)

    # Rasterise by point-in-polygon test on cell centres (ray casting).
    for rings, height in polygons:
        outer = rings[0]
        px0, py0, px1, py1 = _polygon_bbox([outer])
        ix0 = max(0, int((px0 - minx) / cell_size))
        ix1 = min(nx, int(np.ceil((px1 - minx) / cell_size)))
        iy0 = max(0, int((py0 - miny) / cell_size))
        iy1 = min(ny, int(np.ceil((py1 - miny) / cell_size)))
        for iy in range(iy0, iy1):
            cy = miny + (iy + 0.5) * cell_size
            for ix in range(ix0, ix1):
                cx = minx + (ix + 0.5) * cell_size
                if _point_in_ring(cx, cy, outer):
                    grid[iy, ix] = max(grid[iy, ix], height)

    return UrbanCanopy(origin=(minx, miny), cell_size=cell_size, height=grid)


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    """Ray-casting point-in-polygon test for a single linear ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        ):
            inside = not inside
        j = i
    return inside
