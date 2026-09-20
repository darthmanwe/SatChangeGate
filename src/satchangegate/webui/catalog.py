"""The scene catalogue: what imagery exists, and where on Earth it is.

Georeferencing here is read, not inferred. That is a correction. OSCD's
``imgs_*_rect`` rasters -- the ones the pipeline analyses -- carry no CRS, and
``data/embeddings.aoi_bbox`` used to state that the AOI polygon was therefore the
only georeference available. It is not: every city's *unrectified* ``imgs_1``
band carries an EPSG:4326 transform, and on this dataset those arrays are
byte-identical to their rectified counterparts, with bounds matching the polygon
exactly. So the transform is authoritative and the polygon is the fallback, in
that order, and each scene records which one it used.

Everything here reads file headers and directory entries only. No pixels, so a
catalogue of 24 cities costs milliseconds and the map can render before anything
heavy has been computed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from satchangegate.data.oscd import default_oscd_root, discover_pairs

BboxSource = Literal["native_transform", "aoi_polygon", "none"]

# Web Mercator cannot represent an equirectangular grid exactly. At OSCD's AOI
# sizes the disagreement is small, but "small" is a measurement, not a promise --
# `mercator_skew_px` reports it per scene rather than letting the UI claim it.


@dataclass(frozen=True)
class Scene:
    """One bitemporal AOI, described without reading a single pixel."""

    city: str
    split: str
    height: int
    width: int
    date_t1: str | None
    date_t2: str | None
    bbox: tuple[float, float, float, float] | None
    bbox_source: BboxSource
    crs: str | None
    has_label: bool
    previews: dict[str, bool] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def center(self) -> tuple[float, float] | None:
        if self.bbox is None:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)

    def pixel_to_lonlat(self, x: float, y: float) -> tuple[float, float] | None:
        """Pixel centre -> (lon, lat). None when the scene has no georeference."""
        if self.bbox is None or not self.width or not self.height:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bbox
        lon = min_lon + (x + 0.5) * (max_lon - min_lon) / self.width
        lat = max_lat - (y + 0.5) * (max_lat - min_lat) / self.height
        return (lon, lat)

    def tile_bounds(self, y0: int, x0: int, y1: int, x1: int) -> list[list[float]] | None:
        """A tile's [[south, west], [north, east]] for a Leaflet overlay."""
        if self.bbox is None:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bbox
        west = min_lon + x0 * (max_lon - min_lon) / self.width
        east = min_lon + x1 * (max_lon - min_lon) / self.width
        north = max_lat - y0 * (max_lat - min_lat) / self.height
        south = max_lat - y1 * (max_lat - min_lat) / self.height
        return [[south, west], [north, east]]

    @property
    def mercator_skew_px(self) -> float | None:
        """Vertical error, in scene pixels, from drawing this equirectangular
        grid on a Web Mercator basemap.

        The grid is linear in latitude; Mercator is not. Overlaying one on the
        other stretches the image by an amount that grows with latitude and with
        the AOI's north-south extent. Reported so the UI can state the number
        rather than assert that it is negligible.
        """
        if self.bbox is None or not self.height:
            return None
        import math

        _, min_lat, _, max_lat = self.bbox
        if max_lat == min_lat:
            return 0.0

        def merc_y(lat: float) -> float:
            lat = max(min(lat, 85.05), -85.05)
            return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

        span = merc_y(max_lat) - merc_y(min_lat)
        mid_linear = (min_lat + max_lat) / 2.0
        mid_merc_frac = (merc_y(mid_linear) - merc_y(min_lat)) / span if span else 0.5
        return abs(mid_merc_frac - 0.5) * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "split": self.split,
            "height": self.height,
            "width": self.width,
            "date_t1": self.date_t1,
            "date_t2": self.date_t2,
            "bbox": list(self.bbox) if self.bbox else None,
            "bbox_source": self.bbox_source,
            "crs": self.crs,
            "center": list(self.center) if self.center else None,
            "has_label": self.has_label,
            "previews": dict(self.previews),
            "mercator_skew_px": (
                round(self.mercator_skew_px, 2) if self.mercator_skew_px is not None else None
            ),
            "notes": list(self.notes),
        }


def _native_geo(city_dir: Path) -> tuple[tuple[float, float, float, float] | None, str | None]:
    """(bbox, crs) from the unrectified red band, which carries a real transform."""
    candidates = sorted(city_dir.glob("imgs_1/*B04.tif"))
    if not candidates:
        return None, None
    try:
        import rasterio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(candidates[0]) as src:
                if src.crs is None:
                    return None, None
                b = src.bounds
                return (float(b.left), float(b.bottom), float(b.right), float(b.top)), str(src.crs)
    except Exception:  # pragma: no cover - a malformed raster is not fatal here
        return None, None


def _rect_shape(city_dir: Path) -> tuple[int, int]:
    """(height, width) of the grid the pipeline actually analyses."""
    for pattern in ("imgs_1_rect/B04.tif", "imgs_1_rect/b04.tif", "imgs_1/*B04.tif"):
        for path in sorted(city_dir.glob(pattern)):
            try:
                import rasterio

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with rasterio.open(path) as src:
                        return int(src.height), int(src.width)
            except Exception:  # pragma: no cover
                continue
    return 0, 0


def build_catalogue(root: Path | None = None) -> list[Scene]:
    """Every discoverable pair, with georeferencing and preview availability."""
    root = Path(root or default_oscd_root())
    scenes: list[Scene] = []
    for pair in discover_pairs(root):
        city_dir = root / pair.pair_id
        height, width = _rect_shape(city_dir)
        bbox, crs = _native_geo(city_dir)
        source: BboxSource = "native_transform"
        notes: list[str] = []
        if bbox is None:
            from satchangegate.data.embeddings import aoi_bbox

            bbox = aoi_bbox(city_dir)
            source = "aoi_polygon" if bbox else "none"
            if bbox:
                notes.append("No CRS on the native raster; falling back to the AOI polygon.")
        scenes.append(
            Scene(
                city=pair.pair_id,
                split=pair.split,
                height=height,
                width=width,
                date_t1=pair.date_t1,
                date_t2=pair.date_t2,
                bbox=bbox,
                bbox_source=source,
                crs=crs,
                has_label=pair.label_path is not None,
                previews={
                    "t1": (city_dir / "pair" / "img1.png").is_file(),
                    "t2": (city_dir / "pair" / "img2.png").is_file(),
                },
                notes=tuple(notes),
            )
        )
    return _annotate_adjacency(scenes)


def _annotate_adjacency(scenes: list[Scene]) -> list[Scene]:
    """Flag scenes that share an edge, because two of them straddle the split.

    ``saclay_e`` and ``saclay_w`` are halves of one Sentinel-2 acquisition, and
    they land on opposite sides of the train/test boundary. That is worth saying
    out loud on a map rather than leaving for someone to notice.
    """
    out: list[Scene] = []
    for scene in scenes:
        notes = list(scene.notes)
        if scene.bbox is not None:
            for other in scenes:
                if other.city == scene.city or other.bbox is None:
                    continue
                if _shares_edge(scene.bbox, other.bbox):
                    across = " across the train/test split" if other.split != scene.split else ""
                    notes.append(f"Edge-adjacent to {other.city}{across}.")
        out.append(
            Scene(**{**scene.__dict__, "notes": tuple(notes)})
            if notes != list(scene.notes)
            else scene
        )
    return out


def _shares_edge(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-6) -> bool:
    a_min_lon, a_min_lat, a_max_lon, a_max_lat = a
    b_min_lon, b_min_lat, b_max_lon, b_max_lat = b
    lat_overlap = min(a_max_lat, b_max_lat) - max(a_min_lat, b_min_lat) > tol
    lon_overlap = min(a_max_lon, b_max_lon) - max(a_min_lon, b_min_lon) > tol
    touches_lon = abs(a_max_lon - b_min_lon) < tol or abs(b_max_lon - a_min_lon) < tol
    touches_lat = abs(a_max_lat - b_min_lat) < tol or abs(b_max_lat - a_min_lat) < tol
    return (touches_lon and lat_overlap) or (touches_lat and lon_overlap)


@lru_cache(maxsize=4)
def cached_catalogue(root_str: str) -> tuple[Scene, ...]:
    """Catalogue for one root, computed once per process."""
    return tuple(build_catalogue(Path(root_str)))
