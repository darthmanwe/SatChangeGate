"""AlphaEarth Foundations embeddings as a landcover-context feature.

Google's Satellite Embedding dataset gives a 64-dimensional vector per 10 m
pixel per year, globally, precomputed and free (CC-BY 4.0). For a gate whose
selling point is ~10 ms per tile that is an unusually good fit: a strong learned
representation with **no GPU inference step** -- the embeddings are looked up,
not computed.

**What can and cannot be measured on this benchmark**

The dataset begins in 2017. OSCD's first acquisitions do not:

    23 of 24 cities have date_1 in 2015 or 2016; only chongqing (2017-04-14)
    falls inside coverage. Every date_2 (2017-2018) is covered.

So the interesting feature -- embedding dissimilarity between the two
acquisition years, which is what the dataset is explicitly designed for -- is
**not measurable on OSCD**. What is measurable is a single-date feature at t2,
and that is what ships as a result:

    emb_centroid_distance  how unlike the rest of its own scene this tile is
    emb_dispersion         how internally heterogeneous the tile is
    emb_norm               the magnitude of the tile's mean embedding

Note what is *not* here: the raw 64 dimensions. Those encode absolute location,
and a model given them could identify the city and key on that rather than on
change -- the same trap that keeps ``season_delta_days`` and
``registration_error_px`` out of ``GateFeatures``. Every column above is
scene-relative or scale-only by construction.

A caveat that belongs on the t2 feature and must travel with it: the annual
composite for a date_2 in 2017 spans all of 2017, so it contains information
from after the acquisition. That is acceptable for a *context* feature -- what
kind of place is this -- and would not be acceptable for a *change* feature.

**The probe.** ``embedding_change_probe`` computes the dissimilarity feature
anyway, substituting the 2017 embedding for a 2015 or 2016 t1. The 2017
composite already contains part or all of the change being detected, so any
lift it shows is an upper bound with unknown bias. It is reported under its own
heading, never in a results table, and never informs threshold selection. Its
only job is to size what the genuine bitemporal version might be worth once
imagery from 2017 onward is available.

**Running this needs credentials.** Earth Engine access is free for research,
education and nonprofit use but requires a Google Cloud project; the same data
also sits in ``gs://alphaearth_foundations`` under requester-pays. When neither
is configured every column below is ``None`` -- not zero. Unknown is not clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# First year of AlphaEarth coverage. Quoted rather than inferred because the
# whole design of this module turns on it.
FIRST_EMBEDDING_YEAR = 2017
EMBEDDING_DIMS = 64
DEFAULT_EMBEDDING_ROOT = Path("data/raw/embeddings")

# Columns that ship as a result. Scene-relative or scale-only by construction --
# see the module docstring on why the raw 64 dimensions are excluded.
EMBEDDING_FEATURE_NAMES = (
    "emb_centroid_distance",
    "emb_dispersion",
    "emb_norm",
)
# Reported separately and never mixed into the above.
PROBE_FEATURE_NAMES = ("emb_change_probe",)


class EmbeddingSource(Protocol):
    """Anything that can return a (H, W, 64) embedding block for an AOI-year."""

    def fetch(
        self, bbox: tuple[float, float, float, float], year: int, shape: tuple[int, int]
    ) -> np.ndarray | None: ...


def aoi_bbox(city_dir: Path) -> tuple[float, float, float, float] | None:
    """(min_lon, min_lat, max_lon, max_lat) from a city's AOI GeoJSON.

    OSCD's ``imgs_*_rect`` rasters carry no CRS or transform -- ``rasterio``
    reports the identity matrix -- so the AOI polygon is the only georeference
    available, and it is what maps a pixel grid onto the embedding grid.
    """
    matches = sorted(city_dir.glob("*.geojson"))
    if not matches:
        return None
    try:
        blob = json.loads(matches[0].read_text(encoding="utf-8"))
        coords = blob["features"][0]["geometry"]["coordinates"][0]
    except (KeyError, IndexError, ValueError):
        return None
    lons = [float(c[0]) for c in coords]
    lats = [float(c[1]) for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


@dataclass
class CachedNpySource:
    """Embeddings exported ahead of time to ``<root>/<city>_<year>.npy``.

    The offline path, and the one this repo can test. Exporting is a one-off:
    an Earth Engine or ``gsutil`` job writes a (H, W, 64) float32 array on the
    scene's own pixel grid.
    """

    root: Path = DEFAULT_EMBEDDING_ROOT
    city: str = ""

    def fetch(
        self,
        bbox: tuple[float, float, float, float],  # noqa: ARG002 - EmbeddingSource shape
        year: int,
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        # The export already covered this AOI, so the bbox is accepted for
        # protocol compatibility and not re-applied.
        path = Path(self.root) / f"{self.city}_{year}.npy"
        if not path.is_file():
            return None
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[2] != EMBEDDING_DIMS:
            raise ValueError(
                f"{path}: expected (H, W, {EMBEDDING_DIMS}) embeddings, got {arr.shape}"
            )
        if arr.shape[:2] != shape:
            arr = _resize_nearest(arr, shape)
        return arr.astype(np.float32)


@dataclass
class EarthEngineSource:
    """Live lookup against the Earth Engine Satellite Embedding collection.

    Requires ``earthengine-api`` and an initialised session. Kept behind a lazy
    import so neither the package nor a Google project is needed to install or
    run anything else in this repo.
    """

    collection: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    project: str | None = None
    scale_m: int = 10

    def fetch(
        self, bbox: tuple[float, float, float, float], year: int, shape: tuple[int, int]
    ) -> np.ndarray | None:
        try:
            import ee
        except ImportError:
            return None
        try:
            ee.Initialize(project=self.project)
            region = ee.Geometry.Rectangle(list(bbox))
            image = (
                ee.ImageCollection(self.collection)
                .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                .filterBounds(region)
                .first()
            )
            if image is None:
                return None
            arr = ee.data.computePixels(
                {
                    "expression": ee.Image(image).clip(region),
                    "fileFormat": "NUMPY_NDARRAY",
                    "grid": {"dimensions": {"width": shape[1], "height": shape[0]}},
                }
            )
        except Exception:
            # Auth, quota and coverage failures are all "no embedding available".
            # They must not take down a run whose classical path is unaffected.
            return None
        stacked = np.stack([arr[name] for name in arr.dtype.names], axis=-1)
        return stacked.astype(np.float32)


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a (H, W, C) block onto ``shape``."""
    rows = np.linspace(0, arr.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, arr.shape[1] - 1, shape[1]).round().astype(int)
    return arr[np.ix_(rows, cols)]


def embedding_features(
    tile_embedding: np.ndarray | None,
    scene_centroid: np.ndarray | None,
) -> dict[str, float | None]:
    """Scene-relative context features for one tile.

    Returns ``None`` for every column when no embedding is available, so a
    missing lookup reads as unknown rather than as a confident zero.
    """
    if tile_embedding is None or scene_centroid is None or tile_embedding.size == 0:
        return dict.fromkeys(EMBEDDING_FEATURE_NAMES, None)

    flat = tile_embedding.reshape(-1, tile_embedding.shape[-1])
    finite = flat[np.isfinite(flat).all(axis=1)]
    if finite.size == 0:
        return dict.fromkeys(EMBEDDING_FEATURE_NAMES, None)

    mean_vec = finite.mean(axis=0)
    return {
        # How unlike the rest of its own scene this tile is. Scene-relative, so
        # it says "unusual here" rather than "located here".
        "emb_centroid_distance": float(np.linalg.norm(mean_vec - scene_centroid)),
        # Internal heterogeneity: a uniform field is low, a built edge is high.
        "emb_dispersion": float(np.linalg.norm(finite - mean_vec, axis=1).mean()),
        "emb_norm": float(np.linalg.norm(mean_vec)),
    }


def scene_centroid(embedding: np.ndarray | None) -> np.ndarray | None:
    """Mean embedding over a whole scene, the reference for the features above."""
    if embedding is None or embedding.size == 0:
        return None
    flat = embedding.reshape(-1, embedding.shape[-1])
    finite = flat[np.isfinite(flat).all(axis=1)]
    return finite.mean(axis=0) if finite.size else None


def embedding_change_probe(
    tile_t1: np.ndarray | None,
    tile_t2: np.ndarray | None,
) -> dict[str, float | None]:
    """CONTAMINATED probe: embedding dissimilarity between two years.

    On OSCD the t1 year is before coverage begins, so callers substitute the
    earliest available year. That composite already contains the change being
    detected, which makes any measured lift an upper bound with unknown bias.
    Never report this alongside a result.
    """
    if tile_t1 is None or tile_t2 is None or tile_t1.size == 0 or tile_t2.size == 0:
        return dict.fromkeys(PROBE_FEATURE_NAMES, None)

    def mean_vec(a: np.ndarray) -> np.ndarray | None:
        flat = a.reshape(-1, a.shape[-1])
        finite = flat[np.isfinite(flat).all(axis=1)]
        return finite.mean(axis=0) if finite.size else None

    v1, v2 = mean_vec(tile_t1), mean_vec(tile_t2)
    if v1 is None or v2 is None:
        return dict.fromkeys(PROBE_FEATURE_NAMES, None)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-9:
        return dict.fromkeys(PROBE_FEATURE_NAMES, None)
    cosine = float(np.dot(v1, v2) / denom)
    return {"emb_change_probe": 1.0 - cosine}


def clamp_to_coverage(year: int) -> tuple[int, bool]:
    """(year to request, whether it had to be clamped).

    Clamping is what makes the probe contaminated, so the caller is told it
    happened rather than left to infer it.
    """
    if year >= FIRST_EMBEDDING_YEAR:
        return year, False
    return FIRST_EMBEDDING_YEAR, True


def coverage_report(pairs: list[Any]) -> dict[str, Any]:
    """How much of a dataset AlphaEarth can actually speak to.

    Produces the table that decided this module's shape, from the data rather
    than from a claim about it.
    """
    rows = []
    for pair in pairs:
        y1 = _year_of(getattr(pair, "date_t1", None))
        y2 = _year_of(getattr(pair, "date_t2", None))
        rows.append(
            {
                "city": pair.pair_id,
                "year_t1": y1,
                "year_t2": y2,
                "t1_covered": y1 is not None and y1 >= FIRST_EMBEDDING_YEAR,
                "t2_covered": y2 is not None and y2 >= FIRST_EMBEDDING_YEAR,
            }
        )
    n = len(rows) or 1
    return {
        "first_embedding_year": FIRST_EMBEDDING_YEAR,
        "n_pairs": len(rows),
        "n_t1_covered": sum(1 for r in rows if r["t1_covered"]),
        "n_t2_covered": sum(1 for r in rows if r["t2_covered"]),
        "n_bitemporal_measurable": sum(1 for r in rows if r["t1_covered"] and r["t2_covered"]),
        "pct_bitemporal_measurable": round(
            100.0 * sum(1 for r in rows if r["t1_covered"] and r["t2_covered"]) / n, 1
        ),
        "pairs": rows,
    }


def _year_of(date_str: str | None) -> int | None:
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(str(date_str)[:4])
    except ValueError:
        return None
