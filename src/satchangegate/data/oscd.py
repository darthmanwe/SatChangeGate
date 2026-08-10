"""OSCD (Onera Satellite Change Detection) dataset loader.

Reads the full 13-band Sentinel-2 L1C release. Bands are returned as
top-of-atmosphere reflectance (raw DN divided by 10000), which is the unit every
threshold in ``configs/thresholds.yaml`` is expressed in. TOA reflectance is
nominally [0, 1] but legitimately exceeds 1 over bright targets such as cloud
tops and specular water, so values are deliberately not clipped.

Historical note: an earlier version of this module synthesised NIR and SWIR from
RGB previews as fixed linear combinations, e.g. ``nir = 0.6*green + 0.4*red``.
Measured against the real bands on Beirut, that made NDVI and NDWI 0.995
correlated (one feature under two names) and left the fake NDVI *negatively*
correlated (-0.49) with the true NDVI, detecting 0% vegetation where the real
index finds 12%. That path is deleted; only real bands are loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rasterio

from satchangegate.config import ALL_BANDS, GATE_BANDS, S2_REFLECTANCE_SCALE
from satchangegate.data.download import OSCD_TEST_CITIES, OSCD_TRAIN_CITIES

DEFAULT_BANDS = list(GATE_BANDS)
Split = Literal["train", "test"]


@dataclass
class ImagePair:
    """One OSCD bitemporal image pair."""

    pair_id: str
    split: Split
    img1_dir: Path
    img2_dir: Path
    label_path: Path | None
    date_t1: str | None = None
    date_t2: str | None = None


def default_oscd_root() -> Path:
    """Default OSCD dataset location (gitignored under data/raw/oscd)."""
    return Path("data/raw/oscd")


def load_splits(root: Path) -> tuple[set[str], set[str]]:
    """Train/test city sets.

    Prefers ``splits.json`` written by the downloader (derived from which label
    archive each city appears in); falls back to the packaged constants.
    """
    splits_file = root / "splits.json"
    if splits_file.is_file():
        data = json.loads(splits_file.read_text(encoding="utf-8"))
        return set(data.get("train", ())), set(data.get("test", ()))
    return set(OSCD_TRAIN_CITIES), set(OSCD_TEST_CITIES)


def _split_for_pair(pair_id: str, train: set[str], test: set[str]) -> Split:
    if pair_id in test:
        return "test"
    if pair_id in train:
        return "train"
    raise KeyError(
        f"City {pair_id!r} is in neither split. Re-run `satchangegate download-oscd` "
        "so splits.json is regenerated from the label archives."
    )


def _format_date(raw: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    raw = raw.strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _parse_dates(pair_root: Path) -> tuple[str | None, str | None]:
    dates_file = pair_root / "dates.txt"
    if not dates_file.is_file():
        return None, None
    d1, d2 = None, None
    for line in dates_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("date_1:"):
            d1 = _format_date(line.split(":", 1)[1])
        elif line.startswith("date_2:"):
            d2 = _format_date(line.split(":", 1)[1])
    return d1, d2


def _pair_sources(root: Path, pair_id: str) -> tuple[Path, Path]:
    """Resolve the t1/t2 band directories for a pair.

    ``imgs_*_rect`` holds the co-registered bands resampled to one common grid
    with plain ``B01.tif`` names. ``imgs_*`` holds the original per-band scene
    tiles at native 10/20/60 m resolutions under long product filenames, which
    are not directly stackable.
    """
    pair_root = root / pair_id
    rect1, rect2 = pair_root / "imgs_1_rect", pair_root / "imgs_2_rect"
    if rect1.is_dir() and rect2.is_dir():
        return rect1, rect2

    raw1, raw2 = pair_root / "imgs_1", pair_root / "imgs_2"
    if raw1.is_dir() and raw2.is_dir():
        return raw1, raw2

    raise FileNotFoundError(
        f"No band directories under {pair_root}. Run `satchangegate download-oscd`."
    )


def _label_path(root: Path, pair_id: str) -> Path | None:
    for candidate in (
        root / pair_id / "cm" / "cm.png",
        root / pair_id / "cm" / f"{pair_id}-cm.tif",
        root / pair_id / "cm" / "cm.tif",
    ):
        if candidate.is_file():
            return candidate
    return None


def _make_pair(root: Path, pair_id: str, train: set[str], test: set[str]) -> ImagePair | None:
    pair_root = root / pair_id
    if not pair_root.is_dir():
        return None
    try:
        img1, img2 = _pair_sources(root, pair_id)
        split = _split_for_pair(pair_id, train, test)
    except (FileNotFoundError, KeyError):
        return None
    d1, d2 = _parse_dates(pair_root)
    return ImagePair(
        pair_id=pair_id,
        split=split,
        img1_dir=img1,
        img2_dir=img2,
        label_path=_label_path(root, pair_id),
        date_t1=d1,
        date_t2=d2,
    )


def discover_pairs(root: Path | None = None) -> list[ImagePair]:
    """Discover all pair directories under root, in deterministic order."""
    root = Path(root or default_oscd_root())
    if not root.is_dir():
        return []
    train, test = load_splits(root)
    pairs: list[ImagePair] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        pair = _make_pair(root, entry.name, train, test)
        if pair is not None:
            pairs.append(pair)
    return pairs


def list_pairs(root: Path | None = None, split: str = "all") -> list[ImagePair]:
    """List OSCD pairs for the given split ('train', 'test', or 'all')."""
    if split not in ("train", "test", "all"):
        raise ValueError(f"split must be train|test|all, got {split!r}")
    pairs = discover_pairs(root)
    if split == "all":
        return pairs
    return [p for p in pairs if p.split == split]


def _band_file(source: Path, band: str) -> Path | None:
    """Locate a band's GeoTIFF, tolerating both rect and raw naming."""
    for name in (f"{band}.tif", f"{band.lower()}.tif"):
        candidate = source / name
        if candidate.is_file():
            return candidate
    # Raw scene tiles: '..._B01.tif'
    matches = sorted(source.glob(f"*_{band}.tif"))
    return matches[0] if matches else None


def load_bands(
    source: Path,
    bands: list[str] | None = None,
    *,
    scale: bool = True,
) -> dict[str, np.ndarray]:
    """Load bands from a timestep directory as float32 TOA reflectance.

    Args:
        source: A directory of per-band GeoTIFFs.
        bands: Band names to load; defaults to the six the gate consumes.
        scale: Divide raw DN by 10000 to yield reflectance in [0, 1].
    """
    source = Path(source)
    bands = list(bands or DEFAULT_BANDS)
    unknown = [b for b in bands if b not in ALL_BANDS]
    if unknown:
        raise ValueError(f"Unknown Sentinel-2 bands {unknown}; expected from {list(ALL_BANDS)}")
    if not source.is_dir():
        raise NotADirectoryError(f"Band source is not a directory: {source}")

    out: dict[str, np.ndarray] = {}
    for band in bands:
        path = _band_file(source, band)
        if path is None:
            raise FileNotFoundError(f"Missing band {band} in {source}")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        if scale:
            arr = arr / np.float32(S2_REFLECTANCE_SCALE)
        out[band] = arr
    return out


def load_label_mask(pair: ImagePair) -> np.ndarray | None:
    """Load the binary change mask for a pair, or None when absent."""
    if pair.label_path is None or not pair.label_path.is_file():
        return None
    path = pair.label_path
    if path.suffix.lower() == ".png":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return (img > 127).astype(np.uint8)
    with rasterio.open(path) as src:
        arr = src.read(1)
    return (arr > 0).astype(np.uint8)


def verify_layout(root: Path | None = None) -> tuple[bool, str]:
    """Check that root looks like a usable OSCD tree."""
    root = Path(root or default_oscd_root())
    if not root.is_dir():
        return False, f"Missing OSCD root {root}. Run: satchangegate download-oscd"
    pairs = discover_pairs(root)
    if not pairs:
        return False, f"No OSCD pairs under {root}. Run: satchangegate download-oscd"
    labelled = sum(1 for p in pairs if p.label_path is not None)
    n_train = sum(1 for p in pairs if p.split == "train")
    n_test = len(pairs) - n_train
    return True, f"{len(pairs)} pairs ({n_train} train / {n_test} test), {labelled} labelled"
