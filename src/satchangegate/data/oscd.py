"""OSCD (Onera Satellite Change Detection) dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

# Official OSCD train/test split (14 train, 10 test)
OSCD_TRAIN_PAIRS = [
    "abudhabi", "aguasclaras", "beirut", "bordeaux", "brasilia", "chongqing",
    "cupertino", "mumbai", "nantes", "paris", "pisa", "saclay_w", "valencia", "vegas",
]
OSCD_TEST_PAIRS = [
    "dubai", "lasvegas", "melbourne", "montpellier", "norcia", "rio",
    "rotterdam", "saclay_e", "toronto", "toulouse",
]

DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


@dataclass
class ImagePair:
    """One OSCD bitemporal image pair."""

    pair_id: str
    split: Literal["train", "test"]
    img1_dir: Path
    img2_dir: Path
    label_path: Path | None


def _pair_dirs(root: Path, pair_id: str) -> tuple[Path, Path]:
    """Resolve img1/img2 directories for a pair."""
    pair_root = root / pair_id
    if (pair_root / "imgs_1").is_dir():
        return pair_root / "imgs_1", pair_root / "imgs_2"
    if (pair_root / "Osm_change_detection").is_dir():
        return pair_root / "Osm_change_detection" / "imgs_1", pair_root / "Osm_change_detection" / "imgs_2"
    # Flat layout: pair_id/ contains band tifs for t1, pair_id_2/ or nested
    if (pair_root / "B02.tif").is_file():
        return pair_root, pair_root
    raise FileNotFoundError(f"Cannot find img1/img2 dirs under {pair_root}")


def _label_path(root: Path, pair_id: str) -> Path | None:
    for candidate in [
        root / pair_id / "cm" / "cm.tif",
        root / pair_id / "Osm_change_detection" / "cm" / "cm.tif",
        root / pair_id / "label" / "cm.tif",
    ]:
        if candidate.is_file():
            return candidate
    return None


def list_pairs(root: Path, split: str = "all") -> list[ImagePair]:
    """List OSCD pairs under root for the given split."""
    root = Path(root)
    if split == "train":
        ids = OSCD_TRAIN_PAIRS
    elif split == "test":
        ids = OSCD_TEST_PAIRS
    else:
        ids = OSCD_TRAIN_PAIRS + OSCD_TEST_PAIRS

    pairs: list[ImagePair] = []
    for pair_id in ids:
        pair_root = root / pair_id
        if not pair_root.exists():
            # Try discovering by scanning
            continue
        try:
            img1, img2 = _pair_dirs(root, pair_id)
        except FileNotFoundError:
            continue
        sp: Literal["train", "test"] = "test" if pair_id in OSCD_TEST_PAIRS else "train"
        pairs.append(
            ImagePair(
                pair_id=pair_id,
                split=sp,
                img1_dir=img1,
                img2_dir=img2,
                label_path=_label_path(root, pair_id),
            )
        )
    return pairs


def discover_pairs(root: Path) -> list[ImagePair]:
    """Discover all pair directories under root."""
    root = Path(root)
    if not root.is_dir():
        return []
    pairs: list[ImagePair] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        pair_id = entry.name
        try:
            img1, img2 = _pair_dirs(root, pair_id)
        except FileNotFoundError:
            continue
        sp: Literal["train", "test"] = "test" if pair_id in OSCD_TEST_PAIRS else "train"
        pairs.append(
            ImagePair(
                pair_id=pair_id,
                split=sp,
                img1_dir=img1,
                img2_dir=img2,
                label_path=_label_path(root, pair_id),
            )
        )
    return pairs


def load_bands(pair_dir: Path, bands: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load Sentinel-2 band arrays from a pair timestep directory."""
    pair_dir = Path(pair_dir)
    bands = bands or DEFAULT_BANDS
    out: dict[str, np.ndarray] = {}
    for band in bands:
        path = pair_dir / f"{band}.tif"
        if not path.is_file():
            path = pair_dir / f"{band.lower()}.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Missing band {band} in {pair_dir}")
        with rasterio.open(path) as src:
            out[band] = src.read(1).astype(np.float32)
    return out


def load_label_mask(pair: ImagePair) -> np.ndarray | None:
    """Load pixel-level change mask (1=change, 0=no change)."""
    if pair.label_path is None or not pair.label_path.is_file():
        return None
    with rasterio.open(pair.label_path) as src:
        arr = src.read(1)
    return (arr > 0).astype(np.uint8)


def print_download_instructions(out: Path) -> None:
    """Print manual download steps for OSCD."""
    out = Path(out)
    print(
        """
OSCD dataset download (manual, one-time):

1. Register at https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection
2. Download and extract the dataset to:
   """
        + str(out.resolve())
        + """

Expected layout (one of):
  data/raw/oscd/<pair_id>/imgs_1/B02.tif ... imgs_2/B02.tif ... cm/cm.tif
  data/raw/oscd/<pair_id>/Osm_change_detection/imgs_1/ ...

Train pairs (14): """
        + ", ".join(OSCD_TRAIN_PAIRS)
        + """
Test pairs (10): """
        + ", ".join(OSCD_TEST_PAIRS)
        + """
"""
    )


def verify_layout(root: Path) -> tuple[bool, str]:
    """Check if OSCD root has at least one valid pair."""
    root = Path(root)
    pairs = discover_pairs(root) if root.exists() else list_pairs(root, "all")
    if not pairs:
        return False, f"No OSCD pairs found under {root}"
    p = pairs[0]
    try:
        load_bands(p.img1_dir, ["B02", "B04", "B08"])
    except FileNotFoundError as e:
        return False, str(e)
    return True, f"Found {len(pairs)} pair(s); first: {p.pair_id}"
