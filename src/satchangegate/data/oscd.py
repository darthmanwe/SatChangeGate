"""OSCD (Onera Satellite Change Detection) dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rasterio

# Fallback split when train.txt / test.txt are absent (full multispectral release)
OSCD_TRAIN_PAIRS = [
    "abudhabi", "aguasclaras", "beirut", "bordeaux", "brasilia", "chongqing",
    "cupertino", "mumbai", "nantes", "paris", "pisa", "saclay_w", "valencia", "vegas",
]
OSCD_TEST_PAIRS = [
    "dubai", "lasvegas", "melbourne", "montpellier", "norcia", "rio",
    "rotterdam", "saclay_e", "toronto", "toulouse",
]

DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
Layout = Literal["multispectral", "rgb_png"]


@dataclass
class ImagePair:
    """One OSCD bitemporal image pair."""

    pair_id: str
    split: Literal["train", "test"]
    img1_dir: Path
    img2_dir: Path
    label_path: Path | None
    layout: Layout = "multispectral"
    date_t1: str | None = None
    date_t2: str | None = None


def default_oscd_root() -> Path:
    """Default OSCD dataset location (gitignored under data/raw/oscd)."""
    return Path("data/raw/oscd")


def _load_split_lists(root: Path) -> tuple[set[str], set[str]]:
    train: set[str] = set()
    test: set[str] = set()
    train_file = root / "train.txt"
    test_file = root / "test.txt"
    if train_file.is_file():
        train = {s.strip() for s in train_file.read_text(encoding="utf-8").split(",") if s.strip()}
    if test_file.is_file():
        test = {s.strip() for s in test_file.read_text(encoding="utf-8").split(",") if s.strip()}
    return train, test


def _split_for_pair(
    pair_id: str,
    train: set[str],
    test: set[str],
) -> Literal["train", "test"]:
    if pair_id in test:
        return "test"
    if pair_id in train:
        return "train"
    if pair_id in OSCD_TEST_PAIRS:
        return "test"
    return "train"


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


def _pair_sources(root: Path, pair_id: str) -> tuple[Path, Path, Layout]:
    """Resolve t1/t2 sources and layout for a pair."""
    pair_root = root / pair_id

    pair_png = pair_root / "pair"
    img1_png = pair_png / "img1.png"
    img2_png = pair_png / "img2.png"
    if img1_png.is_file() and img2_png.is_file():
        return img1_png, img2_png, "rgb_png"

    if (pair_root / "imgs_1").is_dir():
        return pair_root / "imgs_1", pair_root / "imgs_2", "multispectral"

    if (pair_root / "Osm_change_detection").is_dir():
        base = pair_root / "Osm_change_detection"
        return base / "imgs_1", base / "imgs_2", "multispectral"

    if (pair_root / "B02.tif").is_file():
        return pair_root, pair_root, "multispectral"

    raise FileNotFoundError(f"Cannot find image pair under {pair_root}")


def _label_path(root: Path, pair_id: str) -> Path | None:
    for candidate in [
        root / pair_id / "cm" / "cm.png",
        root / pair_id / "cm" / f"{pair_id}-cm.tif",
        root / pair_id / "cm" / "cm.tif",
        root / pair_id / "Osm_change_detection" / "cm" / "cm.tif",
        root / pair_id / "label" / "cm.tif",
    ]:
        if candidate.is_file():
            return candidate
    return None


def _make_pair(root: Path, pair_id: str, train: set[str], test: set[str]) -> ImagePair | None:
    pair_root = root / pair_id
    if not pair_root.is_dir():
        return None
    try:
        img1, img2, layout = _pair_sources(root, pair_id)
    except FileNotFoundError:
        return None
    d1, d2 = _parse_dates(pair_root)
    return ImagePair(
        pair_id=pair_id,
        split=_split_for_pair(pair_id, train, test),
        img1_dir=img1,
        img2_dir=img2,
        label_path=_label_path(root, pair_id),
        layout=layout,
        date_t1=d1,
        date_t2=d2,
    )


def list_pairs(root: Path | None = None, split: str = "all") -> list[ImagePair]:
    """List OSCD pairs under root for the given split."""
    root = Path(root or default_oscd_root())
    train, test = _load_split_lists(root)
    if train or test:
        ids = sorted(train | test)
    elif split == "train":
        ids = OSCD_TRAIN_PAIRS
    elif split == "test":
        ids = OSCD_TEST_PAIRS
    else:
        ids = OSCD_TRAIN_PAIRS + OSCD_TEST_PAIRS

    pairs: list[ImagePair] = []
    for pair_id in ids:
        pair = _make_pair(root, pair_id, train, test)
        if pair is None:
            continue
        if split != "all" and pair.split != split:
            continue
        pairs.append(pair)
    return pairs


def discover_pairs(root: Path | None = None) -> list[ImagePair]:
    """Discover all pair directories under root."""
    root = Path(root or default_oscd_root())
    if not root.is_dir():
        return []
    train, test = _load_split_lists(root)
    pairs: list[ImagePair] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        pair = _make_pair(root, entry.name, train, test)
        if pair is not None:
            pairs.append(pair)
    return pairs


def _png_to_pseudo_bands(path: Path, bands: list[str]) -> dict[str, np.ndarray]:
    """Map RGB PNG preview to pseudo Sentinel-2 bands for index/gate math."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    nir = np.clip(0.6 * green + 0.4 * red, 0.0, 1.0)
    swir1 = np.clip(0.7 * red + 0.2 * green + 0.1 * blue, 0.0, 1.0)
    swir2 = np.clip(0.8 * red + 0.15 * green, 0.0, 1.0)
    mapping = {
        "B02": blue,
        "B03": green,
        "B04": red,
        "B08": nir,
        "B11": swir1,
        "B12": swir2,
    }
    return {band: mapping[band] for band in bands}


def load_bands(source: Path, bands: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load band arrays from a timestep directory or RGB PNG file."""
    source = Path(source)
    bands = bands or DEFAULT_BANDS

    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            return _png_to_pseudo_bands(source, bands)
        if suffix in {".tif", ".tiff"}:
            with rasterio.open(source) as src:
                arr = src.read(1).astype(np.float32)
            return {bands[0]: arr}

    if (source / "img1.png").is_file():
        raise ValueError(f"Ambiguous PNG dir {source}; pass img1.png or img2.png path")

    out: dict[str, np.ndarray] = {}
    for band in bands:
        path = source / f"{band}.tif"
        if not path.is_file():
            path = source / f"{band.lower()}.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Missing band {band} in {source}")
        with rasterio.open(path) as src:
            out[band] = src.read(1).astype(np.float32)
    return out


def load_label_mask(pair: ImagePair) -> np.ndarray | None:
    """Load pixel-level change mask (1=change, 0=no change)."""
    path = pair.label_path
    if path is None or not path.is_file():
        path = pair.img1_dir.parent.parent / "cm" / "cm.png"
    if not path.is_file():
        return None
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return (img > 127).astype(np.uint8)
    with rasterio.open(path) as src:
        arr = src.read(1)
    return (arr > 0).astype(np.uint8)


def print_download_instructions(out: Path | None = None) -> None:
    """Print manual download steps for OSCD."""
    out = Path(out or default_oscd_root())
    print(
        """
OSCD dataset download (manual, one-time):

1. Register at https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection
2. Download and extract the dataset to:
   """
        + str(out.resolve())
        + """

Supported layouts:
  <root>/<city>/pair/img1.png + img2.png   (IEEE Images zip — RGB previews)
  <root>/<city>/imgs_1/B02.tif ... imgs_2/ ... cm/cm.tif   (full multispectral)
  <root>/<city>/cm/cm.png                  (train/test label zips merged per city)

IEEE provides three separate downloads:
  1) Images zip (~489 MB) — may be RGB previews OR multispectral depending on archive version
  2) Train Labels zip
  3) Test Labels zip

Split files: train.txt, test.txt at dataset root.
"""
    )


def verify_layout(root: Path | None = None) -> tuple[bool, str]:
    """Check if OSCD root has at least one valid pair."""
    root = Path(root or default_oscd_root())
    pairs = discover_pairs(root) or list_pairs(root, "all")
    if not pairs:
        return False, f"No OSCD pairs found under {root}"
    p = pairs[0]
    try:
        load_bands(p.img1_dir, ["B02", "B04", "B08"])
    except (FileNotFoundError, ValueError) as e:
        return False, str(e)
    labeled = sum(1 for pair in pairs if pair.label_path is not None)
    layout_note = f"layout={p.layout}"
    label_note = f"{labeled}/{len(pairs)} with pixel labels"
    return True, f"Found {len(pairs)} pair(s); first: {p.pair_id} ({layout_note}; {label_note})"
