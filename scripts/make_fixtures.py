"""Regenerate the committed synthetic fixtures.

The fixtures are synthetic rather than crops of OSCD so they can be committed
without inheriting the dataset's licence, and so a clean clone can run the
pipeline end to end with no download. Values are physically plausible
top-of-atmosphere reflectances for the six bands the gate consumes.

The scene contains a known change: a rectangular block converts from vegetation
to built-up between t1 and t2, which is the signature the gate is built to find
(NDVI falls, NDBI rises). A second "stable" pair contains no change at all.

Run: python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini_oscd"
SIZE = 128
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
SCALE = 10_000.0

# Representative TOA reflectance by surface type: blue, green, red, NIR, SWIR1, SWIR2
VEGETATION = (0.030, 0.060, 0.035, 0.320, 0.150, 0.070)
BUILT_UP = (0.120, 0.130, 0.150, 0.180, 0.240, 0.200)
SOIL = (0.090, 0.110, 0.140, 0.210, 0.260, 0.210)
WATER = (0.045, 0.040, 0.028, 0.015, 0.008, 0.006)


def _base_scene(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A scene of vegetation with a soil strip and a water body."""
    out = {b: np.empty((SIZE, SIZE), dtype=np.float32) for b in BANDS}
    for i, band in enumerate(BANDS):
        arr = np.full((SIZE, SIZE), VEGETATION[i], dtype=np.float32)
        arr[:, 96:] = SOIL[i]
        arr[100:, :40] = WATER[i]
        arr += rng.normal(0.0, 0.004, size=arr.shape).astype(np.float32)
        out[band] = np.clip(arr, 0.0, 1.5)
    return out


def _apply_change(scene: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert a rectangle from vegetation to built-up; return scene and label."""
    changed = {b: a.copy() for b, a in scene.items()}
    label = np.zeros((SIZE, SIZE), dtype=np.uint8)
    y0, y1, x0, x1 = 24, 72, 20, 68
    for i, band in enumerate(BANDS):
        changed[band][y0:y1, x0:x1] = BUILT_UP[i]
    label[y0:y1, x0:x1] = 1
    return changed, label


def _write_bands(bands: dict[str, np.ndarray], out_dir: Path) -> None:
    """Write each band as a uint16 DN GeoTIFF, matching the real OSCD layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0.0, 0.0, 10.0, 10.0)
    for name, arr in bands.items():
        dn = np.clip(arr * SCALE, 0, 65535).astype(np.uint16)
        with rasterio.open(
            out_dir / f"{name}.tif",
            "w",
            driver="GTiff",
            height=SIZE,
            width=SIZE,
            count=1,
            dtype="uint16",
            crs="EPSG:32636",
            transform=transform,
        ) as dst:
            dst.write(dn, 1)


def _write_label(label: np.ndarray, out_dir: Path) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "cm.png"), (label * 255).astype(np.uint8))


def build() -> Path:
    rng = np.random.default_rng(20260810)
    root = FIXTURE_ROOT
    root.mkdir(parents=True, exist_ok=True)

    # City with a real change.
    base = _base_scene(rng)
    changed, label = _apply_change(base)
    _write_bands(base, root / "fixtureville" / "imgs_1_rect")
    _write_bands(changed, root / "fixtureville" / "imgs_2_rect")
    _write_label(label, root / "fixtureville" / "cm")
    (root / "fixtureville" / "dates.txt").write_text(
        "date_1: 20200115\ndate_2: 20220310\n", encoding="utf-8"
    )

    # City with no change: an independent noise draw only.
    stable_t1 = _base_scene(rng)
    stable_t2 = {
        b: a + rng.normal(0.0, 0.004, a.shape).astype(np.float32) for b, a in stable_t1.items()
    }
    _write_bands(stable_t1, root / "stableton" / "imgs_1_rect")
    _write_bands(stable_t2, root / "stableton" / "imgs_2_rect")
    _write_label(np.zeros((SIZE, SIZE), dtype=np.uint8), root / "stableton" / "cm")
    (root / "stableton" / "dates.txt").write_text(
        "date_1: 20200115\ndate_2: 20220310\n", encoding="utf-8"
    )

    (root / "splits.json").write_text(
        '{\n  "train": ["fixtureville"],\n  "test": ["stableton"]\n}\n', encoding="utf-8"
    )
    return root


if __name__ == "__main__":
    path = build()
    print(f"Fixtures written to {path}")
