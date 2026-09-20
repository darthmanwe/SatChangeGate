"""Turning an uploaded pair of images into something the funnel may judge.

``run_from_bands`` takes two dictionaries of arrays and asks nothing else, which
makes it a clean seam and a dangerous one. ``resample_to_common_grid`` resizes
the second timestep to the first one's *shape*; it does not transform
coordinates, intersect footprints, or preserve nodata. Two equal-sized GeoTIFFs
of different continents satisfy every shape check and produce a confident,
meaningless answer.

So this module does not "accept images". It accepts a **declaration** and
verifies the files against it. Nothing physical is inferred:

- **Bands are named, not guessed.** Band 4 of an arbitrary GeoTIFF is not NDVI's
  red channel just because it is fourth.
- **Reflectance scaling is declared.** Dividing by 10000 is right for Sentinel-2
  L1C and wrong for almost everything else, and dtype does not reveal which you
  have. Values above 1.0 are kept: bright targets legitimately exceed it.
- **Footprints must actually overlap**, in a shared CRS, and the pair is cropped
  to the intersection rather than stretched onto a common shape.
- **Nodata becomes invalid**, not zero. Filling it with zero fabricates a clean
  observation, which is the failure this repo names as its own rule.
- **Resolution is stated.** Pixel-based morphology, component sizes, context
  windows and a 1.5 px registration tolerance all change physical meaning with
  ground sample distance; the same thresholds at 30 m mean something different.

What cannot be verified is refused, not approximated. A narrow contract that
says no is worth more here than a broad one that guesses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from satchangegate.config import ALL_BANDS, GATE_BANDS, S2_REFLECTANCE_SCALE

Alignment = Literal["georeferenced", "already_aligned"]

#: Raster drivers that read exactly the file handed to them. VRT, and anything
#: else that can point at other paths or URLs, is refused: an upload must not be
#: able to make the server fetch something.
ALLOWED_DRIVERS = frozenset({"GTiff", "PNG", "JPEG"})

#: Sentinel-2 L1C band order as the archive ships it, for the common case.
S2_BAND_ORDER = ALL_BANDS


@dataclass(frozen=True)
class Limits:
    """Bounds applied before anything large is decoded.

    Compressed size is not a memory bound: a few hundred kilobytes of PNG can
    decode to gigabytes, so pixels and bands are checked from the header first.
    """

    max_file_bytes: int = 512 * 1024 * 1024
    max_pixels: int = 120_000_000
    max_bands: int = 16
    max_files: int = 64
    max_total_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class Declaration:
    """What the uploader states. Verified, never inferred."""

    band_map: dict[str, int]
    reflectance_scale: float = S2_REFLECTANCE_SCALE
    reflectance_offset: float = 0.0
    date_t1: str | None = None
    date_t2: str | None = None
    alignment: Alignment = "georeferenced"
    target_resolution_m: float | None = None
    source_name: str = "upload"

    def validate(self) -> None:
        unknown = sorted(set(self.band_map) - set(ALL_BANDS))
        if unknown:
            raise IngestRefused(
                f"Unknown band names: {', '.join(unknown)}. Names follow the "
                f"Sentinel-2 convention ({', '.join(ALL_BANDS[:4])}, ...), because "
                f"every threshold in this repo is expressed against those bands."
            )
        if not self.band_map:
            raise IngestRefused("No bands declared; a raster index per band is required.")
        if self.reflectance_scale <= 0:
            raise IngestRefused("reflectance_scale must be positive.")


@dataclass
class Capability:
    """What this source can and cannot support, decided from the bands present."""

    bands: tuple[str, ...]
    spectral_indices: bool
    masks: bool
    full_gate: bool
    missing_for_gate: tuple[str, ...]
    lane: Literal["full", "rgb_only"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bands": list(self.bands),
            "spectral_indices": self.spectral_indices,
            "masks": self.masks,
            "full_gate": self.full_gate,
            "missing_for_gate": list(self.missing_for_gate),
            "lane": self.lane,
            "explanation": (
                "Every gate threshold is defined over NDVI, NDBI and NDWI, which "
                "need near-infrared and short-wave infrared. Without them the gate "
                "refuses rather than scoring on a subset."
                if not self.full_gate
                else "All bands the gate needs are present."
            ),
        }


@dataclass
class IngestedPair:
    """A verified bitemporal pair, ready for ``run_from_bands``."""

    bands_t1: dict[str, np.ndarray]
    bands_t2: dict[str, np.ndarray]
    valid: np.ndarray
    capability: Capability
    crs: str | None
    bbox: tuple[float, float, float, float] | None
    resolution_m: float | None
    height: int
    width: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "crs": self.crs,
            "bbox": list(self.bbox) if self.bbox else None,
            "resolution_m": self.resolution_m,
            "valid_fraction": round(float(self.valid.mean()), 4),
            "capability": self.capability.to_dict(),
            "warnings": list(self.warnings),
        }


class IngestRefused(ValueError):
    """The upload cannot be verified, and says exactly why."""


# --------------------------------------------------------------------- reading


def _open(path: Path, limits: Limits) -> Any:
    import rasterio

    size = path.stat().st_size
    if size > limits.max_file_bytes:
        raise IngestRefused(
            f"{path.name} is {size / 1e6:.1f} MB, over the {limits.max_file_bytes / 1e6:.0f} MB limit."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        src = rasterio.open(path)
    driver = str(src.driver)
    if driver not in ALLOWED_DRIVERS:
        src.close()
        raise IngestRefused(
            f"{path.name} is a {driver} raster. Only {', '.join(sorted(ALLOWED_DRIVERS))} "
            f"are read, because formats that can reference other files or URLs would "
            f"let an upload decide what this server opens next."
        )
    if src.count > limits.max_bands:
        src.close()
        raise IngestRefused(
            f"{path.name} has {src.count} bands, over the limit of {limits.max_bands}."
        )
    if src.height * src.width > limits.max_pixels:
        src.close()
        raise IngestRefused(
            f"{path.name} is {src.width}x{src.height} = "
            f"{src.height * src.width / 1e6:.0f} Mpx, over the "
            f"{limits.max_pixels / 1e6:.0f} Mpx limit. Compressed size is not a "
            f"memory bound, so this is checked from the header."
        )
    return src


def _rotated(transform: Any) -> bool:
    """True when the grid is not axis-aligned.

    A rotated raster cannot be described by a bounding box, so placing it on a
    map from one would put its pixels in the wrong place. Refused rather than
    silently mislocated.
    """
    return bool(
        abs(getattr(transform, "b", 0.0)) > 1e-9 or abs(getattr(transform, "d", 0.0)) > 1e-9
    )


def read_pair(
    path_t1: Path,
    path_t2: Path,
    declaration: Declaration,
    *,
    limits: Limits | None = None,
) -> IngestedPair:
    """Verify two rasters against a declaration and return aligned bands."""
    limits = limits or Limits()
    declaration.validate()

    src1 = _open(Path(path_t1), limits)
    src2 = _open(Path(path_t2), limits)
    try:
        notes: list[str] = []
        for name, index in declaration.band_map.items():
            for src, label in ((src1, "t1"), (src2, "t2")):
                if index < 1 or index > src.count:
                    raise IngestRefused(
                        f"Band {name} was declared at raster index {index}, but the "
                        f"{label} image has {src.count} band(s)."
                    )

        if declaration.alignment == "already_aligned":
            if (src1.height, src1.width) != (src2.height, src2.width):
                raise IngestRefused(
                    f"Declared already-aligned, but the images are "
                    f"{src1.width}x{src1.height} and {src2.width}x{src2.height}. "
                    f"Aligned means pixel-for-pixel; resizing one to match would "
                    f"invent a correspondence rather than verify one."
                )
            window1 = window2 = None
            crs = None
            bbox = None
            notes.append(
                "Declared already-aligned: no map position is claimed, and none is "
                "invented. Registration is still measured, not assumed."
            )
        else:
            crs, bbox, window1, window2 = _georeferenced_overlap(src1, src2, notes)

        resolution = _resolution(src1, declaration, notes)
        bands_t1, valid1 = _read_bands(src1, declaration, window1)
        bands_t2, valid2 = _read_bands(src2, declaration, window2)

        shape1 = next(iter(bands_t1.values())).shape
        shape2 = next(iter(bands_t2.values())).shape
        if shape1 != shape2:
            raise IngestRefused(
                f"After cropping to the shared footprint the two images are "
                f"{shape1} and {shape2}. They do not describe the same ground."
            )

        valid = valid1 & valid2
        if not valid.any():
            raise IngestRefused(
                "No pixel is valid in both images once nodata is accounted for. "
                "There is nothing here to compare."
            )
        if valid.mean() < 0.5:
            notes.append(
                f"Only {valid.mean():.0%} of pixels carry data in both images; Tier 0 "
                f"will refuse this pair on valid fraction."
            )

        capability = _capability(tuple(sorted(bands_t1)))
        if capability.lane == "rgb_only":
            notes.append(
                "This source cannot support the gate. Structural evidence is still "
                "computed and can still be sent for verification, but no gate "
                "decision is produced."
            )
        return IngestedPair(
            bands_t1=bands_t1,
            bands_t2=bands_t2,
            valid=valid,
            capability=capability,
            crs=crs,
            bbox=bbox,
            resolution_m=resolution,
            height=int(shape1[0]),
            width=int(shape1[1]),
            warnings=notes,
        )
    finally:
        src1.close()
        src2.close()


def _georeferenced_overlap(src1: Any, src2: Any, notes: list[str]) -> tuple[Any, Any, Any, Any]:
    from rasterio.coords import disjoint_bounds
    from rasterio.windows import from_bounds

    for src, label in ((src1, "t1"), (src2, "t2")):
        if src.crs is None:
            raise IngestRefused(
                f"The {label} image has no CRS. Declare `already_aligned` if the two "
                f"are known to be pixel-for-pixel, or supply georeferenced rasters -- "
                f"equal dimensions alone do not mean equal ground."
            )
        if _rotated(src.transform):
            raise IngestRefused(
                f"The {label} image has a rotated transform. A bounding box cannot "
                f"describe where its pixels are, so it would be placed wrongly on a "
                f"map. Reproject it to an axis-aligned grid first."
            )
    if src1.crs != src2.crs:
        raise IngestRefused(
            f"The two images are in different coordinate systems ({src1.crs} and "
            f"{src2.crs}). Reproject them onto one grid; this will not guess which."
        )
    if disjoint_bounds(src1.bounds, src2.bounds):
        raise IngestRefused(
            "The two images do not overlap on the ground. Equal pixel dimensions do "
            "not make two rasters comparable."
        )

    left = max(src1.bounds.left, src2.bounds.left)
    bottom = max(src1.bounds.bottom, src2.bounds.bottom)
    right = min(src1.bounds.right, src2.bounds.right)
    top = min(src1.bounds.top, src2.bounds.top)
    bounds = (left, bottom, right, top)

    full1 = src1.bounds.right - src1.bounds.left
    shared = right - left
    if full1 > 0 and shared / full1 < 0.99:
        notes.append(
            f"Cropped to the shared footprint: {shared / full1:.0%} of the first "
            f"image's width. Only ground seen at both dates is compared."
        )
    window1 = from_bounds(*bounds, transform=src1.transform)
    window2 = from_bounds(*bounds, transform=src2.transform)
    return str(src1.crs), bounds, window1, window2


def _resolution(src: Any, declaration: Declaration, notes: list[str]) -> float | None:
    declared = declaration.target_resolution_m
    if declared is not None:
        if declared <= 0:
            raise IngestRefused("target_resolution_m must be positive.")
        if abs(declared - 10.0) > 1e-6:
            notes.append(
                f"Declared ground resolution is {declared} m, not the 10 m these "
                f"thresholds were fitted at. Pixel-based despeckling, component "
                f"sizes and the 1.5 px registration tolerance all change physical "
                f"meaning with resolution; read any result with that in mind."
            )
        return declared
    if src.crs is not None and src.transform is not None:
        return abs(float(src.transform.a))
    notes.append(
        "Ground resolution unknown and undeclared, so the physical meaning of the "
        "pixel-based thresholds here is unknown too."
    )
    return None


def _read_bands(
    src: Any, declaration: Declaration, window: Any
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Read declared bands as reflectance, and the mask of pixels that carry data."""
    bands: dict[str, np.ndarray] = {}
    valid: np.ndarray | None = None
    for name, index in sorted(declaration.band_map.items()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = src.read(index, window=window, boundless=False).astype(np.float32)
            observed = src.read_masks(index, window=window, boundless=False) > 0
        # Scale, never clip: bright targets legitimately exceed 1.0, and clipping
        # them is how a cloud top becomes indistinguishable from bare concrete.
        bands[name] = (raw - declaration.reflectance_offset) / declaration.reflectance_scale
        finite = np.isfinite(bands[name])
        layer = observed & finite
        valid = layer if valid is None else (valid & layer)
    assert valid is not None
    return bands, valid


def _capability(bands: tuple[str, ...]) -> Capability:
    from satchangegate.features.classical import INDEX_BANDS
    from satchangegate.preprocess.masks import REQUIRED_BANDS

    present = set(bands)
    has_indices = set(INDEX_BANDS) <= present
    has_masks = set(REQUIRED_BANDS) <= present
    missing = tuple(sorted(set(GATE_BANDS) - present))
    full = not missing
    return Capability(
        bands=bands,
        spectral_indices=has_indices,
        masks=has_masks,
        full_gate=full,
        missing_for_gate=missing,
        lane="full" if has_indices else "rgb_only",
    )


def rgb_declaration(**overrides: Any) -> Declaration:
    """The common three-band case, named explicitly rather than sniffed.

    An 8-bit RGB render is not reflectance and this does not pretend otherwise:
    the scale is 255 so values land in [0, 1], and the capability check will put
    the pair in the degraded lane, where the gate refuses and says why.
    """
    base = {
        "band_map": {"B04": 1, "B03": 2, "B02": 3},
        "reflectance_scale": 255.0,
        "alignment": "already_aligned",
    }
    base.update(overrides)
    return Declaration(**base)  # type: ignore[arg-type]
