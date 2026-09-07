"""Typed project configuration.

Thresholds live in exactly one place: the pydantic models below define every
default, and ``configs/thresholds.yaml`` (packaged alongside this module)
overrides them. Consumers read attributes off a validated ``Settings`` object
rather than calling ``dict.get(key, default)`` at each site — the previous
approach let four call-site defaults silently disagree with the shipped YAML.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Sentinel-2 L1C reflectance is distributed as integer DN with this scale factor.
# Dividing by it yields top-of-atmosphere reflectance, the unit every threshold
# in this file is expressed in. Nominally [0, 1], but bright targets (cloud
# tops, specular water) legitimately exceed 1, so values are never clipped.
S2_REFLECTANCE_SCALE = 10_000.0

# The six bands the classical gate consumes. Names follow the Sentinel-2 MSI
# convention: B02 blue, B03 green, B04 red, B08 NIR, B11 SWIR-1, B12 SWIR-2.
GATE_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

# Full Sentinel-2 L1C band set as shipped in the OSCD archive.
ALL_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)

RGB_BANDS = ("B04", "B03", "B02")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MaskThresholds(_Base):
    """Ephemeral-mask thresholds, in TOA reflectance units."""

    cloud_brightness: float = 0.35
    cloud_ndvi_max: float = 0.20
    snow_ndsi_min: float = 0.40
    snow_nir_min: float = 0.25
    water_ndwi_min: float = 0.30
    shadow_nir_max: float = 0.15
    shadow_brightness_max: float = 0.12
    # Cloud-shadow search radius in pixels at 10 m GSD. A cloud at height h casts
    # its shadow h*tan(solar_zenith) away, so this is a coarse proxy: the previous
    # hardcoded 7x7 kernel could only ever find shadows within 30 m of a cloud.
    shadow_search_radius_px: int = 30
    # s2cloudless probability above which a pixel is called cloud, when the
    # optional s2cloudless extra is installed.
    cloud_probability_min: float = 0.40


class QualityThresholds(_Base):
    cloud_fraction_max: float = 0.25
    # quality_score is the usable-pixel fraction, so this is the only floor.
    # A separate min_quality_score would be the same check twice.
    min_valid_pixel_fraction: float = 0.50
    # Maximum tolerated co-registration error before a pair is called unusable.
    registration_error_px_max: float = 1.5


class PreprocessSettings(_Base):
    """Optional corrections applied before any index is differenced."""

    # Relative radiometric normalization via pseudo-invariant features. Off by
    # default: it is a real intervention on the input, and whether it helps on a
    # given benchmark is an empirical question, not an assumption. Compare the
    # precision-recall curves with it on and off before enabling it anywhere.
    normalize: bool = False
    pif_iterations: int = 3
    pif_invariant_percentile: float = 60.0


class GateThresholds(_Base):
    """Classical change-gate decision thresholds.

    Fitted by ``satchangegate tune --split train`` on the 14 training cities,
    with the 10 test cities held out. Every field here is read by ``decide`` or
    ``compute_change_mask``; the previous version carried eleven keys that no
    code path consumed, left behind by a decision ladder that had been replaced.
    """

    min_changed_area_percent: float = 25.0

    ndvi_delta_mean_min: float = 0.10
    ndbi_delta_mean_min: float = 0.08
    ndwi_delta_mean_min: float = 0.15
    ndvi_strong_min: float = 0.12
    ndbi_strong_min: float = 0.10

    ssim_no_change_min: float = 0.65
    phash_no_change_max: int = 16

    # Directional evidence. NDBI up while NDVI goes down is the physically
    # correct signature of urban change, and the gate previously could not
    # express it: `decide` read only absolute index deltas, which erase sign, so
    # construction and demolition looked identical to it.
    urbanization_score_min: float = 0.10
    # Concentrated evidence. A tile whose mean delta is unremarkable but whose
    # upper tail is decisive is exactly the "one new building in a quiet tile"
    # case that a mean-only feature vector cannot see.
    magnitude_p95_min: float = 0.20
    # Minimum size of the single largest contiguous changed region. Real change
    # is contiguous; what survives despeckling otherwise tends to be seams.
    min_largest_component_px: int = 64

    # Per-pixel change-mask thresholding.
    #
    # A single absolute reflectance threshold cannot serve every scene: the
    # scene-wide radiometric offset between two acquisitions (season, sun angle,
    # atmosphere) is common to changed and unchanged pixels alike, so a fixed
    # cut flags 0.7% of one city and 75% of another. Thresholding at a
    # percentile of each scene's own delta distribution removes that offset and
    # measured +50% pixel F1 on the train split (0.083 -> 0.126).
    adaptive_threshold: bool = True
    # Background model: threshold at median + background_sigma * robust-sigma of
    # the per-pixel change magnitude. Robust statistics rather than a high
    # percentile, which assumes change is rare and self-suppresses when it isn't.
    background_sigma: float = 2.0
    # Absolute floor beneath the adaptive cut. Without it a percentile always
    # selects its top N% of pixels, so a scene where nothing changed still
    # yields a full change mask and a global illumination shift reads as change.
    min_absolute_delta: float = 0.10
    # Ceiling on the adaptive cut, so a noisy scene cannot raise the bar past
    # the point where unmistakable change would be missed.
    max_absolute_delta: float = 0.30
    index_delta_small: float = 0.10
    cva_magnitude_threshold: float = 0.24
    cva_magnitude_mean_min: float = 0.28

    # Minimum connected-component size (pixels) retained in the change mask.
    # Without this the mask is a per-pixel OR of correlated thresholds and
    # single-pixel compression speckle counts toward changed area.
    min_component_size_px: int = 32
    # Morphological opening radius applied before component filtering.
    open_radius_px: int = 1


class ScorerSettings(_Base):
    """Which Tier-1 scorer runs.

    Deliberately not a field on ``GateThresholds``: everything in that model is a
    threshold read by ``decide``, and a test asserts exactly that so a dead
    config key cannot reappear. This is a runtime mode, not a threshold.

    ``rules`` stays the default. The learned scorer measurably wins on accuracy
    (+0.14 average precision on held-out cities), and the rules still win on
    properties that are worth real money to a spend-control filter: every
    decision returns the rule that fired, there is no runtime dependency, and
    there is no model artifact to version, retrain, or watch drift. The point of
    this setting is that the trade is now the operator's to make.
    """

    kind: Literal["rules", "learned"] = "rules"
    path: str = "data/models/gate_scorer.pkl"
    # Operating point for the learned scorer. `satchangegate conformal` can pick
    # this with a distribution-free guarantee instead of by eye.
    threshold: float = 0.5


class Settings(_Base):
    masks: MaskThresholds = Field(default_factory=MaskThresholds)
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    gate: GateThresholds = Field(default_factory=GateThresholds)
    preprocess: PreprocessSettings = Field(default_factory=PreprocessSettings)
    scorer: ScorerSettings = Field(default_factory=ScorerSettings)
    bands: tuple[str, ...] = GATE_BANDS
    rgb_bands: tuple[str, ...] = RGB_BANDS


def _packaged_thresholds() -> Path:
    """Path to the YAML shipped inside the installed package."""
    return Path(str(resources.files("satchangegate") / "configs" / "thresholds.yaml"))


def find_project_root() -> Path | None:
    """Nearest ancestor directory containing a pyproject.toml, if any.

    Used only for locating a developer's .env; never for locating package data.
    """
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (never overrides existing)."""
    env_path = path
    if env_path is None:
        root = find_project_root()
        if root is None:
            return
        env_path = root / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings(path: Path | None = None) -> Settings:
    """Load and validate thresholds. Unknown keys raise rather than pass silently."""
    cfg_path = Path(path) if path is not None else _packaged_thresholds()
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if loaded:
            raw = loaded
    return Settings.model_validate(raw)


@lru_cache(maxsize=8)
def _cached_settings(key: str | None) -> Settings:
    return load_settings(Path(key) if key else None)


def get_settings(path: Path | None = None) -> Settings:
    """Cached settings accessor. Previously re-read the YAML once per pair."""
    return _cached_settings(str(path) if path is not None else None)
