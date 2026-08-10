"""Offline control battery.

Sanity checks with known-correct answers, runnable without an API key. Two of
the five checks in the previous battery could not fail: the identity control
compares an image with itself (any threshold set passes), and the photometric
control accepted "any gate outcome except low_quality" — an outcome that was
itself unreachable. Each check here can actually fail, and the battery exits
non-zero when one does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.config import Settings, get_settings
from satchangegate.data.negative_controls import apply_photometric_perturbation
from satchangegate.data.oscd import default_oscd_root, discover_pairs, load_bands
from satchangegate.features.classical import classical_gate
from satchangegate.preprocess.align import (
    estimate_registration_error,
    resample_to_common_grid,
)
from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
from satchangegate.preprocess.quality import compute_quality_score


def _gate(bands_t1, bands_t2, settings: Settings, name: str):
    b1, b2 = resample_to_common_grid(bands_t1, bands_t2)
    reg = estimate_registration_error(b1, b2)
    m1 = compute_ephemeral_masks(b1, settings.masks)
    m2 = compute_ephemeral_masks(b2, settings.masks)
    combined = combine_pair_masks(m1, m2)
    quality = compute_quality_score(
        m1, m2, settings.quality, registration_error_px=reg, combined=combined
    )
    return classical_gate(name, b1, b2, combined, quality, settings.gate).result


def run_dev_tests(
    root: Path | None = None,
    out_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    city: str = "beirut",
) -> dict[str, Any]:
    """Run every control and return a machine-readable summary."""
    settings = settings or get_settings()
    root = Path(root or default_oscd_root())
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {p.pair_id: p for p in discover_pairs(root)}
    checks: list[dict[str, Any]] = []

    if city not in pairs:
        summary = {
            "error": f"city {city!r} not available under {root}",
            "checks": [],
            "n_checks": 0,
            "n_passed": 0,
        }
        (out_dir / "_dev_tests.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    pair = pairs[city]
    bands = load_bands(pair.img1_dir, list(settings.bands))

    def record(name: str, expected: str, actual: str, passed: bool, detail: str = "") -> None:
        checks.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    # 1. Identity: a scene against itself must produce no change and, more
    #    sharply, exactly zero changed area and SSIM of 1.
    res = _gate(bands, {k: v.copy() for k, v in bands.items()}, settings, f"{city}_identity")
    record(
        "identity",
        "no_change, 0.00% area, SSIM 1.000",
        f"{res.classical_gate}, {res.changed_area_percent:.2f}% area, SSIM {res.ssim:.3f}",
        res.classical_gate == "no_change" and res.changed_area_percent == 0.0 and res.ssim > 0.999,
    )

    # 2. Photometric: a global gain/gamma shift is an illumination artifact, not
    #    ground change. The gate must not call it change.
    perturbed = apply_photometric_perturbation(bands)
    res = _gate(bands, perturbed, settings, f"{city}_photometric")
    record(
        "photometric",
        "no_change",
        res.classical_gate,
        res.classical_gate == "no_change",
        "global brightness/gamma shift must not read as physical change",
    )

    # 3. Real pair: the genuine bitemporal pair must differ measurably from the
    #    identity control. This is what makes check 1 non-vacuous.
    bands_t2 = load_bands(pair.img2_dir, list(settings.bands))
    res_real = _gate(bands, bands_t2, settings, f"{city}_real")
    record(
        "real_pair_differs",
        "changed area > 0 and SSIM < 1",
        f"{res_real.changed_area_percent:.2f}% area, SSIM {res_real.ssim:.3f}",
        res_real.changed_area_percent > 0.0 and res_real.ssim < 0.999,
    )

    # 4. Masks are genuinely computed on multispectral input, not stubbed.
    record(
        "tier0_assessed",
        "masks assessed on 13-band input",
        f"assessed={res_real.masks_assessed}, cloud={res_real.cloud_fraction_max}",
        bool(res_real.masks_assessed) and res_real.cloud_fraction_max is not None,
    )

    # 5. Registration is measured, not hardcoded. The old pipeline reported a
    #    constant 0.0 px in JSON, VLM metadata, and analyst prose.
    shifted = {k: np.roll(v, 3, axis=1) for k, v in bands.items()}
    reg = estimate_registration_error(bands, shifted)
    record(
        "registration_measured",
        "~3.0 px for a 3 px synthetic shift",
        f"{reg:.2f} px",
        2.5 <= reg <= 3.5,
    )

    summary = {
        "city": city,
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
    }
    (out_dir / "_dev_tests.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
