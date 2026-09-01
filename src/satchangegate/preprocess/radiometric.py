"""Relative radiometric normalization via pseudo-invariant features.

The README names the dominant false-positive mode on this benchmark plainly:
OSCD labels only *urban* change, while seasonal, agricultural and hydrological
change across a multi-year gap is spectrally enormous and labelled negative. A
magnitude-based detector keys on the wrong signal. The scene-level adaptive
threshold already cancels a scene-wide *additive* offset; what it cannot cancel
is a multiplicative one, or a shift that differs between bands -- which is what
a different sun angle, atmosphere, and phenological state actually produce.

Pseudo-invariant feature (PIF) matching is the standard answer. Fit a per-band
gain and offset that map the second acquisition onto the first, using **only the
pixels that did not change**, then difference. The circularity is obvious and is
the whole difficulty: identifying unchanged pixels is the problem being solved.
The standard resolution, used here, is to iterate -- start from a permissive
guess at the unchanged population, fit, recompute change against the normalized
image, re-select, refit. Two or three passes is where it settles.

**This is off by default.** It is a real intervention on the input, and whether
it helps on this benchmark is an empirical question the repo should answer with a
measurement rather than an assumption. Turn it on with ``preprocess.normalize:
true`` and compare the precision-recall curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bands whose reflectance is genuinely comparable between acquisitions after
# gain/offset correction. Every band the gate consumes qualifies; the list exists
# so a caller passing a wider stack does not silently normalise a quality band.
DEFAULT_ITERATIONS = 3
# Share of pixels treated as unchanged on the first pass. Deliberately generous:
# starting too tight biases the fit toward whatever the initial guess considered
# stable, which is the failure mode PIF selection is prone to.
DEFAULT_INVARIANT_PERCENTILE = 60.0
# Below this many usable pixels a per-band regression is noise. The pair is
# returned unchanged rather than normalised against nothing.
MIN_INVARIANT_PIXELS = 256


@dataclass
class NormalizationReport:
    """What the normalization did, so it can be inspected rather than trusted."""

    applied: bool
    iterations: int
    n_invariant: int
    gains: dict[str, float]
    offsets: dict[str, float]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "iterations": self.iterations,
            "n_invariant_pixels": self.n_invariant,
            "gains": {k: round(v, 5) for k, v in self.gains.items()},
            "offsets": {k: round(v, 5) for k, v in self.offsets.items()},
            "reason": self.reason,
        }


def _robust_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Gain and offset mapping ``x`` onto ``y``, resistant to outliers.

    Ordinary least squares would be pulled by the changed pixels that inevitably
    survive selection. This uses a two-point robust estimator on quantiles --
    the line through the 25th and 75th percentile pairs -- which ignores the
    tails entirely and needs no iteration of its own.
    """
    if x.size < MIN_INVARIANT_PIXELS:
        return 1.0, 0.0
    x_lo, x_hi = np.percentile(x, [25.0, 75.0])
    y_lo, y_hi = np.percentile(y, [25.0, 75.0])
    spread = x_hi - x_lo
    if not np.isfinite(spread) or abs(spread) < 1e-6:
        return 1.0, 0.0
    gain = float((y_hi - y_lo) / spread)
    # A gain far from 1 means the two acquisitions disagree about dynamic range
    # by more than illumination can explain; clamp rather than amplify noise.
    gain = float(np.clip(gain, 0.5, 2.0))
    offset = float(np.median(y) - gain * np.median(x))
    return gain, offset


def _initial_change_proxy(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    names: list[str],
) -> np.ndarray:
    """Per-pixel spectral distance, used only to rank pixels by stability."""
    stack1 = np.stack([bands_t1[b] for b in names], axis=0)
    stack2 = np.stack([bands_t2[b] for b in names], axis=0)
    return np.sqrt(((stack2 - stack1) ** 2).sum(axis=0)).astype(np.float32)


def normalize_to_reference(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    valid: np.ndarray | None = None,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    invariant_percentile: float = DEFAULT_INVARIANT_PERCENTILE,
) -> tuple[dict[str, np.ndarray], NormalizationReport]:
    """Map ``bands_t2`` onto ``bands_t1``'s radiometry using unchanged pixels.

    The first acquisition is the reference and is returned untouched, so indices
    computed from it keep their physical meaning and any threshold expressed in
    reflectance still applies.
    """
    names = sorted(set(bands_t1) & set(bands_t2))
    if not names:
        return dict(bands_t2), NormalizationReport(False, 0, 0, {}, {}, "no bands in common")

    shape = bands_t1[names[0]].shape
    mask = np.ones(shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    for name in names:
        mask &= np.isfinite(bands_t1[name]) & np.isfinite(bands_t2[name])

    if int(mask.sum()) < MIN_INVARIANT_PIXELS:
        return dict(bands_t2), NormalizationReport(
            False, 0, int(mask.sum()), {}, {}, "too few valid pixels to fit"
        )

    out = {name: np.asarray(bands_t2[name], dtype=np.float32).copy() for name in names}
    gains = dict.fromkeys(names, 1.0)
    offsets = dict.fromkeys(names, 0.0)
    n_invariant = 0
    done = 0

    for _ in range(max(1, iterations)):
        proxy = _initial_change_proxy(bands_t1, out, names)
        cut = float(np.percentile(proxy[mask], invariant_percentile))
        invariant = mask & (proxy <= cut)
        n_invariant = int(invariant.sum())
        if n_invariant < MIN_INVARIANT_PIXELS:
            break

        # Refit from the *original* t2 each pass, not from the last iterate, so
        # gains compose once rather than compounding across iterations.
        for name in names:
            gain, offset = _robust_fit(
                np.asarray(bands_t2[name], dtype=np.float32)[invariant],
                np.asarray(bands_t1[name], dtype=np.float32)[invariant],
            )
            gains[name], offsets[name] = gain, offset
            out[name] = (np.asarray(bands_t2[name], dtype=np.float32) * gain + offset).astype(
                np.float32
            )
        done += 1

    # Bands present in only one timestep pass through untouched.
    for name in bands_t2:
        out.setdefault(name, np.asarray(bands_t2[name], dtype=np.float32))

    applied = done > 0
    return out, NormalizationReport(
        applied=applied,
        iterations=done,
        n_invariant=n_invariant,
        gains=gains,
        offsets=offsets,
        reason="" if applied else "no iteration found enough invariant pixels",
    )
