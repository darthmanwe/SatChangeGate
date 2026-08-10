"""Tiled evaluation set with a genuine negative class.

Evaluating OSCD at the pair level cannot produce a meaningful precision: all 24
cities were selected because they contain urban change, so every pair is
positive, `fp` can never increment, precision is pinned at 1.0 by construction,
and F1 collapses to 2r/(1+r) — a restatement of the candidate rate.

Tiling fixes that. Each city is cut into fixed-size tiles and labelled from the
pixel-level change mask the dataset already ships:

    positive  tile contains at least ``pos_min_fraction`` changed pixels
    negative  tile contains no changed pixels at all
    discarded anything in between (ambiguous, and would blur the boundary)

Positives and negatives come from the same scenes, the same acquisition dates,
and the same illumination — so the only systematic difference between the two
classes is the presence of change, which is exactly what the gate should key on.
The train/test split is inherited from the parent city, so no city contributes
tiles to both sides.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from satchangegate.data.oscd import ImagePair, discover_pairs, load_label_mask

# 64 px at 10 m GSD is a 640 m window — a sensible monitoring AOI, and small
# enough that unchanged ground genuinely exists inside these scenes. At 256 px
# almost every tile clips some change, which yields ~0 negatives and reproduces
# the degenerate pair-level metric this module exists to fix.
DEFAULT_TILE_SIZE = 64
# 0.5% of a 64x64 tile is ~20 pixels at 10 m: roughly one building footprint.
DEFAULT_POS_MIN_FRACTION = 0.005


@dataclass(frozen=True)
class Tile:
    """One labelled sub-window of a city pair."""

    tile_id: str
    city: str
    split: str
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int
    label: int
    change_fraction: float

    @property
    def slice_yx(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


def tiles_for_pair(
    pair: ImagePair,
    label: np.ndarray,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int | None = None,
    pos_min_fraction: float = DEFAULT_POS_MIN_FRACTION,
) -> list[Tile]:
    """Cut one pair into labelled tiles, discarding ambiguous ones."""
    stride = stride or tile_size
    height, width = label.shape
    out: list[Tile] = []
    for row, y0 in enumerate(range(0, height - tile_size + 1, stride)):
        for col, x0 in enumerate(range(0, width - tile_size + 1, stride)):
            window = label[y0 : y0 + tile_size, x0 : x0 + tile_size]
            frac = float(window.mean())
            if frac == 0.0:
                lbl = 0
            elif frac >= pos_min_fraction:
                lbl = 1
            else:
                continue  # ambiguous sliver of change; excluded from the set
            out.append(
                Tile(
                    tile_id=f"{pair.pair_id}_r{row}c{col}",
                    city=pair.pair_id,
                    split=pair.split,
                    row=row,
                    col=col,
                    y0=y0,
                    y1=y0 + tile_size,
                    x0=x0,
                    x1=x0 + tile_size,
                    label=lbl,
                    change_fraction=round(frac, 6),
                )
            )
    return out


def build_tile_index(
    root: Path | None = None,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int | None = None,
    pos_min_fraction: float = DEFAULT_POS_MIN_FRACTION,
) -> list[Tile]:
    """Build the full labelled tile index across all cities, deterministically."""
    tiles: list[Tile] = []
    for pair in discover_pairs(root):
        label = load_label_mask(pair)
        if label is None:
            continue
        tiles.extend(
            tiles_for_pair(
                pair,
                label,
                tile_size=tile_size,
                stride=stride,
                pos_min_fraction=pos_min_fraction,
            )
        )
    return sorted(tiles, key=lambda t: t.tile_id)


def summarise(tiles: list[Tile]) -> dict[str, dict[str, int]]:
    """Per-split positive/negative counts."""
    out: dict[str, dict[str, int]] = {}
    for tile in tiles:
        bucket = out.setdefault(tile.split, {"positive": 0, "negative": 0, "cities": 0})
        bucket["positive" if tile.label else "negative"] += 1
    for split in out:
        out[split]["cities"] = len({t.city for t in tiles if t.split == split})
    return out


def save_tile_index(tiles: list[Tile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_tiles": len(tiles),
        "summary": summarise(tiles),
        "tiles": [asdict(t) for t in tiles],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tile_index(path: Path) -> list[Tile]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Tile(**t) for t in data["tiles"]]
