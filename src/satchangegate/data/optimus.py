"""OPTIMUS Sentinel-2 time-series loader (eval subset; full corpus is multi-TB)."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from satchangegate.data.oscd import load_bands

DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
HF_REPO = "optimus-change/optimus-dataset"
# Each images/{N}.tar is ~25 GB on Hugging Face; use dev fixtures for local work.
TAR_SIZE_GB_WARN = 25


@dataclass
class OptimusSeries:
    """One OPTIMUS tile time series reference."""

    tile_id: str
    label: Literal[0, 1]  # 0=no persistent change, 1=change
    group_index: int | None
    tar_path: Path | None
    extracted_root: Path | None


def default_optimus_root() -> Path:
    return Path("data/raw/optimus")


def _tile_png_name(tile_id: str) -> str:
    return f"{tile_id}.png"


def load_index(root: Path | None = None) -> list[dict[str, list[str]]]:
    """Load index.json mapping tar groups to tile pngs and available months."""
    root = Path(root or default_optimus_root())
    path = root / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: satchangegate download-optimus --with-index")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_group_index(tile_id: str, root: Path | None = None) -> int:
    """Return images/{index}.tar group for a tile id."""
    return build_tile_group_lookup(root)[tile_id]


def build_tile_group_lookup(root: Path | None = None) -> dict[str, int]:
    """Map tile_id -> tar group index (cached per root)."""
    root = Path(root or default_optimus_root())
    cache_path = root / ".cache" / "tile_group_lookup.json"
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in raw.items()}

    index = load_index(root)
    lookup: dict[str, int] = {}
    for gi, group in enumerate(index):
        for png in group:
            lookup[png.replace(".png", "")] = gi
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(lookup), encoding="utf-8")
    return lookup


def list_eval_tiles_in_group(group_index: int, root: Path | None = None) -> list[tuple[str, int]]:
    """Return (tile_id, label) for eval-labeled tiles inside one tar group."""
    root = Path(root or default_optimus_root())
    labels = load_eval_labels(root)
    index = load_index(root)
    group = index[group_index]
    out: list[tuple[str, int]] = []
    for png in group:
        tile_id = png.replace(".png", "")
        if tile_id in labels:
            out.append((tile_id, labels[tile_id]))
    return sorted(out, key=lambda x: x[0])


def extract_group_tar(group_index: int, root: Path | None = None) -> Path:
    """Extract images/{group_index}.tar from local disk if present."""
    root = Path(root or default_optimus_root())
    out_dir = root / "extracted" / str(group_index)
    tar_path = root / "images" / f"{group_index}.tar"
    if not tar_path.is_file():
        raise FileNotFoundError(f"Missing local tar: {tar_path}")
    marker = out_dir / ".extract_complete"
    if marker.is_file() and out_dir.is_dir() and any(out_dir.rglob("*.png")):
        return out_dir
    if out_dir.is_dir():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")
    return out_dir


def load_eval_labels(root: Path | None = None) -> dict[str, int]:
    root = Path(root or default_optimus_root())
    path = root / "2024_dataset_evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: satchangegate download-optimus --metadata-only"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_timestamps_from_index(tile_id: str, root: Path | None = None) -> list[str]:
    root = Path(root or default_optimus_root())
    index = load_index(root)
    name = _tile_png_name(tile_id)
    for group in index:
        if name in group:
            return sorted(group[name])
    raise KeyError(f"Tile {tile_id} not in index.json")


def list_eval_series(
    root: Path | None = None,
    *,
    label: int | None = None,
    limit: int | None = None,
) -> list[OptimusSeries]:
    root = Path(root or default_optimus_root())
    labels = load_eval_labels(root)
    items: list[OptimusSeries] = []
    for tile_id, lab in labels.items():
        if label is not None and lab != label:
            continue
        group_index: int | None = None
        tar_path: Path | None = None
        extracted: Path | None = None
        try:
            group_index = resolve_group_index(tile_id, root)
            tar_path = root / "images" / f"{group_index}.tar"
            if not tar_path.is_file():
                tar_path = None
            extracted_dir = root / "extracted" / str(group_index)
            if extracted_dir.is_dir() and any(extracted_dir.rglob(_tile_png_name(tile_id))):
                extracted = extracted_dir
        except FileNotFoundError:
            pass
        except KeyError:
            pass
        items.append(
            OptimusSeries(
                tile_id=tile_id,
                label=lab,  # type: ignore[arg-type]
                group_index=group_index,
                tar_path=tar_path,
                extracted_root=extracted,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def download_optimus_metadata(root: Path | None = None) -> Path:
    """Download evaluation JSON and README via huggingface_hub."""
    root = Path(root or default_optimus_root())
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for filename in ("2024_dataset_evaluation.json", "README.md"):
        src = hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")
        dst = root / filename
        dst.write_bytes(Path(src).read_bytes())
    return root


def download_optimus_index(root: Path | None = None) -> Path:
    """Download index.json (~464 MB) for tar group lookup without pulling imagery."""
    root = Path(root or default_optimus_root())
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "index.json"
    if dest.is_file():
        return dest
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(repo_id=HF_REPO, filename="index.json", repo_type="dataset")
    dest.write_bytes(Path(src).read_bytes())
    return dest


def download_optimus_tar(
    tile_id: str,
    root: Path | None = None,
    *,
    allow_large_download: bool = False,
) -> Path:
    """Download one images/{group_index}.tar containing tile_id (~25 GB each)."""
    root = Path(root or default_optimus_root())
    (root / "images").mkdir(parents=True, exist_ok=True)
    download_optimus_index(root)
    group_index = resolve_group_index(tile_id, root)
    dest = root / "images" / f"{group_index}.tar"
    if dest.is_file():
        return dest
    if not allow_large_download:
        raise RuntimeError(
            f"Refusing to download images/{group_index}.tar (~{TAR_SIZE_GB_WARN} GB). "
            "Use --allow-large-download or create a dev fixture: "
            "satchangegate download-optimus --dev-fixture"
        )
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"images/{group_index}.tar",
        repo_type="dataset",
    )
    dest.write_bytes(Path(src).read_bytes())
    return dest


def extract_tar_if_needed(tile_id: str, root: Path | None = None) -> Path:
    """Extract images/{group}.tar once to extracted/{group}/."""
    root = Path(root or default_optimus_root())
    download_optimus_index(root)
    group_index = resolve_group_index(tile_id, root)
    out_dir = root / "extracted" / str(group_index)
    if out_dir.is_dir() and any(out_dir.rglob(_tile_png_name(tile_id))):
        return out_dir
    tar_path = download_optimus_tar(tile_id, root, allow_large_download=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(out_dir)
    return out_dir


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _has_frame_files(tile_id: str, root: Path) -> bool:
    try:
        return len(list_timestamps_for_tile(tile_id, root)) >= 2
    except (FileNotFoundError, ValueError):
        return False


def list_timestamps_for_tile(tile_id: str, root: Path | None = None) -> list[str]:
    """Return sorted YYYY-MM folders containing tile_id.png on disk."""
    root = Path(root or default_optimus_root())
    name = _tile_png_name(tile_id)
    stamps: list[str] = []
    extracted_root = root / "extracted"
    if extracted_root.is_dir():
        for png in extracted_root.rglob(name):
            month_dir = png.parent.parent
            if month_dir.is_dir() and month_dir.name[:4].isdigit():
                stamps.append(month_dir.name)
    return sorted(set(stamps))


def load_optimus_frame(
    tile_id: str, timestamp: str, root: Path | None = None
) -> dict[str, np.ndarray]:
    """Load one OPTIMUS RGB PNG as pseudo Sentinel-2 bands."""
    root = Path(root or default_optimus_root())
    name = _tile_png_name(tile_id)
    matches = list((root / "extracted").rglob(name)) if (root / "extracted").is_dir() else []
    png = next((p for p in matches if p.parent.parent.name == timestamp), None)
    if png is None or not png.is_file():
        raise FileNotFoundError(
            f"Frame not found for {tile_id} @ {timestamp}. "
            "Run: satchangegate download-optimus --dev-fixture"
        )
    return load_bands(png, DEFAULT_BANDS)


def get_bitemporal_frames(
    tile_id: str,
    root: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, str]:
    """First and last available timestamps for a series."""
    root = Path(root or default_optimus_root())
    stamps = list_timestamps_for_tile(tile_id, root)
    if len(stamps) < 2:
        raise FileNotFoundError(
            f"Tile {tile_id} has fewer than 2 extracted timestamps under {root}. "
            "Earlier versions silently fabricated OPTIMUS frames from OSCD previews "
            "here, so an 'OPTIMUS evaluation' could be an evaluation on duplicated "
            "OSCD imagery with nothing in the output to say so."
        )
    if len(stamps) < 2:
        raise ValueError(f"Need >=2 timestamps for {tile_id}, got {stamps}")
    t1, t2 = stamps[0], stamps[-1]
    return load_optimus_frame(tile_id, t1, root), load_optimus_frame(tile_id, t2, root), t1, t2


def verify_layout(root: Path | None = None) -> tuple[bool, str]:
    root = Path(root or default_optimus_root())
    try:
        labels = load_eval_labels(root)
    except FileNotFoundError as e:
        return False, str(e)
    n0 = sum(1 for v in labels.values() if v == 0)
    n1 = sum(1 for v in labels.values() if v == 1)
    has_index = (root / "index.json").is_file()
    extracted = list((root / "extracted").glob("*")) if (root / "extracted").is_dir() else []
    dev = (root / "extracted" / ".dev_fixture").is_dir()
    parts = [
        f"Eval labels: {len(labels)} series ({n0} no-change, {n1} change)",
        f"index.json: {'yes' if has_index else 'no'}",
        f"extracted groups: {len(extracted)}",
    ]
    if dev:
        parts.append("dev fixture: yes")
    return True, "; ".join(parts)
