"""Checksum-verified dataset acquisition.

The OSCD archive is mirrored on the Hugging Face Hub, which means the full
13-band multispectral release can be fetched non-interactively and verified.
The previous flow required a manual IEEE DataPort registration, three separate
zip downloads, and a hand-merge of label directories — so no clone of this repo
could reproduce any published number.

Only ``huggingface_hub`` is required. TorchGeo exposes the same archive, but
depends on torch (~2 GB) which this project has no use for.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

OSCD_REPO = "hkristen/oscd"

# The official split is defined by which label archive a city appears in.
# It is recovered from the archives at download time rather than hardcoded;
# these constants are only a fallback for an already-extracted tree.
OSCD_TRAIN_CITIES = (
    "abudhabi",
    "aguasclaras",
    "beihai",
    "beirut",
    "bercy",
    "bordeaux",
    "cupertino",
    "hongkong",
    "mumbai",
    "nantes",
    "paris",
    "pisa",
    "rennes",
    "saclay_e",
)
OSCD_TEST_CITIES = (
    "brasilia",
    "chongqing",
    "dubai",
    "lasvegas",
    "milano",
    "montpellier",
    "norcia",
    "rio",
    "saclay_w",
    "valencia",
)

# SHA256 of each archive, cross-checked against TorchGeo's published constants.
OSCD_FILES: dict[str, str] = {
    "Onera Satellite Change Detection dataset - Images.zip": (
        "940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656"
    ),
    "Onera Satellite Change Detection dataset - Train Labels.zip": (
        "89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5"
    ),
    "Onera Satellite Change Detection dataset - Test Labels.zip": (
        "2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d"
    ),
}

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing entries that escape ``dest``.

    ``ZipFile.extractall`` does not protect against absolute paths or ``..``
    traversal in member names.
    """
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Unsafe path in {archive.name}: {member.filename!r}")
        zf.extractall(dest)


def _fetch(filename: str, zips_dir: Path, *, verify: bool) -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        OSCD_REPO,
        filename,
        repo_type="dataset",
        local_dir=str(zips_dir),
    )
    path = Path(local)
    if verify:
        expected = OSCD_FILES[filename]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {filename}\n  expected {expected}\n  actual   {actual}"
            )
    return path


def _cities_in_archive(archive: Path) -> set[str]:
    """City directory names present in a label archive."""
    out: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            parts = [p for p in name.split("/") if p]
            if len(parts) >= 2 and "." not in parts[1]:
                out.add(parts[1])
    return out


def _normalise_tree(staging: Path, root: Path) -> int:
    """Merge the three extracted archives into one ``root/<city>/`` tree.

    The archives unpack to sibling top-level directories ("... - Images",
    "... - Train Labels", "... - Test Labels"), each containing per-city
    folders. Downstream code wants a single tree keyed by city.
    """
    cities: set[str] = set()
    for top in sorted(staging.iterdir()):
        if not top.is_dir():
            continue
        # Some archives nest one extra level with the same name.
        bases = [top]
        inner = top / top.name
        if inner.is_dir():
            bases = [inner]
        for base in bases:
            for city_dir in sorted(base.iterdir()):
                if not city_dir.is_dir() or city_dir.name.startswith("."):
                    continue
                dest = root / city_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for item in city_dir.iterdir():
                    target = dest / item.name
                    if target.exists():
                        continue
                    shutil.move(str(item), str(target))
                cities.add(city_dir.name)
    return len(cities)


def download_oscd(
    root: Path | None = None,
    *,
    force: bool = False,
    verify: bool = True,
    keep_archives: bool = False,
) -> Path:
    """Download, verify, and unpack the 13-band OSCD dataset.

    Returns the dataset root containing one directory per city.
    """
    root = Path(root or Path("data/raw/oscd"))
    root.mkdir(parents=True, exist_ok=True)

    marker = root / ".oscd_complete"
    if marker.is_file() and not force:
        return root

    zips_dir = root / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    staging = root / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    splits: dict[str, list[str]] = {}
    for filename in OSCD_FILES:
        archive = _fetch(filename, zips_dir, verify=verify)
        _safe_extract(archive, staging)
        if "Train Labels" in filename:
            splits["train"] = sorted(_cities_in_archive(archive))
        elif "Test Labels" in filename:
            splits["test"] = sorted(_cities_in_archive(archive))

    # Persist the split derived from the archives so downstream code never has
    # to guess. The previously hardcoded lists disagreed with the real dataset:
    # they placed four test cities in train and named four cities that do not exist.
    if splits.get("train") and splits.get("test"):
        overlap = set(splits["train"]) & set(splits["test"])
        if overlap:
            raise RuntimeError(f"OSCD split archives overlap: {sorted(overlap)}")
        (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    n_cities = _normalise_tree(staging, root)
    shutil.rmtree(staging, ignore_errors=True)
    if not keep_archives:
        shutil.rmtree(zips_dir, ignore_errors=True)

    marker.write_text(f"cities={n_cities}\n", encoding="utf-8")
    return root
