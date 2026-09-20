"""Staging uploaded imagery, and running the funnel over it.

Uploads are staged under the run root, never under the dataset or the reports
directory, and every file is written with a name this module chose. A name the
client supplied is data, not a path: it is recorded for display and never used
to decide where anything lands.

Batch upload takes an explicit manifest. Sorting filenames and pairing
neighbours would be convenient and would silently mis-pair the moment someone
uploads `site_a_2019.tif, site_a_2021.tif, site_b_2021.tif` -- and a mis-paired
before/after is not a detectable error downstream: it produces a confident
answer about two places.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satchangegate.webui.ingest import Declaration, IngestRefused, Limits, read_pair

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class StagedPair:
    """One before/after pair as the client described it."""

    key: str
    t1_name: str
    t2_name: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "t1": self.t1_name, "t2": self.t2_name, "label": self.label}


@dataclass
class Manifest:
    """What the client says its files are. Verified against them, never inferred."""

    declaration: Declaration
    pairs: list[StagedPair] = field(default_factory=list)

    @classmethod
    def parse(cls, blob: str | dict[str, Any], filenames: set[str]) -> Manifest:
        data = json.loads(blob) if isinstance(blob, str) else dict(blob)
        decl_raw = dict(data.get("declaration") or {})
        band_map = decl_raw.get("band_map") or {}
        if not isinstance(band_map, dict):
            raise IngestRefused("declaration.band_map must be a mapping of band name to index.")
        decl_raw["band_map"] = {str(k): int(v) for k, v in band_map.items()}
        try:
            declaration = Declaration(**decl_raw)
        except TypeError as exc:
            raise IngestRefused(f"Bad declaration: {exc}") from None

        raw_pairs = data.get("pairs")
        if not raw_pairs:
            raise IngestRefused(
                "No pairs declared. Which file is the before and which the after is "
                "not recoverable from filenames, and guessing wrong produces a "
                "confident answer about the wrong comparison."
            )
        pairs: list[StagedPair] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_pairs):
            t1, t2 = str(item.get("t1", "")), str(item.get("t2", ""))
            for name in (t1, t2):
                if name not in filenames:
                    raise IngestRefused(f"The manifest names {name!r}, which was not uploaded.")
            key = str(item.get("key") or f"pair_{index:03d}")
            if not _SAFE_NAME.match(key):
                raise IngestRefused(f"Pair key {key!r} is not a usable identifier.")
            if key in seen:
                raise IngestRefused(f"Duplicate pair key {key!r}.")
            seen.add(key)
            pairs.append(StagedPair(key=key, t1_name=t1, t2_name=t2, label=item.get("label")))
        return cls(declaration=declaration, pairs=pairs)


class UploadStore:
    """Staged uploads, under the run root and addressed by opaque id."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)

    def new(self) -> str:
        upload_id = uuid.uuid4().hex
        (self.root / upload_id).mkdir(parents=True)
        return upload_id

    def directory(self, upload_id: str) -> Path:
        if not _UPLOAD_ID.match(upload_id):
            raise IngestRefused("Not an upload id.")
        path = self.root / upload_id
        if not path.is_dir():
            raise IngestRefused(f"No staged upload {upload_id!r}.")
        return path

    def store(self, upload_id: str, client_name: str, data: bytes, limits: Limits) -> str:
        """Write one file under a name we choose, and record the client's.

        The returned name is what the manifest refers to. A client-supplied path
        never reaches the filesystem, so there is nothing for a traversal to
        traverse.
        """
        directory = self.directory(upload_id)
        if len(data) > limits.max_file_bytes:
            raise IngestRefused(
                f"{client_name} is {len(data) / 1e6:.1f} MB, over the "
                f"{limits.max_file_bytes / 1e6:.0f} MB limit."
            )
        existing = list(directory.glob("file_*"))
        if len(existing) >= limits.max_files:
            raise IngestRefused(f"At most {limits.max_files} files per upload.")
        total = sum(p.stat().st_size for p in existing) + len(data)
        if total > limits.max_total_bytes:
            raise IngestRefused(
                f"This upload totals {total / 1e6:.0f} MB, over the "
                f"{limits.max_total_bytes / 1e6:.0f} MB limit."
            )
        suffix = Path(client_name).suffix.lower()
        if suffix not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            raise IngestRefused(
                f"{client_name}: only GeoTIFF, PNG and JPEG are read. Formats that "
                f"can reference other files or URLs are refused outright."
            )
        stored = f"file_{len(existing):03d}{suffix}"
        (directory / stored).write_bytes(data)
        names = self._names(directory)
        names[stored] = client_name
        (directory / "names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
        return stored

    def _names(self, directory: Path) -> dict[str, str]:
        path = directory / "names.json"
        if not path.is_file():
            return {}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:  # pragma: no cover - written by us moments earlier
            return {}

    def filenames(self, upload_id: str) -> set[str]:
        return {p.name for p in self.directory(upload_id).glob("file_*")}

    def client_names(self, upload_id: str) -> dict[str, str]:
        return self._names(self.directory(upload_id))

    def path(self, upload_id: str, stored_name: str) -> Path:
        if not _SAFE_NAME.match(stored_name) or not stored_name.startswith("file_"):
            raise IngestRefused(f"{stored_name!r} is not a staged file.")
        path = self.directory(upload_id) / stored_name
        if not path.is_file():
            raise IngestRefused(f"{stored_name!r} is not in this upload.")
        return path

    def discard(self, upload_id: str) -> None:
        shutil.rmtree(self.directory(upload_id), ignore_errors=True)


def inspect(store: UploadStore, upload_id: str, manifest: Manifest) -> dict[str, Any]:
    """Verify every declared pair without running anything.

    A preview before expensive work, and before any paid work: the point is that
    a mis-declared band map or a mis-paired before/after is caught while it is
    still cheap to fix.
    """
    limits = Limits()
    results: list[dict[str, Any]] = []
    for pair in manifest.pairs:
        entry: dict[str, Any] = {**pair.to_dict()}
        try:
            verified = read_pair(
                store.path(upload_id, pair.t1_name),
                store.path(upload_id, pair.t2_name),
                manifest.declaration,
                limits=limits,
            )
            entry["ok"] = True
            entry.update(verified.to_dict())
        except IngestRefused as exc:
            entry["ok"] = False
            entry["refused"] = str(exc)
        results.append(entry)
    usable = [r for r in results if r["ok"]]
    return {
        "upload_id": upload_id,
        "n_pairs": len(results),
        "n_usable": len(usable),
        "lanes": {
            lane: sum(1 for r in usable if r["capability"]["lane"] == lane)
            for lane in ("full", "rgb_only")
        },
        "pairs": results,
        "client_names": store.client_names(upload_id),
    }
