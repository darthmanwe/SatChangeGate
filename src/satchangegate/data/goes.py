"""GOES-R ABI weather/illumination context (coarse, not primary change imagery)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_GOES_ROOT = Path("data/raw/goes")
_GOES2GO_DEFAULT_TOML = """# GOES-2-go Defaults (minimal bootstrap for SatChangeGate)
["default"]
save_dir = "{save_dir}"
satellite = "noaa-goes16"
product = "ABI-L2-MCMIPC"
domain = "C"
download = true
return_as = "filelist"
overwrite = false
max_cpus = 1
s3_refresh = true
ignore_missing = false
verbose = false

["timerange"]
s3_refresh = false
ignore_missing = true

["latest"]
return_as = "xarray"

["nearesttime"]
within = "1h"
return_as = "filelist"
"""


@dataclass
class GoesContext:
    satellite: int
    timestamp: str
    product: str
    domain: str
    local_path: Path | None
    brightness_mean: float | None
    notes: str = ""


def default_goes_root() -> Path:
    return DEFAULT_GOES_ROOT


def _goes2go_config_path() -> Path:
    import os

    env = os.getenv("GOES2GO_CONFIG_PATH")
    if env:
        return Path(env)
    return Path.home() / ".config" / "goes2go" / "config.toml"


def _ensure_goes2go_config(save_dir: Path) -> None:
    """Create goes2go config without Unicode box-drawing prints (Windows cp1252 safe)."""
    path = _goes2go_config_path()
    save_dir.mkdir(parents=True, exist_ok=True)
    content = _GOES2GO_DEFAULT_TOML.format(save_dir=str(save_dir.resolve()).replace("\\", "/"))
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _import_goes(save_dir: Path):
    _ensure_goes2go_config(save_dir)
    from goes2go import GOES

    return GOES


def _brightness_from_netcdf(path: Path) -> float | None:
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None
    with Dataset(path) as nc:
        for var in ("CMI", "CMI_C02", "Rad"):
            if var in nc.variables:
                arr = np.asarray(nc.variables[var][:], dtype=np.float32)
                return float(np.nanmean(arr))
    return None


def _to_goes2go_time(when: datetime) -> datetime:
    """goes2go compares against tz-naive pandas timestamps."""
    if when.tzinfo is None:
        return when
    return when.astimezone(timezone.utc).replace(tzinfo=None)


def fetch_goes_abi_snapshot(
    when: datetime | str,
    *,
    satellite: int = 16,
    domain: str = "C",
    product: str = "ABI-L2-MCMIPC",
    out_dir: Path | None = None,
) -> GoesContext:
    """
    Download one GOES ABI CONUS snapshot nearest `when` using goes2go.

    Returns metadata for attachment to pipeline quality JSON.
    """
    out_dir = Path(out_dir or default_goes_root())
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        GOES = _import_goes(out_dir)
    except ImportError as e:
        raise ImportError("Install goes2go: pip install satchangegate[goes]") from e

    if isinstance(when, str):
        when = datetime.fromisoformat(when.replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    g = GOES(satellite=satellite, product=product, domain=domain)
    result = g.nearesttime(_to_goes2go_time(when), return_as="filelist")

    local_nc: Path | None = None
    timestamp = when.isoformat()
    if hasattr(result, "iloc") and len(result) > 0:
        raw_file = str(result.iloc[0]["file"])
        src = Path(raw_file)
        if not src.is_file():
            src = out_dir / raw_file
        if not src.is_file():
            matches = list(out_dir.rglob("*.nc"))
            src = matches[-1] if matches else src
        ts = str(result.iloc[0].get("start", timestamp))[:19].replace(":", "-")
        local_nc = out_dir / f"goes{satellite}_{domain}_{ts}.nc"
        timestamp = str(result.iloc[0].get("start", timestamp))
        if not local_nc.is_file() and src.is_file():
            local_nc.write_bytes(src.read_bytes())
    else:
        downloaded = sorted(out_dir.rglob("*.nc"), key=lambda p: p.stat().st_mtime)
        if downloaded:
            local_nc = downloaded[-1]
            timestamp = str(local_nc.stem)

    if local_nc is None or not local_nc.is_file():
        raise FileNotFoundError(f"No GOES NetCDF downloaded under {out_dir}")

    brightness = _brightness_from_netcdf(local_nc)

    ctx = GoesContext(
        satellite=satellite,
        timestamp=timestamp,
        product=product,
        domain=domain,
        local_path=local_nc,
        brightness_mean=brightness,
        notes="GOES ABI context only; not used for 10m change detection.",
    )
    ts_slug = str(timestamp)[:19].replace(":", "-")
    meta_path = out_dir / f"goes{satellite}_{domain}_{ts_slug}.json"
    meta_path.write_text(
        json.dumps(
            {
                "satellite": ctx.satellite,
                "timestamp": ctx.timestamp,
                "product": ctx.product,
                "domain": ctx.domain,
                "local_path": str(ctx.local_path),
                "brightness_mean": ctx.brightness_mean,
                "notes": ctx.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ctx


def verify_layout(root: Path | None = None) -> tuple[bool, str]:
    root = Path(root or default_goes_root())
    if not root.is_dir():
        return False, f"GOES root missing: {root}"
    files = list(root.glob("*.nc")) + list(root.glob("*.json"))
    if not files:
        return (
            False,
            f"No GOES files under {root}. Run: fetch_goes_abi_snapshot() (no CLI command; this track is not wired in)",
        )
    return True, f"Found {len(files)} GOES artifact(s) under {root}"
