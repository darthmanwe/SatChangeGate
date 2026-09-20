"""The local web application.

Loopback binding is not authorization. This process holds a live
``ANTHROPIC_API_KEY`` and can spend money, so the browser is treated as an
untrusted client even though it is on the same machine: ``Host`` is validated,
state-changing routes need a per-launch token, and ``GET`` never changes
anything. A page on another origin can make your browser issue requests to
127.0.0.1; it cannot read the token embedded in a page it is not allowed to read.

Nothing here serves a path the caller chose. Scenes, artifacts and evidence are
addressed by validated keys that a registry resolves, because the alternative is
a server that can be asked for ``.env``.

``load_env_file`` is injectable for one specific reason: ``tests/conftest.py``
unsets ``ANTHROPIC_API_KEY`` for the whole session so no test can spend by
accident, and an app factory that re-read a real ``.env`` at startup would walk
straight through that protection.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from satchangegate.webui.artifacts import ARTIFACTS as ARTIFACT_SPECS
from satchangegate.webui.artifacts import ArtifactMissing, ArtifactStore
from satchangegate.webui.catalog import build_catalogue

STATIC_DIR = Path(__file__).parent / "static"

#: Hosts a browser may claim when talking to us. Anything else is a rebinding
#: attempt or a misconfiguration, and either way is refused.
ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})


@dataclass
class AppConfig:
    """Everything the app needs that is not a request."""

    oscd_root: Path | None = None
    reports: Path | None = None
    sample: Path | None = None
    models: Path | None = None
    allow_spend: bool = False
    spend_cap_usd: float = 1.0
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    load_env: bool = True


def _require_token(config: AppConfig) -> Callable[..., None]:
    def dependency(x_scg_token: str | None = Header(default=None)) -> None:
        if not x_scg_token or not secrets.compare_digest(x_scg_token, config.token):
            raise HTTPException(
                status_code=403,
                detail=(
                    "This route changes state and needs the session token that "
                    "the page it is served from carries."
                ),
            )

    return dependency


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    if config.load_env:
        from satchangegate.config import load_env_file

        load_env_file()

    store = ArtifactStore(reports=config.reports, sample=config.sample, models=config.models)
    app = FastAPI(
        title="SatChangeGate",
        description="Local review surface for a cost-controlled change-detection funnel.",
        docs_url=None,
        redoc_url=None,
    )
    app.state.config = config
    app.state.store = store
    token_guard = Depends(_require_token(config))

    @app.middleware("http")
    async def _guard_origin(request: Request, call_next: Callable[..., Any]) -> Response:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host and host not in ALLOWED_HOSTS:
            return JSONResponse(
                {"detail": f"Refusing a request claiming Host {host!r}."}, status_code=421
            )
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and not any(h in origin for h in ALLOWED_HOSTS):
                return JSONResponse(
                    {"detail": "Cross-origin state change refused."}, status_code=403
                )
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------ pages

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = STATIC_DIR / "index.html"
        if not page.is_file():  # pragma: no cover - only in a broken install
            return HTMLResponse("<h1>SatChangeGate</h1><p>Static assets missing.</p>", 500)
        html = page.read_text(encoding="utf-8").replace("{{SESSION_TOKEN}}", config.token)
        return HTMLResponse(html)

    # ------------------------------------------------------------------- meta

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "satchangegate-webui"}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        """What this installation can actually do, so the UI never offers what it cannot."""
        import importlib.util
        import os

        from satchangegate.config import get_settings
        from satchangegate.pipeline import provenance
        from satchangegate.services import SERVICES

        scenes = _catalogue(config)
        has_extra = importlib.util.find_spec("sklearn") is not None
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        present = {
            "dataset": bool(scenes),
            "baseline-extra": has_extra,
            "api-key": has_key,
            "ledger": store.status("e2e_ledger")["available"],
        }
        operations = []
        for name, spec in SERVICES.items():
            missing = [r for r in spec.requires if not present.get(r, True)]
            blocked = list(missing)
            if spec.spends_money and not config.allow_spend:
                blocked.append("spend-disabled")
            operations.append(
                {
                    "name": name,
                    "summary": spec.summary,
                    "speed": spec.speed,
                    "spends_money": spec.spends_money,
                    "needs_network": spec.needs_network,
                    "requires": list(spec.requires),
                    "available": not blocked,
                    "blocked_by": blocked,
                    "schema": _schema(spec.request_type),
                }
            )
        return {
            "provenance": provenance(get_settings()),
            "present": present,
            "spend": {
                "allowed": config.allow_spend,
                "cap_usd": config.spend_cap_usd,
                "note": (
                    "A key being present is never permission to spend. Paid routes "
                    "arrive with the reservation ledger."
                ),
            },
            "n_scenes": len(scenes),
            "operations": operations,
        }

    @app.get("/api/operations")
    def operations() -> dict[str, Any]:
        from satchangegate.services import SERVICES

        return {
            "operations": [
                {
                    "name": name,
                    "summary": spec.summary,
                    "speed": spec.speed,
                    "spends_money": spec.spends_money,
                    "command": spec.request_type.command_name,
                    "schema": _schema(spec.request_type),
                }
                for name, spec in SERVICES.items()
            ]
        }

    @app.post("/api/commands/preview", dependencies=[token_guard])
    def preview_command(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate a request and render the command, without running anything.

        The console uses this so the command on screen is generated by the same
        validation the run would use, rather than assembled in JavaScript.
        """
        from satchangegate.services import SERVICES, ServiceUnavailable

        name = str(payload.get("operation", ""))
        spec = SERVICES.get(name)
        if spec is None:
            raise HTTPException(404, f"Unknown operation {name!r}")
        try:
            request = spec.request_type(**(payload.get("params") or {}))
        except (ValueError, ServiceUnavailable) as exc:
            raise HTTPException(422, str(exc)) from None
        return {
            "operation": name,
            "command": request.to_command(),
            "command_with_paths": request.to_command(include_locations=True),
            "normalised": request.model_dump(mode="json"),
        }

    # ----------------------------------------------------------------- scenes

    @app.get("/api/scenes")
    def scenes() -> dict[str, Any]:
        items = _catalogue(config)
        return {
            "scenes": [s.to_dict() for s in items],
            "georeferencing": (
                "Bounds come from each city's unrectified EPSG:4326 raster, with "
                "the AOI polygon as a fallback; bbox_source says which. "
                "mercator_skew_px is the measured cost of drawing an "
                "equirectangular grid on a Web Mercator basemap."
            ),
        }

    @app.get("/api/scenes/{city}")
    def scene(city: str) -> dict[str, Any]:
        match = next((s for s in _catalogue(config) if s.city == city), None)
        if match is None:
            raise HTTPException(404, f"No scene named {city!r}")
        return match.to_dict()

    @app.get("/api/scenes/{city}/preview/{which}")
    def scene_preview(city: str, which: str) -> Response:
        if which not in ("t1", "t2"):
            raise HTTPException(404, "Preview must be t1 or t2")
        match = next((s for s in _catalogue(config) if s.city == city), None)
        if match is None:
            raise HTTPException(404, f"No scene named {city!r}")
        from satchangegate.data.oscd import default_oscd_root

        root = Path(config.oscd_root or default_oscd_root())
        name = "img1.png" if which == "t1" else "img2.png"
        path = root / match.city / "pair" / name
        if not path.is_file():
            raise HTTPException(404, "No committed preview for this scene")
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/api/scenes/{city}/layers")
    def scene_layers(city: str) -> dict[str, Any]:
        """The layer menu for a scene, and the configuration that would render it."""
        from satchangegate.config import get_settings
        from satchangegate.webui.render import SIGNED_FULL_SCALE, describe_layers

        if not any(s.city == city for s in _catalogue(config)):
            raise HTTPException(404, f"No scene named {city!r}")
        return {
            "city": city,
            "layers": describe_layers(),
            "fingerprint": provenance_fingerprint(),
            "signed_full_scale": SIGNED_FULL_SCALE,
            "rendering_notes": [
                "Signed index deltas use a fixed symmetric range, not a per-scene "
                "stretch, so two scenes are comparable and a quiet one looks quiet.",
                "Pixels outside the valid mask are hatched rather than filled: "
                "unobserved is not zero.",
                "Quality masks name their timestep. Cloud at t1, cloud at t2 and "
                "their union are three different pictures.",
            ],
            "settings": get_settings().gate.model_dump(),
        }

    @app.get("/api/scenes/{city}/layers/{name}")
    def scene_layer(city: str, name: str) -> Response:
        """One rendered layer.

        Rendered from the configuration this server is running, and stamped with
        its fingerprint, so a picture cannot be mistaken for one produced under
        different thresholds.
        """
        from satchangegate.config import get_settings
        from satchangegate.data.oscd import default_oscd_root
        from satchangegate.webui.render import compute_layers, encode_layer

        if not any(s.city == city for s in _catalogue(config)):
            raise HTTPException(404, f"No scene named {city!r}")
        root = Path(config.oscd_root or default_oscd_root())
        try:
            layers = compute_layers(root, city, get_settings())
            blob, legend = encode_layer(layers, name)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None
        headers = {
            "X-SCG-Layer-Scale": str(legend.get("scale", "")),
            "X-SCG-Config": str(legend.get("fingerprint", "")),
        }
        return Response(blob, media_type="image/png", headers=headers)

    # -------------------------------------------------------------- artifacts

    @app.get("/api/artifacts")
    def artifacts() -> dict[str, Any]:
        return {"artifacts": store.catalogue()}

    @app.get("/api/artifacts/{key}")
    def artifact(key: str) -> dict[str, Any]:
        from satchangegate.webui.artifacts import ARTIFACTS

        spec = ARTIFACTS.get(key)
        if spec is None:
            raise HTTPException(404, f"Unknown artifact {key!r}")
        try:
            if spec.kind == "json":
                return store.read_json(key)
            if spec.kind in ("markdown", "csv"):
                return store.read_text(key)
            if spec.kind == "jsonl":
                return {
                    "source": store.resolve(key).source,
                    "produced_by": spec.produced_by,
                    "rows": store.read_ledger(key),
                }
        except ArtifactMissing as exc:
            raise HTTPException(
                404, {"detail": str(exc), "produced_by": spec.produced_by, "available": False}
            ) from None
        raise HTTPException(415, f"{key} is binary; use /api/artifacts/{key}/raw")

    @app.get("/api/artifacts/{key}/raw")
    def artifact_raw(key: str) -> Response:
        try:
            blob, media = store.read_bytes(key)
        except ArtifactMissing as exc:
            raise HTTPException(404, str(exc)) from None
        return Response(blob, media_type=media)

    # ------------------------------------------------------------- playground

    @app.get("/api/playground/thresholds")
    def playground_thresholds() -> dict[str, Any]:
        """The shipped thresholds, split by whether a slider can honour them live."""
        from satchangegate.config import GateThresholds, get_settings
        from satchangegate.webui.features import (
            DECISION_ONLY,
            DERIVED_FROM_MASK,
            FEATURE_CHANGING,
        )

        shipped = get_settings().gate
        schema = GateThresholds.model_json_schema().get("properties", {})
        fields = []
        for name, value in shipped.model_dump().items():
            kind = (
                "decision"
                if name in DECISION_ONLY
                else "mask"
                if name in FEATURE_CHANGING
                else "other"
            )
            spec = schema.get(name, {})
            fields.append(
                {
                    "name": name,
                    "value": value,
                    "kind": kind,
                    "type": spec.get("type", "number"),
                    "description": spec.get("description"),
                    "minimum": spec.get("minimum"),
                    "maximum": spec.get("maximum"),
                }
            )
        return {
            "thresholds": fields,
            "decision_only": sorted(DECISION_ONLY),
            "feature_changing": sorted(FEATURE_CHANGING),
            "invalidates": list(DERIVED_FROM_MASK),
            "note": (
                "Decision thresholds re-score the cached matrix instantly, because "
                "they are read only inside decide(). The eight mask thresholds "
                "change the change mask itself, and so change six of the eighteen "
                "cached features: re-scoring without recomputing would look live "
                "and be wrong."
            ),
        }

    @app.post("/api/playground/score", dependencies=[token_guard])
    def playground_score(payload: dict[str, Any]) -> dict[str, Any]:
        """Re-score a cached feature matrix under threshold overrides.

        Side-effect free: it writes nothing and spends nothing. It is a POST
        because it carries a body, and it carries the session token because every
        POST here does.
        """
        from satchangegate.config import get_settings
        from satchangegate.webui.features import classify_overrides, load_matrix, score

        split = str(payload.get("split", "train"))
        if split not in ("train", "test"):
            raise HTTPException(422, "split must be train or test")
        overrides = dict(payload.get("thresholds") or {})

        # Validate the request before touching the filesystem, so a bad threshold
        # name is a 422 whether or not a matrix happens to exist here.
        settings = get_settings()
        known = set(settings.gate.model_dump())
        unknown = sorted(k for k in overrides if k not in known)
        if unknown:
            raise HTTPException(422, f"Unknown thresholds: {', '.join(unknown)}")

        key = "features_train" if split == "train" else "features_test"
        try:
            resolved = store.resolve(key)
        except ArtifactMissing as exc:
            raise HTTPException(
                404,
                {
                    "detail": str(exc),
                    "produced_by": ARTIFACT_SPECS[key].produced_by,
                    "available": False,
                },
            ) from None

        matrix = load_matrix(str(resolved.path), split, resolved.source, resolved.spec.produced_by)

        decision_only, feature_changing = classify_overrides(overrides)
        applied = {k: v for k, v in overrides.items() if k in decision_only}
        thresholds = settings.gate.model_copy(update=applied)
        result = score(matrix, thresholds)

        return {
            "split": split,
            "population": {
                "n": len(matrix),
                "source": matrix.source,
                "produced_by": matrix.produced_by,
                "is_test": split == "test",
            },
            "applied": applied,
            "ignored_feature_changing": feature_changing,
            "stale": bool(feature_changing),
            "stale_note": (
                "These change the change mask, so the cached features no longer "
                "describe this configuration. They were not applied. Run "
                f"`satchangegate eval --split {split}` with them set in "
                "thresholds.yaml to see their real effect."
                if feature_changing
                else None
            ),
            "test_informed": (
                "Thresholds tuned against this population are test-informed, not "
                "independently validated. Fit on train and report on test."
                if split == "test"
                else None
            ),
            "metrics": result,
            "command": (
                f"satchangegate eval --split {split}"
                if not applied
                else f"# set these in thresholds.yaml, then: satchangegate eval --split {split}"
            ),
            "yaml": _threshold_yaml(applied) if applied else None,
        }

    # --------------------------------------------------------------- evidence

    @app.get("/api/evidence")
    def evidence() -> dict[str, Any]:
        tiles = store.evidence_tiles()
        rows: dict[str, dict[str, Any]] = {}
        try:
            rows = {r["tile_id"]: r for r in store.read_ledger()}
        except ArtifactMissing:
            rows = {}
        return {
            "n_packages": len(tiles),
            "n_ledger_rows": len(rows),
            "retained_verdict_fields": ["vlm_verdict", "vlm_change_type", "vlm_confidence"],
            "not_retained": ["artifact_risk", "visual_evidence", "requires_human_review"],
            "not_retained_note": (
                "The VLM schema validates these three fields and the run used them, "
                "but no artifact on disk kept them. For the recorded run they are "
                "unrecoverable; they are persisted for runs made from here on."
            ),
            "tiles": [
                {"tile_id": t, **{k: rows.get(t, {}).get(k) for k in _LEDGER_FIELDS}} for t in tiles
            ],
        }

    @app.get("/api/evidence/{tile_id}")
    def evidence_tile(tile_id: str) -> dict[str, Any]:
        import json

        try:
            blob, _ = store.evidence_file(tile_id, "metadata.json")
        except ArtifactMissing as exc:
            raise HTTPException(404, str(exc)) from None
        try:
            row = next((r for r in store.read_ledger() if r.get("tile_id") == tile_id), None)
        except ArtifactMissing:
            row = None
        return {
            "tile_id": tile_id,
            "metadata_shown_to_model": json.loads(blob),
            "ledger_row": row,
            "images": ["before_rgb.png", "after_rgb.png", "change_overlay.png"],
            "redaction_note": (
                "The metadata above is exactly what the model was shown. The "
                "unredacted gate features are withheld deliberately, so agreement "
                "between the gate and its verifier stays meaningful; they are not "
                "served here at all."
            ),
        }

    @app.get("/api/evidence/{tile_id}/{name}")
    def evidence_image(tile_id: str, name: str) -> Response:
        try:
            blob, media = store.evidence_file(tile_id, name)
        except ArtifactMissing as exc:
            raise HTTPException(404, str(exc)) from None
        return Response(blob, media_type=media)

    return app


_LEDGER_FIELDS = (
    "city",
    "label",
    "gate",
    "gate_reason",
    "gate_confidence",
    "vlm_called",
    "vlm_verdict",
    "vlm_change_type",
    "vlm_confidence",
    "cost_usd",
    "error",
)


def provenance_fingerprint() -> str:
    """The settings fingerprint stamped onto every rendered layer."""
    from satchangegate.config import get_settings
    from satchangegate.pipeline import provenance

    return str(provenance(get_settings())["config_sha256"])


def _schema(model: Any) -> dict[str, Any]:
    """JSON schema for a request model.

    Path-valued defaults are not JSON-serialisable, so pydantic drops them and
    warns. Those fields are locations rather than parameters and the console does
    not render them, which makes the warning noise rather than news.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(model.model_json_schema())


def _threshold_yaml(applied: dict[str, Any]) -> str:
    """A paste-able gate block, in the same shape `tune` writes."""
    lines = [
        "# Hand-set in the threshold playground, not fitted.",
        "# `satchangegate tune` writes this same block from a sweep; these values",
        "# were chosen by eye and carry no in-sample optimality claim.",
        "gate:",
    ]
    lines += [f"  {k}: {applied[k]}" for k in sorted(applied)]
    return "\n".join(lines) + "\n"


def _catalogue(config: AppConfig) -> list[Any]:
    from satchangegate.data.oscd import default_oscd_root

    root = Path(config.oscd_root or default_oscd_root())
    if not root.is_dir():
        return []
    return build_catalogue(root)
