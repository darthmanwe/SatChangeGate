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
    run_root: Path | None = None
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
    app.state.reservations = {}
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
            # An operation that *can* spend is not blocked outright: only the
            # requests that actually would are. The UI shows the flag so a form
            # can warn before submitting.
            if spec.spends_money and not config.allow_spend and not spec.spend_fields:
                blocked.append("spend-disabled")
            operations.append(
                {
                    "name": name,
                    "summary": spec.summary,
                    "speed": spec.speed,
                    "spends_money": spec.spends_money,
                    "spend_fields": list(spec.spend_fields),
                    "needs_network": spec.needs_network,
                    "requires": list(spec.requires),
                    "available": not blocked,
                    "blocked_by": blocked,
                    "schema": _schema(spec.request_type),
                }
            )
        return {
            "provenance": provenance(get_settings()),
            "scorer": _scorer_status(),
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

    # ---------------------------------------------------------------- uploads

    @app.get("/api/uploads/contract")
    def upload_contract() -> dict[str, Any]:
        """What an upload has to declare, and why each part is required."""
        from satchangegate.config import ALL_BANDS, GATE_BANDS
        from satchangegate.webui.ingest import ALLOWED_DRIVERS, Limits

        limits = Limits()
        return {
            "band_names": list(ALL_BANDS),
            "bands_the_gate_needs": list(GATE_BANDS),
            "allowed_drivers": sorted(ALLOWED_DRIVERS),
            "limits": {
                "max_file_mb": limits.max_file_bytes // (1024 * 1024),
                "max_total_mb": limits.max_total_bytes // (1024 * 1024),
                "max_megapixels": limits.max_pixels // 1_000_000,
                "max_bands": limits.max_bands,
                "max_files": limits.max_files,
            },
            "requirements": [
                {
                    "what": "Name every band",
                    "why": "Band 4 of an arbitrary raster is not NDVI's red channel "
                    "just because it is fourth.",
                },
                {
                    "what": "Declare the reflectance scale",
                    "why": "10000 is right for Sentinel-2 L1C and wrong for almost "
                    "everything else, and dtype does not reveal which you have. "
                    "Values above 1.0 are kept: bright targets legitimately exceed it.",
                },
                {
                    "what": "Georeferenced, or explicitly already-aligned",
                    "why": "Equal pixel dimensions do not mean equal ground. Two "
                    "same-sized rasters of different continents would otherwise "
                    "produce a confident answer about nothing.",
                },
                {
                    "what": "State the ground resolution",
                    "why": "Despeckling, component sizes, the 3x context window and "
                    "the 1.5 px registration tolerance are all in pixels. The same "
                    "thresholds mean something different at 30 m.",
                },
                {
                    "what": "Pair the files explicitly",
                    "why": "Which file is before and which is after is not "
                    "recoverable from filenames, and a mis-paired comparison is not "
                    "detectable downstream -- it just answers about the wrong thing.",
                },
            ],
        }

    @app.post("/api/uploads", dependencies=[token_guard])
    async def create_upload(request: Request) -> dict[str, Any]:
        """Stage files. Nothing is decoded here beyond what the limits need."""
        from satchangegate.webui.ingest import IngestRefused, Limits
        from satchangegate.webui.uploads import UploadStore

        store_ = UploadStore(_run_root(config))
        limits = Limits()
        form = await request.form()
        upload_id = store_.new()
        stored: list[dict[str, str]] = []
        try:
            for value in form.getlist("files"):
                # A multipart field can be a plain string; only real file parts
                # carry bytes, and a string masquerading as one is not an upload.
                if isinstance(value, str):
                    continue
                client_name = getattr(value, "filename", None)
                if not client_name:
                    continue
                data = await value.read()
                name = store_.store(upload_id, client_name, data, limits)
                stored.append({"stored": name, "client_name": client_name})
        except IngestRefused as exc:
            store_.discard(upload_id)
            raise HTTPException(413, str(exc)) from None
        if not stored:
            store_.discard(upload_id)
            raise HTTPException(422, "No files were uploaded.")
        return {
            "upload_id": upload_id,
            "files": stored,
            "note": (
                "Stored under names this server chose. The name you sent is kept "
                "for display only: a client-supplied path never reaches the "
                "filesystem."
            ),
        }

    @app.post("/api/uploads/{upload_id}/inspect", dependencies=[token_guard])
    def inspect_upload(upload_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Verify every declared pair without running or spending anything."""
        from satchangegate.webui.ingest import IngestRefused
        from satchangegate.webui.uploads import Manifest, UploadStore, inspect

        store_ = UploadStore(_run_root(config))
        try:
            manifest = Manifest.parse(payload, store_.filenames(upload_id))
            return inspect(store_, upload_id, manifest)
        except IngestRefused as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/uploads/{upload_id}/run", dependencies=[token_guard])
    def run_upload(upload_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Queue one verified pair. Same service, same command, as the CLI."""
        from satchangegate.services import SERVICES, RunImagesRequest
        from satchangegate.webui.ingest import IngestRefused
        from satchangegate.webui.jobs import QueueFull
        from satchangegate.webui.uploads import Manifest, UploadStore

        store_ = UploadStore(_run_root(config))
        try:
            manifest = Manifest.parse(payload, store_.filenames(upload_id))
        except IngestRefused as exc:
            raise HTTPException(422, str(exc)) from None

        key = str(payload.get("pair") or (manifest.pairs[0].key if manifest.pairs else ""))
        pair = next((p for p in manifest.pairs if p.key == key), None)
        if pair is None:
            raise HTTPException(404, f"No pair {key!r} in this manifest.")

        spec = SERVICES["run-images"]
        decl = manifest.declaration
        try:
            request = RunImagesRequest(
                t1=store_.path(upload_id, pair.t1_name),
                t2=store_.path(upload_id, pair.t2_name),
                bands=",".join(f"{k}={v}" for k, v in sorted(decl.band_map.items())),
                reflectance_scale=decl.reflectance_scale,
                reflectance_offset=decl.reflectance_offset,
                date_t1=decl.date_t1,
                date_t2=decl.date_t2,
                alignment=decl.alignment,
                resolution_m=decl.target_resolution_m,
                name=pair.key,
                vlm=bool(payload.get("vlm")),
            )
        except (ValueError, IngestRefused) as exc:
            raise HTTPException(422, str(exc)) from None

        if spec.request_spends(request) and not config.allow_spend:
            raise HTTPException(
                403,
                "Verification would make a paid API call and this server was "
                "started without --allow-spend.",
            )

        runner = _runner(app, config)
        params = request.model_dump(mode="json")

        def prepare(run_id: str, queued: dict[str, Any]) -> None:
            queued["out"] = _isolated_out(runner, run_id, spec)

        try:
            run_id = runner.submit(
                "run-images",
                params,
                command=request.to_command(),
                display_name=f"upload {pair.key}",
                fingerprint=provenance_fingerprint(),
                prepare=prepare,
            )
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from None
        return {"run_id": run_id, "command": request.to_command(), "pair": pair.to_dict()}

    @app.delete("/api/uploads/{upload_id}", dependencies=[token_guard])
    def discard_upload(upload_id: str) -> dict[str, Any]:
        from satchangegate.webui.ingest import IngestRefused
        from satchangegate.webui.uploads import UploadStore

        try:
            UploadStore(_run_root(config)).discard(upload_id)
        except IngestRefused as exc:
            raise HTTPException(404, str(exc)) from None
        return {"discarded": upload_id}

    @app.get("/api/batches")
    def batches() -> dict[str, Any]:
        """Batch manifests on disk, and what they mean for spending again.

        A manifest exists because a batch was submitted, and a submitted batch
        has already been paid for. Reading one before doing anything is the
        difference between rejoining work and buying it twice.
        """
        import json as _json

        reports = Path(config.reports or "data/reports")
        found = []
        for path in sorted(reports.glob("_e2e_*_batch.json")):
            entry: dict[str, Any] = {"path": str(path), "readable": True}
            try:
                blob = _json.loads(path.read_text(encoding="utf-8"))
                entry.update(
                    {
                        "batch_id": blob.get("batch_id"),
                        "model": blob.get("model"),
                        "split": blob.get("split"),
                        "n_tiles": len(blob.get("tile_ids") or []),
                    }
                )
            except (ValueError, OSError) as exc:
                entry.update({"readable": False, "error": str(exc)})
            found.append(entry)
        unreadable = [e for e in found if not e["readable"]]
        return {
            "manifests": found,
            "note": (
                "A manifest records a batch that was already paid for at submission. "
                "A rerun reattaches to it rather than resubmitting; never submit one "
                "by hand."
            ),
            "warning": (
                "One or more manifests cannot be read. The funnel now refuses to "
                "submit over an unreadable manifest rather than assuming there is "
                "no batch, because assuming that buys the same work twice. Inspect "
                "or collect them before running again."
                if unreadable
                else None
            ),
        }

    # ------------------------------------------------------------------ spend

    @app.get("/api/spend")
    def spend_summary() -> dict[str, Any]:
        return {
            **_ledger(app, config).summary(),
            "enabled": config.allow_spend,
            "policy": (
                "Spend is reserved at a conservative upper bound before anything "
                "is dispatched, and settled afterwards from the tokens the API "
                "reported. A request whose outcome is unknown keeps its hold: it "
                "may have been served and billed."
            ),
        }

    @app.post("/api/spend/quote", dependencies=[token_guard])
    def spend_quote(payload: dict[str, Any]) -> dict[str, Any]:
        """Price a request before anyone confirms it.

        The quote is bound to this exact request. Changing the images, the model,
        the output cap or the call count invalidates it rather than carrying an
        old approval onto new work.
        """
        from satchangegate.services import SERVICES
        from satchangegate.webui.spend import SpendRefused

        if not config.allow_spend:
            raise HTTPException(
                403, "This server was started without --allow-spend; nothing here can be bought."
            )
        name = str(payload.get("operation", ""))
        spec = SERVICES.get(name)
        if spec is None:
            raise HTTPException(404, f"Unknown operation {name!r}")
        try:
            request = spec.request_type(**(payload.get("params") or {}))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if not spec.request_spends(request):
            return {"quote": None, "note": "This request spends nothing; no quote is needed."}

        from satchangegate.vlm.client import (
            DEFAULT_MAX_RETRIES,
            DEFAULT_MAX_TOKENS,
            DEFAULT_VLM_MODEL,
            resolve_model,
        )

        model = resolve_model(
            getattr(request, "model", None) or getattr(request, "vlm_model", None),
            "ANTHROPIC_VLM_MODEL",
            DEFAULT_VLM_MODEL,
        )
        n_calls = int(payload.get("n_calls") or getattr(request, "max_vlm_calls", None) or 1)
        try:
            quote = _ledger(app, config).quote(
                request=request.model_dump(mode="json"),
                model=model,
                n_calls=n_calls,
                batch=bool(getattr(request, "batch", False)),
                max_output_tokens=DEFAULT_MAX_TOKENS,
                attempts=DEFAULT_MAX_RETRIES + 1,
            )
        except SpendRefused as exc:
            raise HTTPException(422, str(exc)) from None
        return {"quote": quote.to_dict(), "ledger": _ledger(app, config).summary()}

    @app.post("/api/spend/{reservation_id}/release", dependencies=[token_guard])
    def release_reservation(reservation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Give back a hold, after someone has checked what actually happened.

        Deliberately manual. An automatic release of an uncertain outcome would
        let the next request spend money that may already be owed.
        """
        from satchangegate.webui.spend import SpendRefused

        note = str(payload.get("note") or "reconciled by hand")
        try:
            _ledger(app, config).release(reservation_id, note)
        except SpendRefused as exc:
            raise HTTPException(404, str(exc)) from None
        return {"released": reservation_id, "ledger": _ledger(app, config).summary()}

    # ------------------------------------------------------------------- runs

    @app.get("/api/worker")
    def worker_status() -> dict[str, Any]:
        runner = _runner(app, config)
        return {
            **runner.status(),
            "note": (
                "One worker by design. Two would contend for the same scene cache "
                "and double the peak memory for no gain."
            ),
        }

    @app.get("/api/runs")
    def list_runs(limit: int = 50) -> dict[str, Any]:
        runner = _runner(app, config)
        return {
            "runs": [r.to_dict() for r in runner.store.recent(limit=min(limit, 200))],
            "root": str(runner.store.root),
            "isolation_note": (
                "Runs made here write only into their own directory. The benchmark "
                "under data/reports is mounted read-only: it holds verifications "
                "that cost money and most entry points would overwrite it."
            ),
        }

    @app.post("/api/runs", dependencies=[token_guard])
    def submit_run(payload: dict[str, Any]) -> dict[str, Any]:
        """Queue an operation. Refuses anything that would spend without permission."""
        from satchangegate.services import SERVICES
        from satchangegate.webui.jobs import QueueFull

        name = str(payload.get("operation", ""))
        spec = SERVICES.get(name)
        if spec is None:
            raise HTTPException(404, f"Unknown operation {name!r}")
        try:
            request = spec.request_type(**(payload.get("params") or {}))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

        # Judge the request, not the operation. `e2e --no-vlm` costs nothing but
        # CPU and is the most useful thing to run; blocking it alongside the paid
        # path would make the spend guard an obstacle rather than a control.
        reservation_id: str | None = None
        if spec.request_spends(request):
            if not config.allow_spend:
                raise HTTPException(
                    403,
                    "This request would make paid API calls and this server was started "
                    "without --allow-spend. Restart with it and a cap you are happy to "
                    "lose, or submit the same operation with the paid flags off.",
                )
            from satchangegate.webui.spend import SpendRefused

            quote_id = str(payload.get("quote_id") or "")
            if not quote_id:
                raise HTTPException(
                    402,
                    "Paid work needs a quote. Ask /api/spend/quote for a price on this "
                    "exact request, then confirm against it -- approval is for one "
                    "request, not for a session.",
                )
            try:
                reservation_id = _ledger(app, config).reserve(
                    quote_id, request=request.model_dump(mode="json")
                )
            except SpendRefused as exc:
                raise HTTPException(402, str(exc)) from None

        runner = _runner(app, config)
        params = request.model_dump(mode="json")

        def prepare(run_id: str, queued: dict[str, Any]) -> None:
            # Point the run's output at a directory named after it. This has to
            # happen before the work is enqueued: rewriting the payload
            # afterwards races a worker that may already have picked it up.
            if spec.isolate_out and "out" in spec.request_type.model_fields:
                queued["out"] = _isolated_out(runner, run_id, spec)

        try:
            run_id = runner.submit(
                name,
                params,
                command=request.to_command(),
                display_name=str(payload.get("name") or name),
                fingerprint=provenance_fingerprint(),
                prepare=prepare,
            )
        except QueueFull as exc:
            if reservation_id:
                # Nothing was dispatched, so this hold can be given back.
                _ledger(app, config).release(reservation_id, "queue full; never dispatched")
            raise HTTPException(429, str(exc)) from None
        if reservation_id:
            app.state.reservations[run_id] = reservation_id
        return {
            "run_id": run_id,
            "command": request.to_command(),
            "reservation_id": reservation_id,
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        runner = _runner(app, config)
        run = runner.store.get(run_id)
        if run is None:
            raise HTTPException(404, f"No run {run_id!r}")
        return {**run.to_dict(), "complete_bundle": runner.store.is_complete(run_id)}

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, after: int = 0) -> dict[str, Any]:
        runner = _runner(app, config)
        if runner.store.get(run_id) is None:
            raise HTTPException(404, f"No run {run_id!r}")
        events = runner.store.events(run_id, after=after)
        return {"run_id": run_id, "after": after, "events": events}

    @app.get("/api/runs/{run_id}/stream")
    def stream_run(run_id: str, after: int = 0) -> Response:
        """Server-sent events over the durable log.

        SSE is transport here and nothing more: every event was written to the
        store before it was streamed, so a reconnect with ``after`` resumes
        exactly, and a client that never connects loses nothing.
        """
        from fastapi.responses import StreamingResponse

        runner = _runner(app, config)
        if runner.store.get(run_id) is None:
            raise HTTPException(404, f"No run {run_id!r}")

        def events() -> Any:
            import json as _json
            import time as _time

            cursor = after
            deadline = _time.time() + 3600
            while _time.time() < deadline:
                for event in runner.store.events(run_id, after=cursor):
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: {event['kind']}\n"
                    yield f"data: {_json.dumps(event)}\n\n"
                run = runner.store.get(run_id)
                if run and run.status in ("succeeded", "failed", "cancelled"):
                    yield f"event: done\ndata: {_json.dumps({'status': run.status})}\n\n"
                    return
                _time.sleep(0.4)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/cancel", dependencies=[token_guard])
    def cancel_run(run_id: str) -> dict[str, Any]:
        runner = _runner(app, config)
        if runner.store.get(run_id) is None:
            raise HTTPException(404, f"No run {run_id!r}")
        asked = runner.cancel(run_id)
        return {
            "run_id": run_id,
            "cancelling": asked,
            "note": (
                "Cooperative. Work already dispatched to a provider is not made "
                "cheaper by stopping the thread that was waiting for it."
            ),
        }

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


def _scorer_status() -> dict[str, Any]:
    """Which scorer is configured, and which code paths actually honour it.

    ``evaluate.score_rows`` dispatches between the rules and the learned model.
    ``classical_gate`` and ``_gate_pass`` call ``decide`` directly and always
    run the rules. A single global toggle would therefore show learned metrics
    beside a funnel that ran the rules, so the honest move -- short of unifying
    the two, which is a larger change than this milestone -- is to say plainly
    where the setting applies and where it does not.
    """
    from satchangegate.config import get_settings

    settings = get_settings()
    kind = settings.scorer.kind
    status: dict[str, Any] = {
        "configured": kind,
        "threshold": settings.scorer.threshold,
        "honoured_by": ["eval", "conformal", "operating-points", "baselines"],
        "always_uses_rules": ["run", "run-images", "e2e", "dev-tests"],
        "caveat": (
            "The single-pair and funnel paths call decide() directly, so they run "
            "the rule gate whatever this is set to. Comparing a learned metric "
            "against a funnel figure would be comparing two different models."
        ),
        "artifact": None,
    }
    if kind != "learned":
        return status
    try:
        from satchangegate.scorer import load_scorer

        artifact = load_scorer(Path(settings.scorer.path))
        status["artifact"] = {
            "loaded": True,
            "model": getattr(artifact, "model_name", None) or type(artifact).__name__,
            "feature_hash": getattr(artifact, "feature_hash", None),
        }
    except Exception as exc:
        # A stale artifact makes evaluate fall back to the rules with a
        # RuntimeWarning nobody sees. Surfacing it is the point.
        status["artifact"] = {
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
            "consequence": (
                "Scoring silently falls back to the rule gate. Any metric labelled "
                "'learned' while this is true would be mislabelled."
            ),
        }
    return status


def _spent_usd(result: dict[str, Any] | None) -> float:
    """What a finished run actually cost, from its own reported usage."""
    if not result:
        return 0.0
    funnel = result.get("funnel_cost") or {}
    cost = funnel.get("cost_usd") or {}
    if isinstance(cost, dict) and "total" in cost:
        return float(cost["total"] or 0.0)
    return float(result.get("cost_usd") or 0.0)


def _ledger(app: FastAPI, config: AppConfig) -> Any:
    """The process-wide spend ledger, created on first use."""
    existing = getattr(app.state, "ledger", None)
    if existing is not None:
        return existing
    from satchangegate.webui.spend import SpendLedger

    ledger = SpendLedger(_run_root(config) / "spend.db", config.spend_cap_usd)
    app.state.ledger = ledger
    return ledger


def _run_root(config: AppConfig) -> Path:
    from satchangegate.webui.runs import DEFAULT_RUN_ROOT

    return Path(config.run_root or DEFAULT_RUN_ROOT)


def _runner(app: FastAPI, config: AppConfig) -> Any:
    """The process-wide job runner, created on first use.

    Lazily, because constructing it takes the run-store write lock, and a server
    that never runs anything should not hold a lock that stops another one from
    starting.
    """
    existing = getattr(app.state, "runner", None)
    if existing is not None:
        return existing
    from satchangegate.webui.jobs import JobRunner
    from satchangegate.webui.runs import DEFAULT_RUN_ROOT, RunStore

    store = RunStore(config.run_root or DEFAULT_RUN_ROOT)

    def on_settled(run_id: str, result: dict[str, Any] | None, error: str | None) -> None:
        reservation = app.state.reservations.pop(run_id, None)
        if reservation is None:
            return
        ledger = _ledger(app, config)
        if error is not None:
            # The run failed somewhere, and from here there is no way to know
            # whether a request had already reached the provider. The hold stays,
            # and a human releases it after looking.
            ledger.mark_uncertain(reservation, f"run failed: {error}")
            return
        spent = _spent_usd(result)
        ledger.settle(reservation, spent)

    runner = JobRunner(store, on_settled=on_settled)
    app.state.runner = runner
    return runner


def _isolated_out(runner: Any, run_id: str, spec: Any) -> str:
    """Where a run writes: its own directory, never the benchmark's.

    An operation whose default ``out`` is a file keeps that filename inside the
    run directory; one that writes a directory of reports gets the directory
    itself. ``download-oscd`` is excluded upstream, because its ``out`` is where
    the dataset lives rather than where a result goes.
    """
    directory = runner.store.run_dir(run_id)
    default = spec.request_type.model_fields["out"].default
    name = Path(str(default)).name if default is not None else ""
    return str(directory / name) if Path(str(default)).suffix else str(directory)


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
