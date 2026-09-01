"""Anthropic vision client for candidate-change verification.

Changes from the previous implementation, each of which was a live defect:

* The default model was ``claude-sonnet-4-20250514``, which is at/past its
  published retirement date — the headline feature returned 404 on a fresh clone.
* Model IDs were read from ``os.environ`` at *import* time, before ``.env`` was
  loaded, so the documented ``ANTHROPIC_VLM_MODEL`` override silently never
  worked. They are now read at call time.
* The JSON schema was described in prose and scraped back out with a greedy
  regex. Structured outputs enforce the schema server-side instead.
* Retry was ``except Exception: return _call()`` — it retried 401s and 400s,
  stacked on the SDK's own retries (up to 6 HTTP requests on a 429), and
  swallowed ValidationError. The SDK's typed retry handles transport now.
* Token usage was never read, so the repo made cost-reduction claims with no
  cost instrumentation at all.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from satchangegate.vlm.schemas import VlmVerdict

DEFAULT_VLM_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 4

# USD per million tokens, by model. Used for the cost model the funnel thesis
# depends on. Update alongside published pricing.
#
# PROVENANCE. An earlier version of this table priced claude-sonnet-5 at
# (3.00, 15.00). That was a scheduled 2026-09-01 increase which was cancelled and
# never took effect; the standard rate is and remains $2/$10. Every cost figure
# this repo published before 2026-08-29 was therefore ~50% too high. The
# percentage saving was unaffected -- it is the gate's filter rate -- but the
# dollar figures were not, and they have been re-derived from the token counts
# the API returned rather than re-measured.
#
# Rates are held as a record rather than a tuple because a flat (in, out) pair
# cannot express the two discounts this funnel actually uses: the Batch API's
# 50% on both directions, and prompt caching's 0.1x read / 1.25x write.


@dataclass(frozen=True)
class ModelRate:
    """Published USD-per-million-token rates for one model."""

    input_usd: float
    output_usd: float
    # Batch API: 50% off both directions. Held explicitly rather than derived so
    # a model that ever departs from the flat 50% can be represented honestly.
    batch_input_usd: float
    batch_output_usd: float
    # Prompt caching multipliers, relative to the base input rate.
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25

    @classmethod
    def with_batch_discount(cls, input_usd: float, output_usd: float) -> ModelRate:
        return cls(input_usd, output_usd, input_usd / 2, output_usd / 2)


PRICING_USD_PER_MTOK: dict[str, ModelRate] = {
    "claude-opus-5": ModelRate.with_batch_discount(5.00, 25.00),
    "claude-sonnet-5": ModelRate.with_batch_discount(2.00, 10.00),
    "claude-sonnet-4-6": ModelRate.with_batch_discount(3.00, 15.00),
    "claude-haiku-4-5": ModelRate.with_batch_discount(1.00, 5.00),
}

VLM_SYSTEM = """You are a remote-sensing change verification analyst.

You are given before/after satellite images of the same area and a heatmap
highlighting where a detector flagged difference. Decide whether the difference
reflects a real, persistent physical change on the ground, or an artifact of
weather, season, illumination, or image misalignment.

Judge from the imagery. The metadata gives acquisition dates and observation
quality only; it deliberately does not tell you what any detector concluded.
Base visual_evidence on what you can actually see in the images."""


class UsageRecord(BaseModel):
    """Token usage and derived cost for a single API call."""

    model: str
    kind: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    # True when this call was submitted through the Batch API, which is billed
    # at half rate in both directions. Recorded per call rather than per run so a
    # mixed run (some resumed from a synchronous pass, some batched) still prices
    # each call the way it was actually billed.
    batch: bool = False

    @property
    def priced(self) -> bool:
        """False when no published rate is known for this model."""
        return self.model in PRICING_USD_PER_MTOK

    @property
    def cost_usd(self) -> float:
        """USD for this call, or 0.0 when the model has no published rate.

        Check ``priced`` before reporting: an unpriced model would otherwise
        make every downstream cost figure read $0.00, which is far worse than
        reading "unknown" — the override the README invites
        (ANTHROPIC_VLM_MODEL) is exactly how a caller reaches this path.

        ``input_tokens`` from the API excludes cached reads and cache writes, so
        the three input terms are added rather than apportioned.
        """
        rate = PRICING_USD_PER_MTOK.get(self.model)
        if rate is None:
            return 0.0
        rate_in = rate.batch_input_usd if self.batch else rate.input_usd
        rate_out = rate.batch_output_usd if self.batch else rate.output_usd
        total = (
            self.input_tokens * rate_in
            + self.output_tokens * rate_out
            + self.cache_read_input_tokens * rate_in * rate.cache_read_multiplier
            + self.cache_creation_input_tokens * rate_in * rate.cache_write_multiplier
        )
        return total / 1_000_000


def resolve_model(explicit: str | None, env_var: str, default: str) -> str:
    """Resolve a model id at call time: explicit > environment > default."""
    return explicit or os.environ.get(env_var) or default


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _usage_from(response: object, model: str, kind: str, *, batch: bool = False) -> UsageRecord:
    usage = getattr(response, "usage", None)
    return UsageRecord(
        model=model,
        kind=kind,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        batch=batch,
    )


def build_vlm_messages(package_dir: Path) -> list[dict]:
    """Assemble the user turn: redacted metadata plus the evidence images."""
    package_dir = Path(package_dir)
    meta_path = package_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"VLM package missing metadata.json: {package_dir}")

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Verify this satellite change-detection candidate.\n\n"
                "Acquisition metadata:\n" + meta_path.read_text(encoding="utf-8")
            ),
        }
    ]
    # The overlay composites the heatmap onto the after-image, which is far more
    # interpretable than the bare grayscale heatmap. It was previously written
    # to disk and never sent.
    for name in ("before_rgb.png", "after_rgb.png", "change_overlay.png"):
        path = package_dir / name
        if path.is_file():
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode_image(path),
                    },
                }
            )
    n_images = sum(1 for c in content if c["type"] == "image")
    if n_images == 0:
        raise FileNotFoundError(
            f"VLM package {package_dir} contains no images; refusing to ask for a "
            "visual verdict with no imagery."
        )
    return content


def verify_candidate(
    package_dir: Path,
    api_key: str | None = None,
    model: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: anthropic.Anthropic | None = None,
) -> tuple[VlmVerdict, UsageRecord]:
    """Ask the vision model to verify a gate candidate.

    Returns the validated verdict and a usage record. A malformed or refused
    response degrades to an ``uncertain`` verdict rather than raising, so one bad
    response cannot abort a long batch.
    """
    model = resolve_model(model, "ANTHROPIC_VLM_MODEL", DEFAULT_VLM_MODEL)
    if client is None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(
            api_key=key, timeout=DEFAULT_TIMEOUT_S, max_retries=DEFAULT_MAX_RETRIES
        )

    content = build_vlm_messages(package_dir)
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=VLM_SYSTEM,
        messages=[{"role": "user", "content": content}],  # type: ignore[typeddict-item]
        output_format=VlmVerdict,
    )
    record = _usage_from(response, model, "vlm")

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        return VlmVerdict.uncertain_fallback("model declined the request"), record
    if stop_reason == "max_tokens":
        return VlmVerdict.uncertain_fallback("response truncated at max_tokens"), record

    parsed = getattr(response, "parsed_output", None)
    if isinstance(parsed, VlmVerdict):
        return parsed, record
    try:
        return VlmVerdict.model_validate(parsed), record
    except ValidationError as exc:
        return VlmVerdict.uncertain_fallback(f"schema validation failed: {exc}"), record


# --- Batch API -------------------------------------------------------------
#
# The funnel's workload is the exact shape the Batch API is for: hundreds of
# independent, latency-insensitive verifications. It bills at half rate in both
# directions, and -- more usefully for this repo -- it puts a second price in the
# funnel, so the reported saving stops being the candidate rate under another
# name.
#
# The batch path cannot use the ``messages.parse`` helper, so the schema is built
# here and the response is validated client-side by the same
# ``model_validate`` -> ``uncertain_fallback`` chain the synchronous path already
# falls back to.

DEFAULT_BATCH_POLL_S = 20.0
DEFAULT_BATCH_TIMEOUT_S = 24 * 3600.0


def verdict_output_config() -> dict:
    """Structured-output config for ``VlmVerdict``, for requests built by hand.

    Prefers the SDK's own schema transform so a batch request and a
    ``messages.parse`` request submit an identical schema. That helper is private,
    so a local fallback inlines pydantic's ``$defs`` rather than letting a future
    SDK layout silently change what the model is asked for.
    """
    schema: dict
    try:
        from anthropic.lib._parse._transform import transform_schema

        schema = transform_schema(VlmVerdict)
    except Exception:  # pragma: no cover - exercised only on SDK layout changes
        schema = _inline_defs(VlmVerdict.model_json_schema())
    return {"format": {"type": "json_schema", "schema": schema}}


def _inline_defs(schema: dict) -> dict:
    """Resolve ``$ref``/``$defs`` into a self-contained schema."""
    defs = schema.pop("$defs", {})

    def walk(node: object) -> object:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                resolved = walk(dict(target))
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**resolved, **extra} if isinstance(resolved, dict) else extra
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    out = walk(schema)
    assert isinstance(out, dict)
    return out


def build_batch_request(
    custom_id: str,
    package_dir: Path,
    *,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """One Batch API request for a packaged candidate."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": VLM_SYSTEM,
            "messages": [{"role": "user", "content": build_vlm_messages(package_dir)}],
            "output_config": verdict_output_config(),
        },
    }


def _verdict_from_message(message: Any) -> VlmVerdict:
    """Validate a batch response body into a verdict, degrading rather than raising."""
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "refusal":
        return VlmVerdict.uncertain_fallback("model declined the request")
    if stop_reason == "max_tokens":
        return VlmVerdict.uncertain_fallback("response truncated at max_tokens")

    text = ""
    blocks: list[Any] = list(getattr(message, "content", None) or [])
    for block in blocks:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "") or ""
    if not text.strip():
        return VlmVerdict.uncertain_fallback("empty response body")
    try:
        return VlmVerdict.model_validate_json(text)
    except ValidationError as exc:
        return VlmVerdict.uncertain_fallback(f"schema validation failed: {exc}")


def submit_batch(
    requests: list[dict],
    *,
    client: anthropic.Anthropic,
) -> str:
    """Submit a batch and return its id."""
    if not requests:
        raise ValueError("refusing to submit an empty batch")
    batch = client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
    return str(batch.id)


def collect_batch(
    batch_id: str,
    *,
    client: anthropic.Anthropic,
    model: str,
    poll_interval_s: float = DEFAULT_BATCH_POLL_S,
    timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    on_poll: Callable[[str], None] | None = None,
) -> dict[str, tuple[VlmVerdict | None, UsageRecord | None, str | None]]:
    """Wait for a batch and return results keyed by ``custom_id``.

    Keyed by id and never by position: batch results come back in arbitrary order,
    and zipping them against the submitted list would mislabel every verdict.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = str(batch.processing_status)
        if on_poll is not None:
            on_poll(status)
        if status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"batch {batch_id} still {status} after {timeout_s:.0f}s")
        time.sleep(poll_interval_s)

    out: dict[str, tuple[VlmVerdict | None, UsageRecord | None, str | None]] = {}
    for entry in client.messages.batches.results(batch_id):
        custom_id = str(entry.custom_id)
        result = entry.result
        kind = getattr(result, "type", None)
        message = getattr(result, "message", None)
        if kind != "succeeded" or message is None:
            detail = getattr(getattr(result, "error", None), "type", kind)
            out[custom_id] = (None, None, f"batch result {kind}: {detail}")
            continue
        record = _usage_from(message, model, "vlm", batch=True)
        out[custom_id] = (_verdict_from_message(message), record, None)
    return out
