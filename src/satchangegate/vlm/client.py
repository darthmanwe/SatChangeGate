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
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

from satchangegate.vlm.schemas import VlmVerdict

DEFAULT_VLM_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 4

# USD per million tokens, by model. Used for the cost model the funnel thesis
# depends on. Update alongside published pricing.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
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

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = PRICING_USD_PER_MTOK.get(self.model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1_000_000


def resolve_model(explicit: str | None, env_var: str, default: str) -> str:
    """Resolve a model id at call time: explicit > environment > default."""
    return explicit or os.environ.get(env_var) or default


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _usage_from(response: object, model: str, kind: str) -> UsageRecord:
    usage = getattr(response, "usage", None)
    return UsageRecord(
        model=model,
        kind=kind,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
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
