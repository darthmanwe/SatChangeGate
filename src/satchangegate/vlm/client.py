"""Anthropic VLM client for candidate-change verification."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from anthropic import Anthropic

from satchangegate.vlm.schemas import VlmVerdict

DEFAULT_VLM_MODEL = os.environ.get("ANTHROPIC_VLM_MODEL", "claude-sonnet-4-20250514")

VLM_SYSTEM = """You are a remote-sensing change verification analyst.
Given before/after satellite RGB images and a change heatmap, decide if the detected change is real physical change or an artifact.
Respond with JSON only matching this schema:
{
  "vlm_verdict": "real_change" | "likely_artifact" | "uncertain",
  "change_type": "construction" | "demolition" | "vegetation_clearing" | "flooding" | "no_meaningful_change" | "weather_artifact" | "alignment_artifact" | "uncertain",
  "visual_evidence": ["..."],
  "artifact_risk": {"cloud_shadow": "low|medium|high", "snow": "low|medium|high", "seasonality": "low|medium|high", "registration": "low|medium|high"},
  "confidence": 0.0-1.0,
  "requires_human_review": true|false
}"""


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.standard_b64encode(data).decode("utf-8")


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


def verify_candidate(
    package_dir: Path,
    api_key: str | None = None,
    model: str | None = None,
) -> VlmVerdict:
    """Call Anthropic vision API on a candidate package."""
    package_dir = Path(package_dir)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)
    model = model or DEFAULT_VLM_MODEL

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    meta_text = json.dumps(metadata, indent=2)

    images = [
        package_dir / "before_rgb.png",
        package_dir / "after_rgb.png",
        package_dir / "change_heatmap.png",
    ]
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Verify this satellite change detection candidate. Metadata:\n{meta_text}",
        }
    ]
    for img_path in images:
        if img_path.is_file():
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode_image(img_path),
                    },
                }
            )

    def _call() -> VlmVerdict:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=VLM_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text
        data = _extract_json(text)
        return VlmVerdict.model_validate(data)

    try:
        return _call()
    except Exception:
        return _call()
