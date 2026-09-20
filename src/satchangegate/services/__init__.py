"""Shared application services: one validated request per operation.

The CLI and the web API both go through here, which is the only reason the UI can
claim that a displayed command reproduces what is on screen.
"""

from __future__ import annotations

from satchangegate.services.base import ServiceRequest
from satchangegate.services.operations import (
    SERVICES,
    AbNormalizeRequest,
    BaselinesRequest,
    ConformalRequest,
    DevTestsRequest,
    DownloadRequest,
    E2ERequest,
    EmbeddingCoverageRequest,
    EvalRequest,
    FitScorerRequest,
    OperatingPointsRequest,
    RunImagesRequest,
    RunPairRequest,
    ServiceResult,
    ServiceSpec,
    ServiceUnavailable,
    TilesRequest,
    TuneRequest,
    VerifyRequest,
    VlmReportRequest,
    run_service,
)

__all__ = [
    "SERVICES",
    "AbNormalizeRequest",
    "BaselinesRequest",
    "ConformalRequest",
    "DevTestsRequest",
    "DownloadRequest",
    "E2ERequest",
    "EmbeddingCoverageRequest",
    "EvalRequest",
    "FitScorerRequest",
    "OperatingPointsRequest",
    "RunImagesRequest",
    "RunPairRequest",
    "ServiceRequest",
    "ServiceResult",
    "ServiceSpec",
    "ServiceUnavailable",
    "TilesRequest",
    "TuneRequest",
    "VerifyRequest",
    "VlmReportRequest",
    "run_service",
]
