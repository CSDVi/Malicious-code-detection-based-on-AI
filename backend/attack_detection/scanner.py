"""Backward-compatible scanner entry points."""

from __future__ import annotations

from typing import Callable

from .languages import (
    BINARY_EXTENSIONS,
    SOURCE_EXTENSIONS,
    detect_source_language,
    display_language,
)
from .orchestrator import DetectionOrchestrator

ALLOWED_EXTENSIONS = SOURCE_EXTENSIONS | BINARY_EXTENSIONS

_orchestrator = DetectionOrchestrator()


def detect_language(filename: str, content: str | None = None) -> str:
    return detect_source_language(filename, content)


def is_allowed_file(filename: str) -> bool:
    from pathlib import PurePath

    return PurePath(filename.lower()).suffix in ALLOWED_EXTENSIONS


def scan_code(
    filename: str, content: str, mode: str = "auto",
    precomputed_semantic: dict[str, object] | None = None,
    cancel_event: object | None = None,
) -> dict[str, object]:
    """Scan code through the new orchestrator while preserving legacy fields."""
    language = detect_language(filename, content)
    result = _orchestrator.scan(
        filename, content, language, selected_mode=mode,
        precomputed_semantic=precomputed_semantic,
        cancel_event=cancel_event,
    )
    result["display_language"] = display_language(language, filename)
    return result


def scan_file(
    filename: str, payload: bytes, mode: str = "auto",
    precomputed_semantic: dict[str, object] | None = None,
    precomputed_quick_result: dict[str, object] | None = None,
    cancel_event: object | None = None,
    generate_line_attributions: bool | None = None,
    analysis_max_bytes: int | None = None,
    precomputed_xgb: dict[str, object] | None = None,
    run_legacy_baseline: bool = True,
) -> dict[str, object]:
    """Scan an upload while preserving its original bytes for hashes and PE parsing."""
    analysis_payload = payload
    analysis_truncated = False
    if analysis_max_bytes and len(payload) > analysis_max_bytes:
        marker = b"\n/* ... project quick-scan middle omitted ... */\n"
        remaining = max(2, int(analysis_max_bytes) - len(marker))
        head_size = remaining // 2
        analysis_payload = (
            payload[:head_size]
            + marker
            + payload[-(remaining - head_size):]
        )
        analysis_truncated = True
    content = analysis_payload.decode("utf-8", errors="ignore")
    language = detect_language(filename, content)
    if language == "binary" or payload[:2] == b"MZ":
        result = _orchestrator.scan_binary(
            filename, payload, selected_mode=mode, cancel_event=cancel_event,
        )
        result["display_language"] = display_language("binary", filename)
        return result
    result = _orchestrator.scan(
        filename, content, language, selected_mode=mode,
        precomputed_semantic=precomputed_semantic,
        precomputed_quick_result=precomputed_quick_result,
        raw_bytes=payload,
        cancel_event=cancel_event,
        generate_line_attributions=generate_line_attributions,
        precomputed_xgb=precomputed_xgb,
        run_legacy_baseline=run_legacy_baseline,
    )
    result["display_language"] = display_language(language, filename)
    if analysis_truncated:
        result["analysis_truncated"] = True
        result["analysis_bytes"] = len(analysis_payload)
        result["original_bytes"] = len(payload)
    return result


def prepare_xgb_batch(
    requests: list[dict[str, str]],
    cancel_event: object | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, object]]:
    """Use the process-wide XGBoost engine to batch project route matrices."""

    return _orchestrator.xgb_engine.prepare_batch(
        requests,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )


def scan_xgb_with_attributions(
    content: str,
    language: str,
    *,
    cancel_event: object | None = None,
    precomputed_batch: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Generate only XGBoost explanations while reusing the project baseline."""

    return _orchestrator.xgb_engine.scan(
        content,
        language,
        generate_line_attributions=True,
        cancel_event=cancel_event,
        precomputed_batch=precomputed_batch,
    )


def scan_xgb_prepared(
    content: str,
    language: str,
    *,
    cancel_event: object | None = None,
    precomputed_batch: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Shape already-batched XGBoost probabilities without line occlusion."""

    return _orchestrator.xgb_engine.scan(
        content,
        language,
        generate_line_attributions=False,
        cancel_event=cancel_event,
        precomputed_batch=precomputed_batch,
    )


def scan_xgb_attribution_batch(
    requests: list[dict[str, str]],
    *,
    prepared_batch: list[dict[str, object]] | None = None,
    cancel_event: object | None = None,
) -> list[list[dict[str, object]]]:
    """Batch line-occlusion candidates across project files by model route."""

    return _orchestrator.xgb_engine.explain_batch(
        requests,
        prepared_batch=prepared_batch,
        cancel_event=cancel_event,
    )
