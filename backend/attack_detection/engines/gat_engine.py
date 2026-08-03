"""Project-level GATv2 inference adapter."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from attack_detection.contracts import EngineResult
from attack_detection.cancellation import run_cancellable
from attack_detection.training.artifact_contracts import validate_gat_manifest

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BACKEND_DIR / "models"
DEFAULT_DEEP_PYTHON = Path(r"D:\software\anaconda\envs\drone\python.exe")
PROJECT_INFERENCE_TIMEOUT_SECONDS = max(
    5.0,
    float(os.environ.get(
        "XIEZHI_GAT_PROJECT_TIMEOUT_SECONDS",
        "60",
    )),
)


class GATEngine:
    name = "gatv2"

    def scan(self, content: str, language: str) -> dict[str, Any]:
        return EngineResult(
            name=self.name,
            status="skipped",
            reason="GATv2 was not executed: the validated model requires a project-level graph",
            duration_ms=0,
        ).to_dict()

    def scan_project(
        self, graph: dict[str, Any], cancel_event: object | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        manifest_path = MODEL_DIR / "gatv2_manifest.json"
        python_path = Path(os.environ.get("XIEZHI_DEEP_PYTHON") or DEFAULT_DEEP_PYTHON)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._unavailable(start, f"GATv2 manifest cannot be read: {exc}")
        errors = validate_gat_manifest(manifest, MODEL_DIR)
        if errors:
            return self._unavailable(start, "; ".join(errors))
        supported_languages = set(manifest.get("supported_languages") or [])
        graph_language_counts = Counter(
            str(node.get("language") or "").lower()
            for node in (graph.get("nodes") or [])
            if node.get("type") == "file" and node.get("language")
        )
        graph_languages = set(graph_language_counts)
        if supported_languages and not graph_languages.intersection(supported_languages):
            return EngineResult(
                name=self.name, status="skipped",
                reason=(
                    "GATv2 was not executed: the project has no language with "
                    "validated malicious/benign graph coverage"
                ),
                duration_ms=int((time.perf_counter() - start) * 1000),
                metadata={"project_languages": sorted(graph_languages)},
            ).to_dict()
        eligible_counts = {
            language: count
            for language, count in graph_language_counts.items()
            if not supported_languages or language in supported_languages
        }
        route_language = max(
            eligible_counts,
            key=lambda language: (eligible_counts[language], language),
            default=None,
        )
        route_settings = (manifest.get("language_models") or {}).get(
            route_language,
            {},
        )
        training = route_settings.get("training") or manifest.get("training") or {}
        feature_schema_version = int(
            training.get("feature_schema_version") or 1
        )
        minimum_nodes = 2 if feature_schema_version >= 7 else 3
        minimum_edges = 1 if feature_schema_version >= 7 else 2
        if (
            int(graph.get("node_count") or 0) < minimum_nodes
            or int(graph.get("edge_count") or 0) < minimum_edges
        ):
            return EngineResult(
                name=self.name,
                status="skipped",
                reason="GATv2 was not executed: graph structure is insufficient",
                duration_ms=int((time.perf_counter() - start) * 1000),
                metadata={
                    "node_count": graph.get("node_count"),
                    "edge_count": graph.get("edge_count"),
                    "feature_schema_version": feature_schema_version,
                },
            ).to_dict()
        if not python_path.is_file():
            return self._unavailable(start, f"PyTorch interpreter is unavailable: {python_path}")
        try:
            completed = run_cancellable(
                [str(python_path), "-m", "attack_detection.training.gat_infer", "--model-dir", str(MODEL_DIR)],
                input_text=json.dumps(graph, ensure_ascii=False),
                cwd=str(BACKEND_DIR),
                timeout=PROJECT_INFERENCE_TIMEOUT_SECONDS,
                cancel_event=cancel_event,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return EngineResult(
                name=self.name, status="failed", error=f"GATv2 inference process failed: {exc}",
                duration_ms=int((time.perf_counter() - start) * 1000),
                metadata={
                    "node_count": graph.get("node_count"),
                    "edge_count": graph.get("edge_count"),
                    "project_languages": sorted(graph_languages),
                    "timeout_seconds": PROJECT_INFERENCE_TIMEOUT_SECONDS,
                },
            ).to_dict()
        if completed.returncode != 0:
            return EngineResult(
                name=self.name, status="failed", error=(completed.stderr or completed.stdout or "unknown error")[-500:],
                duration_ms=int((time.perf_counter() - start) * 1000),
            ).to_dict()
        try:
            output = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return EngineResult(
                name=self.name, status="failed", error="GATv2 inference returned invalid JSON",
                duration_ms=int((time.perf_counter() - start) * 1000),
            ).to_dict()
        return EngineResult(
            name=self.name, status="completed", decision=str(output["decision"]),
            probability=float(output["probability"]), threshold=float(output["threshold"]),
            model_version=str(output["model_version"]), duration_ms=int((time.perf_counter() - start) * 1000),
            metadata={
                "node_count": graph.get("node_count"), "edge_count": graph.get("edge_count"),
                "route_language": output.get("route_language"),
                "artifact_version": output.get("artifact_version"),
            },
        ).to_dict()

    def _unavailable(self, start: float, reason: str) -> dict[str, Any]:
        return EngineResult(
            name=self.name, status="unavailable", reason=reason,
            duration_ms=int((time.perf_counter() - start) * 1000),
        ).to_dict()
