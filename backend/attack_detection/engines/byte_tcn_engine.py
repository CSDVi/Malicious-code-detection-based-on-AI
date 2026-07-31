"""ByteCNN-TCN inference adapter using the configured PyTorch interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from attack_detection.contracts import EngineResult
from attack_detection.cancellation import run_cancellable
from attack_detection.training.artifact_contracts import validate_bytetcn_manifest

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BACKEND_DIR / "models"
DEFAULT_DEEP_PYTHON = Path(r"D:\software\anaconda\envs\drone\python.exe")


class ByteTCNEngine:
    name = "bytetcn"

    def scan(
        self, content: str, language: str, cancel_event: object | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        manifest_path = MODEL_DIR / "bytetcn_manifest.json"
        python_path = Path(os.environ.get("XIEZHI_DEEP_PYTHON") or DEFAULT_DEEP_PYTHON)
        if not manifest_path.is_file():
            return self._unavailable(start, "ByteCNN-TCN model manifest is not loaded")
        if not python_path.is_file():
            return self._unavailable(start, f"PyTorch interpreter is unavailable: {python_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._failed(start, f"ByteCNN-TCN manifest cannot be read: {exc}")
        errors = validate_bytetcn_manifest(manifest, MODEL_DIR)
        if errors:
            return self._unavailable(start, "; ".join(errors))
        try:
            completed = run_cancellable(
                [str(python_path), "-m", "attack_detection.training.byte_tcn_infer", "--model-dir", str(MODEL_DIR)],
                input_text=json.dumps(
                    {"content": content, "language": language}, ensure_ascii=False,
                ),
                cwd=str(BACKEND_DIR),
                timeout=45,
                cancel_event=cancel_event,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._failed(start, f"ByteCNN-TCN inference process failed: {exc}")
        if completed.returncode != 0:
            return self._failed(start, (completed.stderr or completed.stdout or "unknown inference error")[-500:])
        try:
            output = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return self._failed(start, "ByteCNN-TCN inference returned invalid JSON")
        if output.get("status") != "completed":
            return self._unavailable(start, str(output.get("reason") or "language/task is not validated"), output.get("model_version"))
        return self._shape_completed(output, content, start)

    def scan_batch(
        self, requests: list[dict[str, str]], cancel_event: object | None = None,
    ) -> list[dict[str, Any]]:
        """Run project candidates in one external process to avoid repeated PyTorch startup."""

        if not requests:
            return []
        start = time.perf_counter()
        manifest_path = MODEL_DIR / "bytetcn_manifest.json"
        python_path = Path(os.environ.get("XIEZHI_DEEP_PYTHON") or DEFAULT_DEEP_PYTHON)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [self._unavailable(start, f"ByteCNN-TCN manifest cannot be read: {exc}") for _ in requests]
        errors = validate_bytetcn_manifest(manifest, MODEL_DIR)
        if errors or not python_path.is_file():
            reason = "; ".join(errors) if errors else f"PyTorch interpreter is unavailable: {python_path}"
            return [self._unavailable(start, reason, manifest.get("model_version")) for _ in requests]
        try:
            completed = run_cancellable(
                [str(python_path), "-m", "attack_detection.training.byte_tcn_infer", "--model-dir", str(MODEL_DIR)],
                input_text=json.dumps({"requests": requests}, ensure_ascii=False),
                cwd=str(BACKEND_DIR),
                timeout=max(60, len(requests) * 8),
                cancel_event=cancel_event,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [self._failed(start, f"ByteCNN-TCN batch inference failed: {exc}") for _ in requests]
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "unknown batch inference error")[-500:]
            return [self._failed(start, reason) for _ in requests]
        try:
            outputs = json.loads(completed.stdout).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            return [self._failed(start, "ByteCNN-TCN batch inference returned invalid JSON") for _ in requests]
        results = []
        for index, request in enumerate(requests):
            if index >= len(outputs):
                results.append(self._failed(start, "ByteCNN-TCN batch response is incomplete"))
                continue
            output = outputs[index]
            if output.get("status") != "completed":
                results.append(self._unavailable(
                    start, str(output.get("reason") or "language/task is not validated"), output.get("model_version"),
                ))
            else:
                results.append(self._shape_completed(
                    output, str(request.get("content") or ""), start,
                    duration_ms=int(output.get("duration_ms") or 0),
                ))
        return results

    def _shape_completed(
        self, output: dict[str, Any], content: str, start: float, duration_ms: int | None = None,
    ) -> dict[str, Any]:
        lines = content.splitlines()
        findings = [{
            "source": self.name,
            "category": "ByteCNN-TCN 行级证据",
            "severity": 6,
            "line": int(item["line"]),
            "snippet": lines[int(item["line"]) - 1][:240] if 0 < int(item["line"]) <= len(lines) else "",
            "behavior": output.get("primary_task"),
            "risk_type": "malicious" if output.get("primary_task") == "malicious_intent" else "vulnerable",
            "description": "ByteCNN-TCN 在该行定位到与风险结论相关的代码特征。",
            "suspicion_score": round(float(item["score"]) * 100, 1),
        } for item in output.get("line_scores", [])]
        return EngineResult(
            name=self.name,
            status="completed",
            decision=str(output["decision"]),
            probability=float(output["probability"]),
            threshold=float(output["threshold"]),
            model_version=str(output["model_version"]),
            duration_ms=duration_ms if duration_ms is not None else int((time.perf_counter() - start) * 1000),
            findings=findings,
            metadata={
                "primary_task": output.get("primary_task"),
                "task_probabilities": output.get("task_probabilities", {}),
                "task_thresholds": output.get("task_thresholds", {}),
                "behavior_labels": output.get("behavior_labels", []),
                "cwe_labels": output.get("cwe_labels", []),
            },
        ).to_dict()

    def _unavailable(self, start: float, reason: str, version: str | None = None) -> dict[str, Any]:
        return EngineResult(
            name=self.name, status="unavailable", reason=reason, model_version=version,
            duration_ms=int((time.perf_counter() - start) * 1000),
        ).to_dict()

    def _failed(self, start: float, reason: str) -> dict[str, Any]:
        return EngineResult(
            name=self.name, status="failed", error=reason,
            duration_ms=int((time.perf_counter() - start) * 1000),
        ).to_dict()
