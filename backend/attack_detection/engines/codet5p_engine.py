"""CodeT5+ inference adapter using the configured deep-learning interpreter."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from attack_detection.contracts import EngineResult
from attack_detection.cancellation import (
    ScanCancelled,
    cancellation_requested,
    run_cancellable,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BACKEND_DIR / "models"
_PERSISTENT_LOCK = threading.Lock()
_PERSISTENT_PROCESS: subprocess.Popen[str] | None = None
PROJECT_INFERENCE_TIMEOUT_SECONDS = max(
    5.0,
    float(os.environ.get(
        "XIEZHI_CODET5_PROJECT_TIMEOUT_SECONDS",
        "12",
    )),
)
PROJECT_SINGLE_INFERENCE_TIMEOUT_SECONDS = max(
    PROJECT_INFERENCE_TIMEOUT_SECONDS,
    float(os.environ.get(
        "XIEZHI_CODET5_PROJECT_SINGLE_TIMEOUT_SECONDS",
        "14",
    )),
)


def configured_deep_python() -> Path:
    """Return an available interpreter for CodeT5+ inference.

    An explicit environment override remains authoritative. Ordinary installs
    use the same interpreter that launched the application.
    """

    configured = str(os.environ.get("XIEZHI_DEEP_PYTHON") or "").strip()
    if configured:
        return Path(configured)
    return Path(sys.executable)


def _run_persistent_batch(
    python_path: Path,
    requests: list[dict[str, str]],
    *,
    timeout: float,
    cancel_event: object | None,
) -> subprocess.CompletedProcess[str]:
    """Submit a file-backed request to the resident CodeT5 worker."""

    acquired = False
    while not acquired:
        if cancellation_requested(cancel_event):
            raise ScanCancelled("project scan cancelled before CodeT5 inference")
        acquired = _PERSISTENT_LOCK.acquire(timeout=0.1)
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        process = _ensure_persistent_process(python_path)
        request_id = uuid.uuid4().hex
        temp_root = BACKEND_DIR / "data" / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        input_fd, input_name = tempfile.mkstemp(
            prefix="codet5_request_", suffix=".json", dir=temp_root,
        )
        output_fd, output_name = tempfile.mkstemp(
            prefix="codet5_response_", suffix=".json", dir=temp_root,
        )
        os.close(input_fd)
        os.close(output_fd)
        input_path = Path(input_name)
        output_path = Path(output_name)
        input_path.write_text(
            json.dumps({"requests": requests}),
            encoding="utf-8",
        )
        command = json.dumps({
            "id": request_id,
            "input_path": str(input_path),
            "output_path": str(output_path),
        })
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("persistent CodeT5 worker pipes are unavailable")
        process.stdin.write(command + "\n")
        process.stdin.flush()

        response_queue: queue.Queue[str] = queue.Queue(maxsize=1)

        def read_response() -> None:
            try:
                response_queue.put(process.stdout.readline())
            except Exception:
                response_queue.put("")

        threading.Thread(
            target=read_response,
            name="codet5-response",
            daemon=True,
        ).start()
        deadline = time.monotonic() + timeout
        while True:
            if cancellation_requested(cancel_event):
                _stop_persistent_process()
                raise ScanCancelled("project scan cancelled during CodeT5 inference")
            if process.poll() is not None:
                _stop_persistent_process()
                raise RuntimeError(
                    f"persistent CodeT5 worker exited with code {process.returncode}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _stop_persistent_process()
                raise subprocess.TimeoutExpired(
                    ["codet5p_infer", "--server"], timeout,
                )
            try:
                line = response_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if not line.startswith("XIEZHI_RESULT "):
                _stop_persistent_process()
                raise RuntimeError(
                    f"persistent CodeT5 worker returned an invalid response: {line[-300:]}"
                )
            response = json.loads(line[len("XIEZHI_RESULT "):])
            if response.get("id") != request_id:
                _stop_persistent_process()
                raise RuntimeError("persistent CodeT5 response id mismatch")
            if response.get("status") != "completed":
                return subprocess.CompletedProcess(
                    ["codet5p_infer", "--server"],
                    1,
                    "",
                    str(response.get("error") or "persistent inference failed"),
                )
            return subprocess.CompletedProcess(
                ["codet5p_infer", "--server"],
                0,
                output_path.read_text(encoding="utf-8"),
                "",
            )
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        _PERSISTENT_LOCK.release()


def _ensure_persistent_process(
    python_path: Path,
) -> subprocess.Popen[str]:
    global _PERSISTENT_PROCESS
    if _PERSISTENT_PROCESS is not None and _PERSISTENT_PROCESS.poll() is None:
        return _PERSISTENT_PROCESS
    creation_flags = (
        subprocess.CREATE_NO_WINDOW
        if os.name == "nt"
        else 0
    )
    _PERSISTENT_PROCESS = subprocess.Popen(
        [
            str(python_path),
            "-m",
            "attack_detection.training.codet5p_infer",
            "--model-dir",
            str(MODEL_DIR),
            "--server",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        cwd=str(BACKEND_DIR),
        creationflags=creation_flags,
    )
    _assign_windows_process_cores(_PERSISTENT_PROCESS, reserve_high_cores=4)
    return _PERSISTENT_PROCESS


def _stop_persistent_process() -> None:
    global _PERSISTENT_PROCESS
    process = _PERSISTENT_PROCESS
    _PERSISTENT_PROCESS = None
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _assign_windows_process_cores(
    process: subprocess.Popen[str],
    *,
    reserve_high_cores: int,
) -> None:
    if os.name != "nt":
        return
    cpu_count = os.cpu_count() or 1
    worker_cores = max(1, cpu_count - reserve_high_cores)
    mask = (1 << worker_cores) - 1
    try:
        import ctypes

        process_handle = ctypes.windll.kernel32.OpenProcess(
            0x0200 | 0x0400,
            False,
            process.pid,
        )
        if not process_handle:
            return
        try:
            ctypes.windll.kernel32.SetProcessAffinityMask(
                process_handle,
                mask,
            )
        finally:
            ctypes.windll.kernel32.CloseHandle(process_handle)
    except (AttributeError, OSError):
        return


class CodeT5PEngine:
    name = "codet5p"

    def scan(
        self, content: str, language: str, cancel_event: object | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        registry_path = MODEL_DIR / "codet5p_registry.json"
        python_path = configured_deep_python()
        if not registry_path.is_file():
            return self._unavailable(started, "CodeT5+ 220M registry is missing")
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._unavailable(started, f"CodeT5+ 220M registry cannot be read: {exc}")
        if not registry.get("active_routes"):
            return self._unavailable(started, "CodeT5+ 220M has no strict-gated active version")
        if not python_path.is_file():
            return self._unavailable(started, f"deep-learning interpreter is unavailable: {python_path}")
        try:
            completed = run_cancellable(
                [
                    str(python_path),
                    "-m",
                    "attack_detection.training.codet5p_infer",
                    "--model-dir",
                    str(MODEL_DIR),
                ],
                input_text=json.dumps({"content": content, "language": language}),
                cwd=str(BACKEND_DIR),
                timeout=120,
                cancel_event=cancel_event,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._failed(started, f"CodeT5+ 220M inference process failed: {exc}")
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "unknown inference error")[-800:]
            return self._failed(started, reason)
        try:
            output = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return self._failed(started, "CodeT5+ 220M inference returned invalid JSON")
        if output.get("status") != "completed":
            return self._unavailable(started, str(output.get("reason") or "task/language is unavailable"))
        return self._shape_completed(output, started)

    def scan_batch(
        self, requests: list[dict[str, str]], cancel_event: object | None = None,
    ) -> list[dict[str, Any]]:
        """Run project candidates in one process so CodeT5+ weights load once."""

        if not requests:
            return []
        started = time.perf_counter()
        registry_path = MODEL_DIR / "codet5p_registry.json"
        python_path = configured_deep_python()
        if not registry_path.is_file():
            return [self._unavailable(started, "CodeT5+ 220M registry is missing") for _ in requests]
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [self._unavailable(started, f"CodeT5+ 220M registry cannot be read: {exc}") for _ in requests]
        if not registry.get("active_routes"):
            return [
                self._unavailable(started, "CodeT5+ 220M has no strict-gated active version")
                for _ in requests
            ]
        if not python_path.is_file():
            return [
                self._unavailable(started, f"deep-learning interpreter is unavailable: {python_path}")
                for _ in requests
            ]
        try:
            completed = _run_persistent_batch(
                python_path,
                requests,
                timeout=(
                    PROJECT_SINGLE_INFERENCE_TIMEOUT_SECONDS
                    if len(requests) == 1
                    else PROJECT_INFERENCE_TIMEOUT_SECONDS
                ),
                cancel_event=cancel_event,
            )
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            return [self._failed(started, f"CodeT5+ 220M batch inference failed: {exc}") for _ in requests]
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "unknown inference error")[-800:]
            return [self._failed(started, reason) for _ in requests]
        try:
            outputs = json.loads(completed.stdout).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            return [self._failed(started, "CodeT5+ 220M batch inference returned invalid JSON") for _ in requests]
        results = []
        for index in range(len(requests)):
            if index >= len(outputs):
                results.append(self._failed(started, "CodeT5+ 220M batch response is incomplete"))
                continue
            output = outputs[index]
            if output.get("status") != "completed":
                results.append(self._unavailable(
                    started, str(output.get("reason") or "task/language is unavailable"),
                ))
            else:
                results.append(self._shape_completed(output, started))
        return results

    def _shape_completed(self, output: dict[str, Any], started: float) -> dict[str, Any]:
        if output.get("primary_task") != "malicious_intent":
            return self._unavailable(started, "CodeT5+ 220M did not return the active malicious-code task")
        task_probabilities = output.get("task_probabilities") or {}
        task_thresholds = output.get("task_thresholds") or {}
        task_trained_thresholds = output.get("task_trained_thresholds") or {}
        task_versions = output.get("task_versions") or {}
        return EngineResult(
            name=self.name,
            status="completed",
            decision=str(output["decision"]),
            probability=float(output["probability"]),
            threshold=float(output["threshold"]),
            model_version=str(output["model_version"]),
            duration_ms=int(output.get("duration_ms") or ((time.perf_counter() - started) * 1000)),
            metadata={
                "primary_task": "malicious_intent",
                "task_probabilities": {
                    "malicious_intent": task_probabilities.get("malicious_intent"),
                },
                "task_thresholds": {
                    "malicious_intent": task_thresholds.get("malicious_intent"),
                },
                "trained_threshold": output.get("trained_threshold"),
                "task_trained_thresholds": {
                    "malicious_intent": task_trained_thresholds.get("malicious_intent"),
                },
                "task_versions": {
                    "malicious_intent": task_versions.get("malicious_intent"),
                },
            },
        ).to_dict()

    def _unavailable(self, started: float, reason: str) -> dict[str, Any]:
        return EngineResult(
            name=self.name,
            status="unavailable",
            reason=reason,
            duration_ms=int((time.perf_counter() - started) * 1000),
        ).to_dict()

    def _failed(self, started: float, reason: str) -> dict[str, Any]:
        return EngineResult(
            name=self.name,
            status="failed",
            error=reason,
            duration_ms=int((time.perf_counter() - started) * 1000),
        ).to_dict()
