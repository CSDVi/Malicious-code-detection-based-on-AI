"""Shared cooperative cancellation helpers for project scans."""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import Any


class ScanCancelled(RuntimeError):
    """Raised when a project scan has been explicitly cancelled."""


def cancellation_requested(cancel_event: object | None) -> bool:
    checker = getattr(cancel_event, "is_set", None)
    return bool(checker and checker())


def raise_if_cancelled(cancel_event: object | None) -> None:
    if cancellation_requested(cancel_event):
        raise ScanCancelled("project scan cancelled")


def run_cancellable(
    command: Sequence[str],
    *,
    input_text: str,
    cwd: str,
    timeout: float,
    cancel_event: object | None = None,
    poll_interval: float = 0.1,
) -> subprocess.CompletedProcess[str]:
    """Run a model subprocess and terminate it promptly when cancellation is requested."""

    if cancel_event is None:
        return subprocess.run(
            list(command),
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=cwd,
            timeout=timeout,
            check=False,
        )

    raise_if_cancelled(cancel_event)
    with tempfile.TemporaryFile(mode="w+b") as stdin_file:
        stdin_file.write(input_text.encode("utf-8"))
        stdin_file.flush()
        stdin_file.seek(0)
        process = subprocess.Popen(
            list(command),
            stdin=stdin_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            cwd=cwd,
        )
        deadline = time.monotonic() + max(0.1, float(timeout))
        while True:
            if cancellation_requested(cancel_event):
                _terminate(process)
                raise ScanCancelled("project scan cancelled during model inference")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                raise subprocess.TimeoutExpired(list(command), timeout)
            try:
                stdout_bytes, stderr_bytes = process.communicate(
                    timeout=min(max(0.02, poll_interval), remaining),
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                return subprocess.CompletedProcess(
                    list(command), process.returncode, stdout, stderr,
                )
            except subprocess.TimeoutExpired:
                continue


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
