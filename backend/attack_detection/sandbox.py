"""Opt-in remote sandbox adapter.

This adapter never invokes a local process. It requires an explicitly configured
remote sandbox endpoint and an opt-in flag before a sample is submitted.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .contracts import EngineResult


class SandboxEngine:
    name = "isolated_sandbox"

    def scan(self, filename: str, payload: bytes, sha256: str) -> dict[str, Any]:
        endpoint = os.environ.get("XIEZHI_SANDBOX_URL", "").strip().rstrip("/")
        if not endpoint:
            return EngineResult(
                name=self.name, status="unavailable", reason="sandbox backend not configured",
                metadata={"execution": "none", "safety": "upload was not executed"},
            ).to_dict()
        if os.environ.get("XIEZHI_SANDBOX_AUTO_SCAN", "0").strip().lower() not in {"1", "true", "yes"}:
            return EngineResult(
                name=self.name, status="skipped", reason="sandbox submission requires XIEZHI_SANDBOX_AUTO_SCAN=1",
                metadata={"execution": "none", "safety": "upload was not executed"},
            ).to_dict()
        max_bytes = int(os.environ.get("XIEZHI_SANDBOX_MAX_BYTES", str(10 * 1024 * 1024)))
        if len(payload) > max_bytes:
            return EngineResult(name=self.name, status="skipped", reason=f"sample exceeds sandbox limit ({max_bytes} bytes)").to_dict()
        body = json.dumps({
            "filename": filename, "sha256": sha256,
            "content_base64": base64.b64encode(payload).decode("ascii"),
            "network": "isolated", "destroy_after_run": True,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "XiezhiCodeGuard/1.0"}
        token = os.environ.get("XIEZHI_SANDBOX_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(endpoint + "/v1/samples", data=body, headers=headers, method="POST")
        timeout = max(3.0, min(60.0, float(os.environ.get("XIEZHI_SANDBOX_TIMEOUT", "15"))))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read(1024 * 1024).decode("utf-8", errors="replace"))
            sample_id = data.get("id") or data.get("sample_id")
            if sample_id and os.environ.get("XIEZHI_SANDBOX_POLL", "0").strip().lower() in {"1", "true", "yes"}:
                data = self._poll(endpoint, str(sample_id), headers, timeout, data)
            return EngineResult(
                name=self.name, status="completed", decision="unknown",
                metadata={
                    "execution": "remote_isolated_only", "sample_id": sample_id,
                    "status": data.get("status", "submitted"), "events": data.get("events", []),
                    "network": "isolated", "destroy_after_run": True,
                },
            ).to_dict()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return EngineResult(name=self.name, status="failed", reason=f"sandbox request failed: {exc}", metadata={"execution": "remote_isolated_only"}).to_dict()

    def _poll(self, endpoint: str, sample_id: str, headers: dict[str, str], timeout: float, initial: dict[str, Any]) -> dict[str, Any]:
        """Optionally collect a bounded result from a compatible remote worker."""
        deadline = time.monotonic() + max(1.0, min(60.0, float(os.environ.get("XIEZHI_SANDBOX_POLL_SECONDS", "10"))))
        data = initial
        while time.monotonic() < deadline:
            status = str(data.get("status") or data.get("state") or "").lower()
            if status in {"completed", "finished", "failed", "error", "terminated"}:
                return data
            time.sleep(min(1.0, max(0.1, deadline - time.monotonic())))
            request = urllib.request.Request(endpoint + "/v1/samples/" + sample_id, headers=headers, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read(1024 * 1024).decode("utf-8", errors="replace"))
        return data
