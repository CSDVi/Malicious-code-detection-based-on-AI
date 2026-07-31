"""Optional hash-only reputation lookups.

The default provider is disabled. No source content is uploaded by this module.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .contracts import EngineResult

_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


class HashReputationEngine:
    name = "hash_reputation"

    def scan(self, sha256: str) -> dict[str, Any]:
        provider = os.environ.get("XIEZHI_REPUTATION_PROVIDER", "none").strip().lower()
        api_key = os.environ.get("XIEZHI_VT_API_KEY", "").strip()
        if provider in {"", "none", "disabled"}:
            return EngineResult(
                name=self.name, status="unavailable", reason="external hash reputation is disabled; set XIEZHI_REPUTATION_PROVIDER and an API key",
                metadata={"provider": None, "privacy": "only SHA256 would be queried; source content is not uploaded"},
            ).to_dict()
        if provider != "virustotal":
            return EngineResult(name=self.name, status="unavailable", reason=f"unsupported reputation provider: {provider}").to_dict()
        if not api_key:
            return EngineResult(name=self.name, status="unavailable", reason="XIEZHI_VT_API_KEY is not configured", metadata={"provider": provider}).to_dict()
        cache_key = (provider, sha256.lower())
        cached = _CACHE.get(cache_key)
        if cached and cached[0] > time.time():
            return cached[1]
        request = urllib.request.Request(
            f"https://www.virustotal.com/api/v3/files/{sha256.lower()}",
            headers={"X-Apikey": api_key, "Accept": "application/json", "User-Agent": "XiezhiCodeGuard/1.0"},
            method="GET",
        )
        timeout = max(2.0, min(20.0, float(os.environ.get("XIEZHI_REPUTATION_TIMEOUT", "8"))))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read(1024 * 1024).decode("utf-8", errors="replace"))
            stats = (((payload.get("data") or {}).get("attributes") or {}).get("last_analysis_stats") or {})
            malicious = int(stats.get("malicious") or 0)
            suspicious = int(stats.get("suspicious") or 0)
            total = sum(int(value or 0) for value in stats.values())
            result = EngineResult(
                name=self.name,
                status="completed",
                decision="unknown",
                risk_score=min(80, malicious * 10 + suspicious * 3),
                metadata={
                    "provider": "VirusTotal", "sha256": sha256.lower(), "malicious": malicious,
                    "suspicious": suspicious, "total_engines": total,
                    "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "privacy": "仅查询 SHA256；本次请求未上传源码或文件内容",
                },
            ).to_dict()
        except urllib.error.HTTPError as exc:
            reason = "hash not found" if exc.code == 404 else f"provider HTTP {exc.code}"
            result = EngineResult(name=self.name, status="unavailable", reason=reason, metadata={"provider": "VirusTotal"}).to_dict()
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            result = EngineResult(name=self.name, status="failed", reason=f"reputation lookup failed: {exc}").to_dict()
        _CACHE[cache_key] = (time.time() + 300, result)
        return result
