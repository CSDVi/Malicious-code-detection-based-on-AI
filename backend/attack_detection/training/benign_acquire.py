"""Acquire a bounded, hash-verified set of popular PyPI source distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PYPI_JSON = "https://pypi.org/pypi/{name}/json"
ALLOWED_ARCHIVE_HOST = "files.pythonhosted.org"


def acquire(
    ranking_path: str | Path,
    malicious_dataset_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    limit: int = 750,
    max_archive_bytes: int = 4 * 1024 * 1024,
    budget_bytes: int = 200 * 1024 * 1024,
) -> dict[str, Any]:
    ranking = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    excluded = _malicious_packages(malicious_dataset_path)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    names = [name for name in _ranking_names(ranking) if _normalize(name) not in excluded][: max(limit * 3, limit)]
    candidates: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_metadata_candidate, name, max_archive_bytes, budget_bytes): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                candidate = future.result()
                if candidate is not None:
                    candidates[name] = candidate
            except Exception as exc:
                errors.append({"package_name": name, "stage": "metadata", "error": str(exc)[:300]})

    planned = []
    planned_bytes = 0
    for name in names:
        item = candidates.get(name)
        if item is None or len(planned) >= limit:
            continue
        if planned_bytes + int(item["size"]) > budget_bytes:
            continue
        planned.append(item)
        planned_bytes += int(item["size"])

    def download(item: dict[str, Any]) -> dict[str, Any]:
        destination = root / str(item["normalized_name"]) / _safe_filename(str(item["filename"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        _download_verified(str(item["url"]), destination, str(item["sha256"]), int(item["size"]))
        return item | {"local_path": str(destination), "status": "verified"}

    verified_by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(download, item): str(item["package_name"]) for item in planned}
        for future in as_completed(futures):
            name = futures[future]
            try:
                verified_by_name[name] = future.result()
            except Exception as exc:
                errors.append({"package_name": name, "stage": "download", "error": str(exc)[:300]})
    for item in planned:
        verified = verified_by_name.get(str(item["package_name"]))
        if verified is not None:
            items.append(verified)
    downloaded = sum(int(item["size"]) for item in items)
    # Static extraction is a separate bounded step; packages are never installed or imported.
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "PyPI official JSON API and files.pythonhosted.org source distributions",
        "ranking_source": str(Path(ranking_path).resolve()),
        "selection_policy": "popular packages excluding known malicious families; latest bounded sdist only",
        "label_scope": "negative candidate for malicious intent only; not asserted vulnerability-free",
        "limit": limit,
        "budget_bytes": budget_bytes,
        "downloaded_bytes": downloaded,
        "verified_archives": len(items),
        "excluded_known_malicious_packages": len(excluded),
        "errors": errors,
        "items": items,
        "executed_samples": False,
    }
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in manifest.items() if key not in {"items", "errors"}} | {"errors": len(errors)}


def _metadata_candidate(name: str, max_archive_bytes: int, budget_bytes: int) -> dict[str, Any] | None:
    metadata = _get_json(PYPI_JSON.format(name=urllib.parse.quote(name, safe="")))
    item = _select_sdist(metadata, max_archive_bytes, budget_bytes)
    if item is None:
        return None
    return item | {
        "package_name": str(metadata["info"]["name"]),
        "normalized_name": _normalize(str(metadata["info"]["name"])),
        "version": str(metadata["info"]["version"]),
        "license": str(metadata["info"].get("license") or "")[:500],
        "project_url": str(metadata["info"].get("project_url") or ""),
    }


def _ranking_names(value: Any) -> list[str]:
    rows = value.get("rows", value) if isinstance(value, dict) else value
    output = []
    for row in rows if isinstance(rows, list) else []:
        name = row.get("project") if isinstance(row, dict) else None
        if name:
            output.append(str(name))
    return output


def _malicious_packages(dataset_path: str | Path) -> set[str]:
    output = set()
    with Path(dataset_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("label") == "malicious" and row.get("package_name"):
                output.add(_normalize(str(row["package_name"])))
    return output


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Code-Security-Research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _select_sdist(metadata: dict[str, Any], max_bytes: int, remaining: int) -> dict[str, Any] | None:
    version = str(metadata.get("info", {}).get("version") or "")
    releases = metadata.get("releases", {}).get(version, [])
    choices = []
    for file in releases:
        size = int(file.get("size") or 0)
        url = str(file.get("url") or "")
        digest = str(file.get("digests", {}).get("sha256") or "")
        upload_time = str(file.get("upload_time_iso_8601") or file.get("upload_time") or "")
        try:
            uploaded = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        if uploaded > datetime.now(timezone.utc) - timedelta(days=30):
            continue
        if file.get("packagetype") != "sdist" or not digest or not (500 <= size <= min(max_bytes, remaining)):
            continue
        if urllib.parse.urlparse(url).hostname != ALLOWED_ARCHIVE_HOST:
            continue
        choices.append({
            "filename": str(file.get("filename") or "source.tar.gz"),
            "url": url,
            "sha256": digest,
            "size": size,
            "upload_time": upload_time,
        })
    return min(choices, key=lambda item: (int(item["size"]), str(item["filename"]))) if choices else None


def _download_verified(url: str, destination: Path, expected_hash: str, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size and _sha256(destination) == expected_hash:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Code-Security-Research/1.0"})
    digest = hashlib.sha256()
    size = 0
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > expected_size:
                raise ValueError("download exceeded declared size")
            digest.update(block)
            stream.write(block)
    if size != expected_size or digest.hexdigest() != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError("download size or SHA-256 mismatch")
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", Path(name).name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download verified popular PyPI source distributions")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--malicious-dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=750)
    parser.add_argument("--max-archive-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--budget-bytes", type=int, default=200 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(acquire(
        args.ranking, args.malicious_dataset, args.output_root, args.manifest,
        args.limit, args.max_archive_bytes, args.budget_bytes,
    ), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
