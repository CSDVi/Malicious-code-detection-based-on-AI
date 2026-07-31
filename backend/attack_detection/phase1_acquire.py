"""Acquire official clean versions paired with local Datadog samples.

Only registry metadata and distribution archives are downloaded. Package code
is never imported, installed, or executed.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .data_pipeline import SafeDownloader, _version_key, sha256_file, utc_now
from .practiceset_layout import resolve_practiceset_layout


MAX_TOTAL_BYTES = 6 * 1024 * 1024 * 1024
MAX_PAIR_BYTES = 100 * 1024 * 1024


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _npm_name(path_segment: str) -> str:
    if path_segment.startswith("@") and path_segment.count("@") >= 2:
        scope, name = path_segment[1:].split("@", 1)
        return f"@{scope}/{name}"
    return path_segment


def _load_affected(path: Path) -> dict[str, set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(name): {str(version) for version in versions}
        for name, versions in raw.items()
        if isinstance(versions, list)
    }


def discover_local_compromised(archives_root: Path) -> dict[str, dict[str, set[str]]]:
    layout = resolve_practiceset_layout(archives_root)
    packages: dict[str, dict[str, set[str]]] = {
        "npm": defaultdict(set),
        "pypi": defaultdict(set),
    }
    npm_urls = layout.javascript / "npm" / "npm_lightweight_urls.txt"
    for line in npm_urls.read_text(encoding="utf-8").splitlines():
        if "/samples/npm/compromised_lib/" not in line:
            continue
        tail = urllib.parse.unquote(line.split("/samples/npm/compromised_lib/", 1)[1])
        parts = tail.split("/")
        if len(parts) >= 3:
            packages["npm"][_npm_name(parts[0])].add(parts[1])

    pypi_root = layout.python / "compromised_lib"
    for archive in pypi_root.rglob("*.zip"):
        relative = archive.relative_to(pypi_root)
        if len(relative.parts) >= 3:
            packages["pypi"][relative.parts[0]].add(relative.parts[1])
    return packages


def _choose_prior(versions: Iterable[str], affected: set[str], malicious_versions: set[str]) -> str:
    bad = affected | malicious_versions
    candidates = [str(value) for value in versions if str(value) not in bad]
    if not candidates:
        return ""
    first_bad = min(malicious_versions, key=_version_key)
    prior = [value for value in candidates if _version_key(value) < _version_key(first_bad)]
    return max(prior, key=_version_key) if prior else ""


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "package"


def _pair_metadata(
    ecosystem: str,
    package_name: str,
    bad_versions: set[str],
    affected: set[str],
) -> dict[str, str] | None:
    if ecosystem == "npm":
        encoded = urllib.parse.quote(package_name, safe="")
        metadata_url = f"https://registry.npmjs.org/{encoded}"
        metadata = SafeDownloader.fetch_json(metadata_url)
        clean_version = _choose_prior(metadata.get("versions", {}).keys(), affected, bad_versions)
        version_data = metadata.get("versions", {}).get(clean_version, {})
        archive_url = str(version_data.get("dist", {}).get("tarball") or "")
        published_at = str(metadata.get("time", {}).get(clean_version) or "")
        license_name = str(version_data.get("license") or metadata.get("license") or "")
    else:
        encoded = urllib.parse.quote(package_name, safe="")
        metadata_url = f"https://pypi.org/pypi/{encoded}/json"
        metadata = SafeDownloader.fetch_json(metadata_url)
        clean_version = _choose_prior(metadata.get("releases", {}).keys(), affected, bad_versions)
        release_files = metadata.get("releases", {}).get(clean_version, [])
        artifact = next((item for item in release_files if item.get("packagetype") == "sdist"), None)
        if artifact is None and release_files:
            artifact = release_files[0]
        artifact = artifact or {}
        archive_url = str(artifact.get("url") or "")
        published_at = str(artifact.get("upload_time_iso_8601") or "")
        license_name = str(metadata.get("info", {}).get("license") or "")
    if not clean_version or not archive_url:
        return None
    return {
        "ecosystem": ecosystem,
        "package_name": package_name,
        "malicious_versions": sorted(bad_versions, key=_version_key),
        "affected_versions": sorted(affected, key=_version_key),
        "clean_version": clean_version,
        "metadata_url": metadata_url,
        "archive_url": archive_url,
        "published_at": published_at,
        "license": license_name,
    }


def acquire_pairs(archives_root: Path, metadata_root: Path) -> dict[str, object]:
    archives_root = archives_root.resolve()
    layout = resolve_practiceset_layout(archives_root)
    output_root = layout.other
    metadata_root = metadata_root.resolve()
    downloader = SafeDownloader(output_root)
    local = discover_local_compromised(archives_root)
    affected = {
        "npm": _load_affected(metadata_root / "datadog_npm_manifest.json"),
        "pypi": _load_affected(metadata_root / "datadog_pypi_manifest.json"),
    }
    pairs: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    current_total = _directory_bytes(output_root / "paired_clean_archives")
    for ecosystem in ("npm", "pypi"):
        for package_name, bad_versions in sorted(local[ecosystem].items()):
            try:
                item = _pair_metadata(ecosystem, package_name, bad_versions, affected[ecosystem].get(package_name, set()))
                if item is None:
                    errors.append({"ecosystem": ecosystem, "package_name": package_name, "error": "no prior clean release"})
                    continue
                archive_url = str(item["archive_url"])
                url_path = Path(urllib.parse.urlparse(archive_url).path)
                suffix = ".tar.gz" if archive_url.endswith(".tar.gz") else (".tgz" if archive_url.endswith(".tgz") else url_path.suffix)
                target = output_root / "paired_clean_archives" / ecosystem / _safe_name(package_name) / f"{item['clean_version']}{suffix}"
                previous_size = target.stat().st_size if target.exists() else 0
                if not target.exists() and current_total + MAX_PAIR_BYTES > MAX_TOTAL_BYTES:
                    raise ValueError("6 GiB dataset directory budget would be exceeded")
                record = downloader.download(archive_url, target, f"{ecosystem}_official_clean_version", MAX_PAIR_BYTES)
                current_total += int(record["size"]) - previous_size
                item.update({
                    "local_path": str(target),
                    "archive_bytes": record["size"],
                    "archive_sha256": record["sha256"],
                })
                pairs.append(item)
            except Exception as exc:  # Keep a complete per-package acquisition report.
                errors.append({"ecosystem": ecosystem, "package_name": package_name, "error": str(exc)})

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "executed_samples": False,
        "practicesets_root": str(archives_root),
        "archives_root": str(output_root),
        "total_directory_bytes": current_total,
        "budget_bytes": MAX_TOTAL_BYTES,
        "requested_packages": sum(len(items) for items in local.values()),
        "paired_packages": len(pairs),
        "errors": errors,
        "pairs": pairs,
    }
    manifest_path = output_root / "paired_clean_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official clean versions paired with local Datadog samples")
    parser.add_argument("archives_root", type=Path)
    parser.add_argument("metadata_root", type=Path)
    args = parser.parse_args()
    result = acquire_pairs(args.archives_root, args.metadata_root)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
