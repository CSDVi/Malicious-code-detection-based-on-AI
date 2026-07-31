"""Plan and verify a bounded download from the ASE 2023 PyPI malware corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = "lxyeternal/pypi_malregistry"
SOURCE_URL = "https://github.com/lxyeternal/pypi_malregistry"


def create_plan(
    tree_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    curl_config_path: str | Path,
    limit: int = 2500,
    budget_bytes: int = 200 * 1024 * 1024,
) -> dict[str, Any]:
    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    by_package: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in tree.get("tree", []):
        path = PurePosixPath(str(item.get("path") or ""))
        if item.get("type") != "blob" or not path.name.lower().endswith((".tar.gz", ".tgz", ".zip")):
            continue
        if len(path.parts) < 3 or ".." in path.parts or int(item.get("size") or 0) < 500:
            continue
        by_package[path.parts[0]].append(item)
    candidates = []
    for package_name, items in by_package.items():
        bounded = [item for item in items if int(item.get("size") or 0) <= 4 * 1024 * 1024]
        if not bounded:
            continue
        candidates.append(max(bounded, key=lambda item: (int(item.get("size") or 0), str(item["path"]))))
    candidates.sort(key=lambda item: hashlib.sha256(PurePosixPath(item["path"]).parts[0].encode()).hexdigest())

    root = Path(output_root).resolve()
    selected = []
    total = 0
    for item in candidates:
        size = int(item.get("size") or 0)
        if len(selected) >= limit:
            break
        if total + size > budget_bytes:
            continue
        relative = PurePosixPath(str(item["path"]))
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded_path = "/".join(urllib.parse.quote(part, safe="@._-+") for part in relative.parts)
        selected.append({
            "package_name": relative.parts[0],
            "version": relative.parts[1],
            "repository_path": relative.as_posix(),
            "size": size,
            "git_blob_sha1": str(item["sha"]),
            "download_url": f"https://raw.githubusercontent.com/{REPOSITORY}/main/{encoded_path}",
            "local_path": str(destination),
            "status": "planned",
        })
        total += size
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "ASE 2023 pypi_malregistry",
        "source_url": SOURCE_URL,
        "repository": REPOSITORY,
        "tree_sha": tree.get("sha"),
        "tree_truncated": bool(tree.get("truncated")),
        "selection_policy": "one largest bounded archive per package, package order by SHA-256",
        "limit": limit,
        "budget_bytes": budget_bytes,
        "selected_packages": len(selected),
        "planned_bytes": total,
        "executed_samples": False,
        "license_note": "repository has no aggregate license; individual package licenses must be retained",
        "items": selected,
    }
    _write_json(Path(manifest_path), manifest)
    _write_curl_config(Path(curl_config_path), selected)
    return {key: value for key, value in manifest.items() if key != "items"}


def verify_downloads(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    counts = defaultdict(int)
    verified_bytes = 0
    for item in manifest.get("items", []):
        local = Path(str(item["local_path"]))
        if not local.is_file():
            item["status"] = "missing"
        elif local.stat().st_size != int(item["size"]):
            item["status"] = "size_mismatch"
        elif _git_blob_sha1(local) != item["git_blob_sha1"]:
            item["status"] = "hash_mismatch"
        else:
            item["status"] = "verified"
            item["sha256"] = _sha256(local)
            verified_bytes += local.stat().st_size
        counts[item["status"]] += 1
    manifest["verified_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["verification"] = dict(counts)
    manifest["verified_bytes"] = verified_bytes
    _write_json(path, manifest)
    return {"selected": len(manifest.get("items", [])), "verified_bytes": verified_bytes, **counts}


def create_retry_config(manifest_path: str | Path, curl_config_path: str | Path) -> dict[str, int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = [item for item in manifest.get("items", []) if item.get("status") != "verified"]
    _write_curl_config(Path(curl_config_path), items)
    return {"retry_items": len(items)}


def _git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _curl_path(value: str) -> str:
    return value.replace("\\", "/").replace('"', '\\"')


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_curl_config(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, item in enumerate(items):
            if index:
                stream.write("next\n")
            stream.write(f'url = "{item["download_url"]}"\n')
            stream.write(f'output = "{_curl_path(item["local_path"])}"\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire a verified subset of pypi_malregistry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--tree", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--curl-config", required=True)
    plan.add_argument("--limit", type=int, default=2500)
    plan.add_argument("--budget-bytes", type=int, default=200 * 1024 * 1024)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    retry = subparsers.add_parser("retry-config")
    retry.add_argument("--manifest", required=True)
    retry.add_argument("--curl-config", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = create_plan(args.tree, args.output_root, args.manifest, args.curl_config, args.limit, args.budget_bytes)
    elif args.command == "verify":
        result = verify_downloads(args.manifest)
    else:
        result = create_retry_config(args.manifest, args.curl_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
