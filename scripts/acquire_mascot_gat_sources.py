"""Acquire a bounded, source-only MASCOT subset for GATv2 language routes.

The downloader never extracts or executes repository contents. It streams
GitHub source ZIPs, rejects oversized archives, reads only selected text
extensions in memory, removes exact duplicate files, and assigns whole
repositories to one deterministic split.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any


MASCOT_TO_CANONICAL = {
    "C": "c",
    "C++": "cpp",
    "C#": "csharp",
    "Shell": "bash",
    "PowerShell": "powershell",
    "Batchfile": "batch",
    "Go": "go",
    "HTML": "html",
}
EXTENSIONS = {
    "c": {".c", ".h"},
    "cpp": {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".h"},
    "csharp": {".cs"},
    "bash": {".sh", ".bash", ".zsh"},
    "powershell": {".ps1", ".psm1", ".psd1"},
    "batch": {".bat", ".cmd"},
    "go": {".go"},
    "html": {".html", ".htm", ".xhtml", ".hta"},
}
IGNORED_PARTS = {
    ".git", ".github", "bin", "obj", "node_modules", "packages", "third_party",
    "vendor", "vendors", "build", "dist",
}
MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024
MAX_FILES_PER_PROJECT = 48
MAX_CODE_CHARS = 30_000
USER_AGENT = "XiezhiCodeGuard-Research/1.0"


def acquire(
    inventory_path: Path,
    output_path: Path,
    report_path: Path,
    languages: list[str],
    target_projects: int,
    candidates_per_language: int,
    workers: int,
) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    by_language = inventory.get("byLanguage") or {}
    requested = [value.strip().lower() for value in languages if value.strip()]
    reverse = {canonical: mascot for mascot, canonical in MASCOT_TO_CANONICAL.items()}
    unknown = sorted(set(requested) - set(reverse))
    if unknown:
        raise ValueError(f"unsupported requested languages: {', '.join(unknown)}")
    jobs: list[tuple[str, dict[str, Any]]] = []
    for language in requested:
        candidates = list(by_language.get(reverse[language]) or [])
        candidates.sort(key=lambda row: _sha(str(row.get("url") or "")))
        jobs.extend((language, row) for row in candidates[:candidates_per_language])

    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_fetch_project, language, metadata): (language, metadata)
            for language, metadata in jobs
        }
        completed = 0
        for future in as_completed(futures):
            language, _ = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # keep one failed repository from stopping acquisition
                failures[f"{language}:{type(exc).__name__}"] += 1
                continue
            if not result["files"]:
                failures[f"{language}:{result['reason']}"] += 1
            elif len(accepted[language]) < target_projects:
                accepted[language].append(result)
            if completed % 25 == 0:
                progress = {key: len(value) for key, value in sorted(accepted.items())}
                print(json.dumps({"completed": completed, "accepted": progress}), flush=True)

    rows: list[dict[str, Any]] = []
    code_hashes: set[str] = set()
    repository_counts: Counter[tuple[str, str]] = Counter()
    file_counts: Counter[tuple[str, str]] = Counter()
    for language in requested:
        projects = sorted(accepted[language], key=lambda item: _sha(item["url"]))
        unique_projects = []
        project_fingerprints: set[str] = set()
        for project in projects:
            aggregate = _sha("|".join(sorted(item["sha256"] for item in project["files"])))
            if aggregate in project_fingerprints:
                failures[f"{language}:duplicate_project"] += 1
                continue
            project_fingerprints.add(aggregate)
            unique_projects.append(project)
        projects = unique_projects
        if len(projects) < 40:
            raise RuntimeError(
                f"{language} acquired {len(projects)} usable projects; 40 are required "
                "for train/validation/test minimums"
            )
        split_by_url = _stratified_splits(projects)
        for project in projects:
            split = split_by_url[project["url"]]
            family = f"mascot:{project['slug']}"
            added = 0
            for item in project["files"]:
                if item["sha256"] in code_hashes:
                    continue
                code_hashes.add(item["sha256"])
                rows.append({
                    "code": item["code"],
                    "normalized_code": item["code"],
                    "label": "malicious",
                    "category": str(project.get("keyword") or "malware_source"),
                    "language": language,
                    "cwe": "",
                    "source": "mascot_human_reviewed",
                    "package_name": project["project_name"],
                    "version": "source",
                    "sample_hash": item["sha256"],
                    "family": family,
                    "split": split,
                    "source_url": project["url"],
                    "file_path": item["path"],
                    "pair_id": family,
                    "label_basis": "MASCOT human-reviewed malware source repository",
                    "label_confidence": 0.95,
                    "review_status": "source_verified",
                    "label_scopes": ["malicious_intent"],
                })
                added += 1
            if added:
                repository_counts[(language, split)] += 1
                file_counts[(language, split)] += added

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(rows, key=lambda value: (
            value["language"], value["split"], value["family"], value["file_path"],
        )):
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "schema_version": 1,
        "source_dataset": "Bojing94/MASCOT",
        "source_url": "https://huggingface.co/datasets/Bojing94/MASCOT",
        "inventory": str(inventory_path.resolve()),
        "output": str(output_path.resolve()),
        "safety": {
            "source_only": True,
            "executed": False,
            "archive_max_bytes": MAX_ARCHIVE_BYTES,
            "uncompressed_max_bytes": MAX_UNCOMPRESSED_BYTES,
            "member_max_bytes": MAX_MEMBER_BYTES,
        },
        "repositories": _nested_counts(repository_counts),
        "files": _nested_counts(file_counts),
        "rows": len(rows),
        "failures": dict(sorted(failures.items())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _fetch_project(language: str, metadata: dict[str, Any]) -> dict[str, Any]:
    url = str(metadata.get("url") or "").strip().rstrip("/")
    match = re.fullmatch(r"https?://github\.com/([^/]+)/([^/#?]+)", url, re.IGNORECASE)
    if not match:
        return {"files": [], "reason": "invalid_url"}
    owner, repository = match.groups()
    repository = repository.removesuffix(".git")
    archive_url = f"https://codeload.github.com/{owner}/{repository}/zip/HEAD"
    request = urllib.request.Request(archive_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > MAX_ARCHIVE_BYTES:
                return {"files": [], "reason": "archive_too_large"}
            buffer = io.BytesIO()
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                if buffer.tell() + len(block) > MAX_ARCHIVE_BYTES:
                    return {"files": [], "reason": "archive_too_large"}
                buffer.write(block)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {"files": [], "reason": "download_failed"}
    try:
        archive = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    except zipfile.BadZipFile:
        return {"files": [], "reason": "invalid_zip"}
    members = archive.infolist()
    if len(members) > 20_000:
        return {"files": [], "reason": "too_many_members"}
    if sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES:
        return {"files": [], "reason": "expanded_size_limit"}
    files = []
    for member in sorted(members, key=lambda value: value.filename):
        if member.is_dir() or member.file_size < 32 or member.file_size > MAX_MEMBER_BYTES:
            continue
        parts = PurePosixPath(member.filename).parts
        if any(part.lower() in IGNORED_PARTS for part in parts):
            continue
        suffix = PurePosixPath(member.filename).suffix.lower()
        if suffix not in EXTENSIONS[language]:
            continue
        if member.compress_size and member.file_size / member.compress_size > 200:
            continue
        try:
            raw = archive.read(member)
        except (KeyError, RuntimeError, zipfile.BadZipFile):
            continue
        code = _decode_source(raw)
        if len(code.strip()) < 32:
            continue
        files.append({
            "path": "/".join(parts[1:]) if len(parts) > 1 else parts[0],
            "code": code[:MAX_CODE_CHARS],
            "sha256": _sha(code[:MAX_CODE_CHARS]),
        })
        if len(files) >= MAX_FILES_PER_PROJECT:
            break
    return {
        "files": files,
        "reason": "ok" if files else "no_matching_source",
        "url": url,
        "slug": f"{owner.lower()}/{repository.lower()}",
        "project_name": str(metadata.get("projectName") or repository),
        "keyword": str(metadata.get("keyword") or "malware_source"),
    }


def _decode_source(raw: bytes) -> str:
    if b"\x00" in raw[:4096]:
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(encoding).replace("\x00", "").strip()
            except UnicodeDecodeError:
                pass
        return ""
    return raw.decode("utf-8", errors="replace").replace("\x00", "").strip()


def _stratified_splits(projects: list[dict[str, Any]]) -> dict[str, str]:
    count = len(projects)
    validation = max(10, count // 5)
    test = max(10, count // 5)
    if count - validation - test < 20:
        raise RuntimeError(f"not enough projects for strict splits: {count}")
    ordered = sorted(projects, key=lambda item: _sha("split:" + item["url"]))
    output = {}
    for index, project in enumerate(ordered):
        if index < validation:
            split = "validation"
        elif index < validation + test:
            split = "test"
        else:
            split = "train"
        output[project["url"]] = split
    return output


def _nested_counts(values: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for (language, split), count in sorted(values.items()):
        output[language][split] = count
    return dict(output)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--languages", nargs="+", default=["c", "cpp", "csharp", "bash"])
    parser.add_argument("--target-projects", type=int, default=65)
    parser.add_argument("--candidates-per-language", type=int, default=130)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = acquire(
        args.inventory.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        args.languages,
        args.target_projects,
        args.candidates_per_language,
        args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
