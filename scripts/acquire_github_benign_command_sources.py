"""Acquire bounded, popular benign PowerShell or Batch projects for GATv2.

Repository source ZIPs are read in memory through the same limits as the
MASCOT acquisition pipeline. Nothing is extracted or executed.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from acquire_mascot_gat_sources import (
    MAX_ARCHIVE_BYTES,
    MAX_MEMBER_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    USER_AGENT,
    _fetch_project,
    _sha,
    _stratified_splits,
)

RISK_TERMS = {
    "malware", "ransomware", "exploit", "payload", "powersploit", "mimikatz",
    "pentest", "redteam", "red-team", "offensive", "phishing", "credential",
    "backdoor", "trojan", "rootkit", "botnet", "rat ", "c2 ", "bypass uac",
}


def acquire(
    output_path: Path,
    report_path: Path,
    *,
    target_projects: int,
    candidate_projects: int,
    workers: int,
    language: str,
) -> dict[str, Any]:
    repositories = _search_repositories(candidate_projects, language)
    candidates = [
        repository for repository in repositories
        if not _looks_security_offensive(repository)
    ][:candidate_projects]
    accepted: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _fetch_project,
                language,
                {
                    "url": repository["html_url"],
                    "projectName": repository["name"],
                    "keyword": "popular_benign_software",
                },
            ): repository
            for repository in candidates
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                failures[type(exc).__name__] += 1
                continue
            if result["files"]:
                if len(accepted) < target_projects:
                    result["stars"] = int(
                        futures[future].get("stargazers_count") or 0
                    )
                    accepted.append(result)
            else:
                failures[str(result["reason"])] += 1
            if completed % 25 == 0:
                print(json.dumps({
                    "completed": completed,
                    "accepted": len(accepted),
                }), flush=True)

    unique_projects = []
    fingerprints: set[str] = set()
    for project in sorted(accepted, key=lambda item: _sha(item["url"])):
        fingerprint = _sha(
            "|".join(sorted(item["sha256"] for item in project["files"]))
        )
        if fingerprint in fingerprints:
            failures["duplicate_project"] += 1
            continue
        fingerprints.add(fingerprint)
        unique_projects.append(project)
    if len(unique_projects) < 40:
        raise RuntimeError(
            f"Only {len(unique_projects)} benign {language} projects were "
            "acquired; 40 are required."
        )

    split_by_url = _stratified_splits(unique_projects)
    rows = []
    split_counts: Counter[str] = Counter()
    file_hashes: set[str] = set()
    for project in unique_projects:
        split = split_by_url[project["url"]]
        family = f"github_benign:{project['slug']}"
        added = 0
        for item in project["files"]:
            if item["sha256"] in file_hashes:
                continue
            file_hashes.add(item["sha256"])
            rows.append({
                "code": item["code"],
                "normalized_code": item["code"],
                "label": "benign",
                "category": "popular_open_source_software",
                "language": language,
                "cwe": "",
                "source": "github_popular_benign_source",
                "package_name": project["project_name"],
                "version": "source",
                "sample_hash": item["sha256"],
                "family": family,
                "split": split,
                "source_url": project["url"],
                "file_path": item["path"],
                "pair_id": family,
                "label_basis": (
                    "Popular non-offensive open-source repository selected "
                    "by GitHub language search"
                ),
                "label_confidence": 0.9,
                "review_status": "source_verified",
                "label_scopes": ["malicious_intent"],
            })
            added += 1
        if added:
            split_counts[split] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(
            rows,
            key=lambda value: (
                value["split"], value["family"], value["file_path"],
            ),
        ):
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    report = {
        "schema_version": 1,
        "source": "GitHub repository search API",
        "query": (
            f"language:{'PowerShell' if language == 'powershell' else 'Batchfile'} "
            "stars:>20 archived:false fork:false"
        ),
        "selection": (
            "Popular repositories with offensive-security terms removed from "
            "name, description, and topics."
        ),
        "output": str(output_path.resolve()),
        "repositories": dict(sorted(split_counts.items())),
        "files": len(rows),
        "safety": {
            "source_only": True,
            "executed": False,
            "archive_max_bytes": MAX_ARCHIVE_BYTES,
            "uncompressed_max_bytes": MAX_UNCOMPRESSED_BYTES,
            "member_max_bytes": MAX_MEMBER_BYTES,
        },
        "failures": dict(sorted(failures.items())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _search_repositories(limit: int, language: str) -> list[dict[str, Any]]:
    output = []
    pages = min(3, max(1, (limit + 99) // 100))
    for page in range(1, pages + 1):
        github_language = "PowerShell" if language == "powershell" else "Batchfile"
        query = urllib.parse.quote(
            f"language:{github_language} stars:>20 archived:false fork:false"
        )
        url = (
            "https://api.github.com/search/repositories"
            f"?q={query}&sort=stars&order=desc&per_page=100&page={page}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request, timeout=35) as response:
            payload = json.load(response)
        output.extend(payload.get("items") or [])
    return output[:limit]


def _looks_security_offensive(repository: dict[str, Any]) -> bool:
    value = " ".join([
        str(repository.get("name") or ""),
        str(repository.get("description") or ""),
        " ".join(str(item) for item in (repository.get("topics") or [])),
    ]).lower()
    return any(term in value for term in RISK_TERMS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-projects", type=int, default=65)
    parser.add_argument("--candidate-projects", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--language",
        choices=("powershell", "batch"),
        default="powershell",
    )
    args = parser.parse_args()
    print(json.dumps(
        acquire(
            args.output.resolve(),
            args.report.resolve(),
            target_projects=args.target_projects,
            candidate_projects=args.candidate_projects,
            workers=args.workers,
            language=args.language,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
