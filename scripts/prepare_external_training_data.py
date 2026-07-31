"""Normalize downloaded vulnerability/malware sources into project JSONL.

This script is intentionally conservative:
* source files are read as data and never executed;
* vulnerability labels come from the upstream vulnerable/patch flag;
* Juliet C# rows are selected by the upstream ``_bad``/``_good`` filename;
* deterministic family splits keep repository/benchmark siblings together;
* external rows are deduplicated against the current reviewed dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable


LANGUAGE_MAP = {
    "c": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "go": "go",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "php": "php",
    "python": "python",
    "ruby": "ruby",
}
MAX_CODE_CHARS = 30000


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _split(family: str) -> str:
    # Keep each repository/Juliet case family in one split.
    bucket = int(_hash(family)[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _code(value: str) -> str:
    value = value.replace("\x00", "").strip()
    return value[:MAX_CODE_CHARS]


def _row(
    *,
    code: str,
    label: str,
    language: str,
    family: str,
    category: str,
    cwe: str,
    source: str,
    file_path: str,
    source_url: str,
    pair_id: str = "",
    label_basis: str = "",
) -> dict:
    code = _code(code)
    sample_hash = _hash(code)
    return {
        "code": code,
        "normalized_code": code,
        "label": label,
        "category": category or ("vulnerable_code" if label == "vulnerable" else "secure_patch"),
        "language": language,
        "cwe": cwe,
        "source": source,
        "package_name": family,
        "version": "",
        "license": "",
        "sample_hash": sample_hash,
        "family": family,
        "published_at": "",
        "split": _split(family),
        "artifact_sha256": sample_hash,
        "source_url": source_url,
        "file_path": file_path,
        "paired_version": "",
        "label_basis": label_basis,
        "behavior_labels": [],
        "cwe_labels": [cwe] if cwe else [],
        "label_confidence": 0.95,
        "review_status": "source_verified",
        "parent_sample_hash": "",
        "pair_id": pair_id,
        "pair_slot": "vulnerable" if label == "vulnerable" else "patch",
        "review_notes": "Imported from a public research benchmark; source code was not executed.",
        "line_labels": [],
        "label_scopes": ["vulnerability_risk"] if label in {"vulnerable", "benign"} else ["malicious_intent"],
    }


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def iter_zenodo_csv(path: Path) -> Iterable[dict]:
    language = LANGUAGE_MAP.get(path.stem.removeprefix("data_").lower())
    if not language:
        return
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, item in enumerate(reader):
            code = item.get("vul_code") or item.get("code") or ""
            if not code.strip():
                continue
            vulnerable = _truthy(item.get("is_vulnerable", ""))
            repo = item.get("repo_url") or item.get("repo_owner") or f"{path.stem}:{index}"
            family = f"zenodo:{language}:{repo}"
            cwe = item.get("cwe_id") or ""
            yield _row(
                code=code,
                label="vulnerable" if vulnerable else "benign",
                language=language,
                family=family,
                category=item.get("cwe_name") or "",
                cwe=cwe,
                source="zenodo_13870382",
                file_path=item.get("file_name") or "",
                source_url="https://zenodo.org/records/13870382",
                pair_id=f"{language}:{item.get('index') or index}",
                label_basis="CVE vulnerable/patch pair",
            )
            patch = item.get("patch") or ""
            if vulnerable and patch.strip() and _code(patch) != _code(code):
                yield _row(
                    code=patch,
                    label="benign",
                    language=language,
                    family=family,
                    category="secure_patch",
                    cwe=cwe,
                    source="zenodo_13870382",
                    file_path=f"{item.get('file_name') or ''}#patched",
                    source_url="https://zenodo.org/records/13870382",
                    pair_id=f"{language}:{item.get('index') or index}",
                    label_basis="CVE fixing code paired with the vulnerable function",
                )


def iter_juliet_csharp(path: Path, max_each: int = 4000) -> Iterable[dict]:
    counters = {"vulnerable": 0, "benign": 0}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if not normalized.lower().endswith(".cs"):
                continue
            lowered = normalized.lower()
            if lowered.endswith("_bad.cs"):
                label = "vulnerable"
            elif re.search(r"_good(?:b2g|g2b)?\.cs$", lowered):
                label = "benign"
            else:
                continue
            if counters[label] >= max_each:
                continue
            try:
                code = archive.read(name).decode("utf-8", errors="replace")
            except (KeyError, RuntimeError, zipfile.BadZipFile):
                continue
            cwe_match = re.search(r"/(cwe\d+)_", lowered)
            cwe = cwe_match.group(1).upper() if cwe_match else ""
            case_family = normalized.rsplit("/", 1)[0]
            family = f"juliet_csharp:{case_family}"
            row = _row(
                code=code,
                label=label,
                language="csharp",
                family=family,
                category=cwe or "juliet_csharp",
                cwe=cwe,
                source="nist_juliet_csharp_1.3",
                file_path=normalized,
                source_url="https://samate.nist.gov/SARD/test-suites",
                pair_id=case_family,
                label_basis="Juliet _bad/_good case label",
            )
            counters[label] += 1
            yield row


def iter_bash_malware(path: Path) -> Iterable[dict]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            lowered = name.lower()
            if not lowered.endswith("_beautified.zip"):
                continue
            try:
                nested_bytes = archive.read(name)
            except (KeyError, RuntimeError, zipfile.BadZipFile):
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                    for nested_name in nested.namelist():
                        if not nested_name.lower().endswith((".sh", ".bash", ".txt")):
                            continue
                        try:
                            # The repository ships the nested samples with the
                            # conventional analysis password ``infected``.
                            code = nested.read(
                                nested_name,
                                pwd=b"infected",
                            ).decode("utf-8", errors="replace")
                        except (KeyError, RuntimeError, zipfile.BadZipFile):
                            continue
                        family_name = name.split("/")[1] if "/" in name else Path(name).stem
                        family = f"bash_malware:{family_name}"
                        yield _row(
                            code=code,
                            label="malicious",
                            language="bash",
                            family=family,
                            category="malware_source",
                            cwe="",
                            source="0xjet_bash_malware",
                            file_path=f"{name}!/{nested_name}",
                            source_url="https://github.com/0xjet/bash-malware",
                            pair_id=family,
                            label_basis="public malware source repository",
                        )
            except zipfile.BadZipFile:
                continue


def _load_existing(path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(value)
            code = str(value.get("code") or "")
            if code:
                hashes.add(_hash(code))
    return rows, hashes


def prepare(
    base_dataset: Path,
    output_dataset: Path,
    external_dir: Path,
    include_juliet: bool,
    include_bash: bool,
) -> dict:
    rows, existing_hashes = _load_existing(base_dataset)
    external: list[dict] = []
    seen = set(existing_hashes)
    for path in sorted(external_dir.glob("data_*.csv")):
        for row in iter_zenodo_csv(path):
            if not row["code"] or _hash(row["code"]) in seen:
                continue
            seen.add(_hash(row["code"]))
            external.append(row)
    if include_juliet:
        juliet = external_dir / "juliet_csharp_v1.3.zip"
        if juliet.is_file():
            for row in iter_juliet_csharp(juliet):
                if _hash(row["code"]) in seen:
                    continue
                seen.add(_hash(row["code"]))
                external.append(row)
    if include_bash:
        bash = external_dir / "bash-malware-main.zip"
        if bash.is_file():
            for row in iter_bash_malware(bash):
                if _hash(row["code"]) in seen:
                    continue
                seen.add(_hash(row["code"]))
                external.append(row)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with output_dataset.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in external:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "base_rows": len(rows),
        "external_rows_added": len(external),
        "output_rows": len(rows) + len(external),
        "by_language": {},
        "by_source": {},
    }
    for row in external:
        summary["by_language"][row["language"]] = summary["by_language"].get(row["language"], 0) + 1
        summary["by_source"][row["source"]] = summary["by_source"].get(row["source"], 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--external-dir", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--no-juliet", action="store_true")
    parser.add_argument("--no-bash", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.base_dataset,
                args.output_dataset,
                args.external_dir,
                include_juliet=not args.no_juliet,
                include_bash=not args.no_bash,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
