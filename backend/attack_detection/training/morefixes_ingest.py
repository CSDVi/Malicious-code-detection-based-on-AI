"""Convert MoreFixes git patches into leakage-resistant Go/PHP fix pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LANGUAGE_EXTENSIONS = {".go": "go", ".php": "php"}
COMMIT_SUFFIX = re.compile(r"_(?P<commit>[0-9a-f]{40})$")
NON_PRODUCTION_PART = re.compile(
    r"^(?:tests?|testing|testdata|specs?|examples?|demos?|benchmarks?|fixtures?|"
    r"vendor|third[_-]?party|node_modules|docs?|documentation|mocks?|generated)$",
    re.IGNORECASE,
)


def ingest(
    patch_dir: Path, output_path: Path, manifest_path: Path,
    maximum_patch_bytes: int = 64 * 1024 * 1024,
    maximum_code_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pairs: set[str] = set()
    seen_code: set[str] = set()
    counts: Counter[Any] = Counter()
    repositories: dict[str, set[str]] = defaultdict(set)
    written: list[dict[str, Any]] = []

    patch_files = sorted(patch_dir.glob("*.patch"))
    for index, patch_path in enumerate(patch_files, 1):
        counts["patch_files_scanned"] += 1
        if patch_path.stat().st_size > maximum_patch_bytes:
            counts["oversized_patches_skipped"] += 1
            continue
        raw = patch_path.read_bytes()
        if not _may_contain_target(raw):
            continue
        text = raw.decode("utf-8", errors="replace")
        repo, commit = _identity_from_name(patch_path.stem)
        split = _repository_split(repo)
        patch_sha256 = hashlib.sha256(raw).hexdigest()
        pairs = parse_patch(text, maximum_code_bytes=maximum_code_bytes)
        if pairs:
            counts["target_patch_files"] += 1
        for pair in pairs:
            pair_digest = hashlib.sha256(
                (pair["bad_code"] + "\0" + pair["good_code"]).encode("utf-8")
            ).hexdigest()
            bad_hash = hashlib.sha256(pair["bad_code"].encode("utf-8")).hexdigest()
            good_hash = hashlib.sha256(pair["good_code"].encode("utf-8")).hexdigest()
            if pair_digest in seen_pairs or bad_hash in seen_code or good_hash in seen_code:
                counts["duplicate_pairs_skipped"] += 1
                continue
            seen_pairs.add(pair_digest)
            seen_code.update((bad_hash, good_hash))
            language = pair["language"]
            pair_id = f"morefixes:{repo}:{commit}:{pair_digest[:16]}"
            source_url = _source_url(repo, commit)
            common = {
                "language": language,
                "cwe": "",
                "source": "morefixes_v4",
                "package_name": repo,
                "license": "source-project-dependent",
                "family": f"morefixes-repo:{repo}",
                "pair_id": pair_id,
                "commit": commit,
                "split": split,
                "artifact_sha256": patch_sha256,
                "source_url": source_url,
                "file_path": pair["file_path"],
                "label_basis": "CVE-associated MoreFixes security patch; unified-diff before/after reconstruction",
                "behavior_labels": [],
                "cwe_labels": [],
                "label_confidence": 0.85,
                "review_status": "external_cve_patch",
                "review_notes": "Only code hunk context is available; commit metadata is not included in the patch-only release.",
                "line_labels": [],
                "label_scopes": [],
            }
            bad = _record(
                common, pair["bad_code"], bad_hash, "vulnerable",
                "real_world_vulnerability", "bad", good_hash,
            )
            good = _record(
                common, pair["good_code"], good_hash, "benign",
                "security_patch", "good", bad_hash,
            )
            written.extend((bad, good))
            counts[(language, split, "pairs")] += 1
            repositories[language].add(repo)
        if index % 5000 == 0:
            print(json.dumps({
                "progress": index, "total": len(patch_files),
                "pairs": len(written) // 2,
            }), flush=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in written:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "schema_version": 1,
        "source": "MoreFixes v4 patch release 2026-06-20",
        "patch_directory": str(patch_dir.resolve()),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "maximum_patch_bytes": maximum_patch_bytes,
        "maximum_code_bytes": maximum_code_bytes,
        "patch_files_scanned": counts["patch_files_scanned"],
        "target_patch_files": counts["target_patch_files"],
        "oversized_patches_skipped": counts["oversized_patches_skipped"],
        "duplicate_pairs_skipped": counts["duplicate_pairs_skipped"],
        "samples": len(written),
        "pairs": len(written) // 2,
        "pairs_by_language_and_split": {
            language: {
                split: counts[(language, split, "pairs")]
                for split in ("train", "validation", "test")
            }
            for language in ("go", "php")
        },
        "repositories_by_language": {
            language: len(values) for language, values in repositories.items()
        },
        "split_policy": "SHA-256(repository) 70/15/15; every repository belongs to one split",
        "repository_isolation_verified": _verify_repository_isolation(written),
        "limitations": [
            "Patch-only release provides changed hunks rather than complete source files.",
            "CWE and MoreFixes confidence score require the optional database dump and are not inferred here.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def parse_patch(text: str, maximum_code_bytes: int = 64 * 1024) -> list[dict[str, str]]:
    blocks = re.split(r"(?m)^diff --git ", text)[1:]
    output = []
    for block in blocks:
        lines = block.splitlines()
        old_path = _header_path(lines, "--- ")
        new_path = _header_path(lines, "+++ ")
        file_path = new_path if new_path != "/dev/null" else old_path
        language = LANGUAGE_EXTENSIONS.get(Path(file_path).suffix.lower())
        if language is None or old_path == "/dev/null" or new_path == "/dev/null":
            continue
        if not _is_production_source(file_path, language):
            continue
        bad_parts, good_parts = [], []
        for hunk in _hunks(lines):
            bad, good, removed, added = _reconstruct_hunk(hunk)
            if not (removed or added):
                continue
            if "".join(bad.split()) == "".join(good.split()):
                continue
            bad_parts.append(bad)
            good_parts.append(good)
        if not bad_parts:
            continue
        bad_code = _limit("\n\n".join(bad_parts), maximum_code_bytes)
        good_code = _limit("\n\n".join(good_parts), maximum_code_bytes)
        if min(len(bad_code.strip()), len(good_code.strip())) < 16 or bad_code == good_code:
            continue
        output.append({
            "language": language,
            "file_path": file_path,
            "bad_code": bad_code,
            "good_code": good_code,
        })
    return output


def _hunks(lines: list[str]) -> Iterable[list[str]]:
    current = None
    for line in lines:
        if line.startswith("@@ "):
            if current is not None:
                yield current
            current = []
        elif current is not None:
            if line.startswith("diff --git "):
                break
            current.append(line)
    if current is not None:
        yield current


def _reconstruct_hunk(lines: list[str]) -> tuple[str, str, int, int]:
    bad, good = [], []
    removed = added = 0
    for line in lines:
        if line == "\\ No newline at end of file":
            continue
        if line.startswith("-"):
            bad.append(line[1:])
            removed += 1
        elif line.startswith("+"):
            good.append(line[1:])
            added += 1
        elif line.startswith(" "):
            bad.append(line[1:])
            good.append(line[1:])
        else:
            # Empty context lines lose their leading space through splitlines only
            # when malformed patches are encountered; retain them symmetrically.
            bad.append(line)
            good.append(line)
    return "\n".join(bad), "\n".join(good), removed, added


def _header_path(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix):].split("\t", 1)[0].strip().strip('"')
            if value.startswith(("a/", "b/")):
                value = value[2:]
            return value
    return ""


def _may_contain_target(raw: bytes) -> bool:
    return bool(re.search(rb"(?m)^\+\+\+ b/.*\.(?:go|php)(?:\t|\r?$)", raw, re.IGNORECASE))


def _is_production_source(file_path: str, language: str) -> bool:
    normalized = file_path.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if any(NON_PRODUCTION_PART.fullmatch(part) for part in parts[:-1]):
        return False
    filename = parts[-1].lower() if parts else ""
    if language == "go" and filename.endswith("_test.go"):
        return False
    if language == "php" and re.search(r"(^|[_.-])tests?([_.-]|$)", filename):
        return False
    return True


def _identity_from_name(stem: str) -> tuple[str, str]:
    match = COMMIT_SUFFIX.search(stem)
    if not match:
        return stem, "unknown"
    return stem[:match.start()], match.group("commit")


def _source_url(repo: str, commit: str) -> str:
    if repo.startswith("github.com_"):
        owner_repo = repo[len("github.com_"):]
        if "_" in owner_repo:
            owner, name = owner_repo.split("_", 1)
            return f"https://github.com/{owner}/{name}/commit/{commit}"
    return ""


def _repository_split(repo: str) -> str:
    bucket = int(hashlib.sha256(repo.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else ("validation" if bucket < 85 else "test")


def _record(
    common: dict[str, Any], code: str, sample_hash: str, label: str,
    category: str, version: str, parent_hash: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "normalized_code": code,
        "label": label,
        "category": category,
        **common,
        "version": version,
        "sample_hash": sample_hash,
        "paired_version": "good" if version == "bad" else "bad",
        "parent_sample_hash": parent_hash,
    }


def _limit(code: str, maximum_bytes: int) -> str:
    raw = code.encode("utf-8", errors="replace")
    return raw[:maximum_bytes].decode("utf-8", errors="ignore")


def _verify_repository_isolation(records: list[dict[str, Any]]) -> bool:
    seen: dict[str, set[str]] = defaultdict(set)
    for record in records:
        seen[str(record["package_name"])].add(str(record["split"]))
    leaking = [repo for repo, splits in seen.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"MoreFixes repository leakage: {leaking[:5]}")
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Go/PHP fix pairs from MoreFixes patches")
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--maximum-patch-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--maximum-code-bytes", type=int, default=64 * 1024)
    args = parser.parse_args()
    print(json.dumps(ingest(
        args.patch_dir, args.output, args.manifest,
        maximum_patch_bytes=args.maximum_patch_bytes,
        maximum_code_bytes=args.maximum_code_bytes,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
