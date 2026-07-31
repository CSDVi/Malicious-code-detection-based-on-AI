"""Add independent official-project shell scripts as train-only negatives.

The Bash route has far fewer benign files than malicious BashBench rows, and
its frozen test split contains build, CI, packaging, Android cross-compile,
and service-management scripts that are absent from training. This augmenter
uses only unrelated official repositories already present in practicesets,
never copies held-out rows into train, and never executes any script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_xgb_practiceset_expansion import (
    MAX_SOURCE_BYTES,
    MIN_CODE_CHARS,
    _decode_source,
    _make_row,
    _sha256_bytes,
)


REPOSITORIES = {
    "containerd-main": (
        "go/containerd-main",
        "https://github.com/containerd/containerd",
    ),
    "coredns-master": (
        "go/coredns-master",
        "https://github.com/coredns/coredns",
    ),
    "grafana-main": (
        "go/grafana-main",
        "https://github.com/grafana/grafana",
    ),
    "kubernetes-master": (
        "go/kubernetes-master",
        "https://github.com/kubernetes/kubernetes",
    ),
    "moby-master": (
        "go/moby-master",
        "https://github.com/moby/moby",
    ),
    "terraform-main": (
        "go/terraform-main",
        "https://github.com/hashicorp/terraform",
    ),
    "openssl-master": (
        "BashBenign/openssl/openssl-master",
        "https://github.com/openssl/openssl",
    ),
    "libffi-master": (
        "BashBenign/libffi/libffi-master",
        "https://github.com/libffi/libffi",
    ),
    "curl-master": (
        "BashBenign/curl/curl-master",
        "https://github.com/curl/curl",
    ),
    "ffmpeg-master": (
        "BashBenign/ffmpeg/FFmpeg-master",
        "https://github.com/FFmpeg/FFmpeg",
    ),
    "docker-compose-main": (
        "BashBenign/docker-compose/compose-main",
        "https://github.com/docker/compose",
    ),
}
EXTENSIONS = {".sh", ".bash"}
MAX_FILES_PER_REPOSITORY = 200


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def augment(
    base: Path,
    source_root: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    base_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    with base.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            base_rows.append(row)
            digest = str(
                row.get("sample_hash") or row.get("artifact_sha256") or ""
            )
            if digest:
                seen_hashes.add(digest)
                hash_splits[digest].add(str(row.get("split") or ""))
            family = str(row.get("family") or "")
            if family:
                family_splits[(
                    str(row.get("source") or ""),
                    family,
                )].add(str(row.get("split") or ""))

    additions: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for repository, (relative_root, source_url) in REPOSITORIES.items():
        repository_root = source_root / relative_root
        candidates: list[dict[str, Any]] = []
        for path in repository_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            counts[(repository, "files_seen")] += 1
            try:
                raw = path.read_bytes()
            except OSError:
                counts[(repository, "read_error")] += 1
                continue
            if len(raw) > MAX_SOURCE_BYTES:
                counts[(repository, "oversize")] += 1
                continue
            code = _decode_source(raw).replace("\x00", "").strip()
            if len(code) < MIN_CODE_CHARS:
                counts[(repository, "short_or_binary")] += 1
                continue
            relative = path.relative_to(repository_root).as_posix()
            row = _make_row(
                code=code,
                label="benign",
                language="bash",
                family=f"github_official_bash:{repository}",
                source="github_official_bash_hardnegatives",
                file_path=f"{repository}/{relative}",
                source_url=source_url,
                label_basis=(
                    "Shell source from an independent official project; "
                    "train-only hard negative for build/CI/package scripts."
                ),
                category="benign_build_ci_shell",
                confidence=0.95,
                review_status="source_verified",
                behavior_labels=("official_project_shell_hardnegative",),
                artifact_sha256=_sha256_bytes(raw),
            )
            if row is None:
                counts[(repository, "normalization_rejected")] += 1
                continue
            row["split"] = "train"
            candidates.append(row)

        selected = sorted(
            candidates,
            key=lambda row: (
                str(row["sample_hash"]),
                str(row["file_path"]),
            ),
        )[:MAX_FILES_PER_REPOSITORY]
        for row in selected:
            digest = str(row["sample_hash"])
            if digest in seen_hashes:
                counts[(repository, "base_or_cross_repo_duplicate")] += 1
                continue
            seen_hashes.add(digest)
            additions.append(row)
            counts[(repository, "rows_added")] += 1
            family_splits[(
                str(row["source"]),
                str(row["family"]),
            )].add("train")
            hash_splits[digest].add("train")

    family_leaks = [
        {
            "source": source,
            "family": family,
            "splits": sorted(splits),
        }
        for (source, family), splits in family_splits.items()
        if len(splits) > 1
    ]
    hash_leaks = [
        {"sample_hash": digest, "splits": sorted(splits)}
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    ]
    if family_leaks or hash_leaks:
        raise RuntimeError("Bash hard-negative augmentation would leak splits")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in base_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in sorted(
            additions,
            key=lambda value: (
                str(value["family"]),
                str(value["sample_hash"]),
            ),
        ):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)

    report = {
        "schema_version": 1,
        "objective": "train-only official Bash hard-negative augmentation",
        "offline_text_only": True,
        "samples_executed_or_compiled": False,
        "base_dataset": str(base.resolve()),
        "base_dataset_sha256": _sha256_file(base),
        "base_rows": len(base_rows),
        "output_dataset": str(output.resolve()),
        "output_dataset_sha256": _sha256_file(output),
        "output_rows": len(base_rows) + len(additions),
        "rows_added": len(additions),
        "families_added": len({
            str(row["family"]) for row in additions
        }),
        "split": "train",
        "label": "benign",
        "per_repository_counts": [
            {
                "repository": repository,
                **{
                    metric: value
                    for (name, metric), value in sorted(counts.items())
                    if name == repository
                },
            }
            for repository in REPOSITORIES
        ],
        "family_split_isolation_verified": True,
        "hash_split_isolation_verified": True,
        "family_split_leaks": [],
        "hash_split_leaks": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        augment(
            args.base_dataset.resolve(),
            args.source_root.resolve(),
            args.output_dataset.resolve(),
            args.report.resolve(),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
