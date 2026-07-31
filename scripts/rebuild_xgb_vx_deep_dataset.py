"""Rebuild the VX C/C++ slice with a full, family-isolated static scan.

The scanner never executes or compiles a sample.  It reads bounded source text
from plain files and from .7z/.zip members, requires file-local malicious
behavior evidence, deduplicates by full-source SHA-256, caps correlated files
inside one malware family, and assigns each canonical family to exactly one
train/validation/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from build_xgb_incoming_multilingual_dataset import (
    MAX_ARCHIVE_BYTES,
    MAX_SOURCE_BYTES,
    MIN_CODE_CHARS,
    VX_IMPLEMENTATION_SUFFIXES,
    VX_SIGNAL_PATTERNS,
    _bounded_code,
    _is_vendored,
    _row,
    _safe_member,
    _sha256_text,
    _vx_signal_score,
)


SOURCE = "vx_underground_malware_source"
LANGUAGES = ("c", "cpp")
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMPLEMENTATION_SUFFIXES = {
    suffix: language
    for suffix, language in VX_IMPLEMENTATION_SUFFIXES.items()
    if language in LANGUAGES
}
GENERIC_FAMILY_PARTS = {
    "android",
    "backdoor",
    "backdoors",
    "bootkit",
    "botnet",
    "botnets",
    "code",
    "dos",
    "freebsd",
    "families",
    "infector",
    "infectors",
    "linux",
    "mac",
    "macos",
    "malware",
    "malwarefamily",
    "malwarefamilies",
    "other",
    "rootkit",
    "rootkits",
    "source",
    "tool",
    "tools",
    "trojan",
    "trojans",
    "virus",
    "viruses",
    "win32",
    "win64",
    "windows",
    "worm",
    "worms",
}
VARIANT_TOKEN = re.compile(
    r"^(?:[a-z]|v?\d+(?:[a-z]?\d*)?|src|source|code|final|leak|master)$",
    re.IGNORECASE,
)
EXCERPT_MARKER = "\n/* ... BEHAVIOR-PRESERVING TRAINING EXCERPT ... */\n"


def _canonical_tokens(value: str) -> list[str]:
    tokens = [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", value)
        if token
    ]
    output = [
        token
        for token in tokens
        if token not in GENERIC_FAMILY_PARTS
    ]
    while len(output) > 1 and VARIANT_TOKEN.fullmatch(output[-1]):
        output.pop()
    return output


def _comment_mask(code: str) -> str:
    """Blank C/C++ comments without changing character offsets."""

    def blank(match: re.Match[str]) -> str:
        return "".join(
            "\n" if character == "\n" else " "
            for character in match.group(0)
        )

    masked = re.sub(r"/\*.*?\*/", blank, code, flags=re.DOTALL)
    return re.sub(r"(?m)(?<!:)//[^\r\n]*", blank, masked)


def _behavior_preserving_code(
    code: str,
    max_chars: int,
) -> tuple[str, bool]:
    """Bound long source while retaining each detected behavior group."""

    if len(code) <= max_chars:
        return code, False
    executable = _comment_mask(code).lower()
    positions = [
        match.start()
        for pattern in VX_SIGNAL_PATTERNS
        if (match := re.search(pattern, executable))
    ]
    if not positions:
        return _bounded_code(code, max_chars)

    marker_budget = len(EXCERPT_MARKER) * (len(positions) + 1)
    content_budget = max(1_000, max_chars - marker_budget)
    edge_chars = content_budget // 5
    behavior_chars = max(
        160,
        (content_budget - (2 * edge_chars)) // len(positions),
    )
    ranges = [
        (0, edge_chars),
        (max(0, len(code) - edge_chars), len(code)),
    ]
    for position in positions:
        start = max(0, position - behavior_chars // 2)
        end = min(len(code), start + behavior_chars)
        start = max(0, end - behavior_chars)
        ranges.append((start, end))
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    excerpt = EXCERPT_MARKER.join(
        code[start:end]
        for start, end in merged
    )
    if len(excerpt) > max_chars:
        # Conservative marker budgeting normally prevents this.  Keep the
        # invariant explicit if the marker or allocation changes later.
        excerpt = excerpt[:max_chars]
    return excerpt, True


def _canonical_archive_family(relative_archive: str) -> str:
    path = PurePosixPath(relative_archive.replace("\\", "/"))
    for parent in reversed(path.parts[:-1]):
        lowered = parent.lower()
        if lowered.endswith("-family") or lowered.endswith("_family"):
            family = re.sub(r"[-_]family$", "", parent, flags=re.IGNORECASE)
            tokens = _canonical_tokens(family)
            if tokens:
                return "vx_family:" + "-".join(tokens)
    for parent in reversed(path.parts[:-1]):
        tokens = _canonical_tokens(parent)
        if tokens and not all(token in GENERIC_FAMILY_PARTS for token in tokens):
            return "vx_family:" + "-".join(tokens)
    tokens = _canonical_tokens(path.stem)
    if tokens:
        return "vx_family:" + "-".join(tokens)
    return "vx_family:" + hashlib.sha256(
        relative_archive.encode("utf-8", errors="ignore")
    ).hexdigest()[:16]


def _plain_family(relative_path: str) -> str:
    path = PurePosixPath(relative_path.replace("\\", "/"))
    parent = path.parent.as_posix()
    tokens = _canonical_tokens(parent)
    if not tokens:
        tokens = _canonical_tokens(path.stem)
    return "vx_plain_family:" + (
        "-".join(tokens)
        if tokens
        else hashlib.sha256(relative_path.encode()).hexdigest()[:16]
    )


def _member_candidates(
    names: Iterable[str],
    counts: Counter[str],
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for raw_name in names:
        name = str(raw_name).replace("\\", "/")
        language = IMPLEMENTATION_SUFFIXES.get(PurePosixPath(name).suffix.lower())
        if not language:
            continue
        counts["archive_raw_implementation_members"] += 1
        if not _safe_member(name):
            counts["unsafe_archive_implementation_members"] += 1
            continue
        if _is_vendored(name):
            counts["vendored_archive_implementation_members"] += 1
            continue
        output.append((name, language))
    return sorted(
        set(output),
        key=lambda item: (
            item[1],
            hashlib.sha256(item[0].encode("utf-8", errors="ignore")).hexdigest(),
        ),
    )


def _decode_source(raw: bytes) -> str:
    if len(raw) < MIN_CODE_CHARS or len(raw) > MAX_SOURCE_BYTES:
        return ""
    if raw.count(b"\x00") > max(2, len(raw) // 100):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


class DeepCollector:
    def __init__(
        self,
        *,
        base_hash_labels: dict[str, set[str]],
        max_code_chars: int,
    ) -> None:
        self.base_hash_labels = base_hash_labels
        self.max_code_chars = max_code_chars
        self.rows_by_family_language: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        self.seen_hashes: set[str] = set()
        self.counts: Counter[str] = Counter()
        self.archives_by_family: dict[str, set[str]] = defaultdict(set)

    def consider(
        self,
        *,
        raw: bytes,
        language: str,
        family: str,
        file_path: str,
        archive_path: str,
    ) -> None:
        self.counts["members_read"] += 1
        text = _decode_source(raw).replace("\x00", "").strip()
        if len(text) < MIN_CODE_CHARS:
            self.counts["empty_short_large_or_binary"] += 1
            return
        digest = _sha256_text(text)
        if digest in self.seen_hashes:
            self.counts["deep_duplicate"] += 1
            return
        base_labels = self.base_hash_labels.get(digest, set())
        if base_labels:
            if "benign" in base_labels:
                self.counts["base_label_conflict"] += 1
            else:
                self.counts["base_duplicate"] += 1
            return
        score = _vx_signal_score(text)
        if score < 2:
            self.counts["no_file_local_signal"] += 1
            return
        bounded, truncated = _behavior_preserving_code(
            text,
            self.max_code_chars,
        )
        if truncated:
            self.counts["behavior_preserving_excerpt"] += 1
        row = _row(
            code=bounded,
            label="malicious",
            language=language,
            family=family,
            source=SOURCE,
            file_path=file_path,
            source_url=(
                "https://github.com/vxunderground/MalwareSourceCode"
                f"#{archive_path}"
            ),
            label_basis=(
                "VX malware-family source archive plus at least two "
                "file-local executable behavior signal groups"
            ),
            category="malicious_native_source",
            split="pending",
            behavior_labels=(f"file_local_signal_groups:{score}",),
            max_code_chars=self.max_code_chars,
        )
        if row is None:
            self.counts["empty_short_large_or_binary"] += 1
            return
        # _row hashes the bounded representation. Preserve the full source
        # digest so truncation cannot hide duplicates.
        row["sample_hash"] = digest
        row["artifact_sha256"] = digest
        row["original_code_length"] = len(text)
        row["training_truncated"] = truncated
        row["_signal_score"] = score
        self.seen_hashes.add(digest)
        self.rows_by_family_language[(family, language)].append(row)
        self.archives_by_family[family].add(archive_path)
        self.counts["file_local_candidates"] += 1


def _scan_plain(vx_root: Path, collector: DeepCollector) -> None:
    for path in sorted(vx_root.rglob("*")):
        if not path.is_file():
            continue
        language = IMPLEMENTATION_SUFFIXES.get(path.suffix.lower())
        if not language:
            continue
        relative = path.relative_to(vx_root).as_posix()
        if _is_vendored(relative):
            collector.counts["vendored_plain"] += 1
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            collector.counts["plain_read_error"] += 1
            continue
        collector.counts["plain_implementation_members"] += 1
        collector.consider(
            raw=raw,
            language=language,
            family=_plain_family(relative),
            file_path=relative,
            archive_path=relative,
        )


def _scan_zip(
    archive: Path,
    relative_archive: str,
    collector: DeepCollector,
) -> None:
    family = _canonical_archive_family(relative_archive)
    with zipfile.ZipFile(archive) as bundle:
        candidates = _member_candidates(bundle.namelist(), collector.counts)
        collector.counts["archive_candidate_members"] += len(candidates)
        collector.counts["archives_with_candidates"] += bool(candidates)
        for name, language in candidates:
            try:
                info = bundle.getinfo(name)
                if info.file_size > MAX_SOURCE_BYTES:
                    collector.counts["oversize_member"] += 1
                    continue
                try:
                    raw = bundle.read(info, pwd=b"infected")
                except RuntimeError:
                    raw = bundle.read(info)
            except (KeyError, OSError, RuntimeError, NotImplementedError):
                collector.counts["zip_member_read_error"] += 1
                continue
            collector.consider(
                raw=raw,
                language=language,
                family=family,
                file_path=f"{relative_archive}#{name}",
                archive_path=relative_archive,
            )


def _scan_7z(
    archive: Path,
    relative_archive: str,
    collector: DeepCollector,
) -> None:
    import py7zr
    from py7zr.io import BytesIOFactory

    family = _canonical_archive_family(relative_archive)
    with py7zr.SevenZipFile(archive, mode="r", password="infected") as bundle:
        archive_entries = bundle.list()
        directory_names = [
            str(info.filename)
            for info in archive_entries
            if info.is_directory
            and IMPLEMENTATION_SUFFIXES.get(
                PurePosixPath(str(info.filename)).suffix.lower()
            )
        ]
        collector.counts["archive_source_suffix_directories"] += len(
            directory_names
        )
        candidates = _member_candidates(
            (
                str(info.filename)
                for info in archive_entries
                if not info.is_directory
            ),
            collector.counts,
        )
        collector.counts["archive_candidate_members"] += len(candidates)
        collector.counts["archives_with_candidates"] += bool(candidates)
        if not candidates:
            return
        targets = [name for name, _ in candidates]
        languages = dict(candidates)
        failed_targets: list[str] = []
        with tempfile.TemporaryDirectory(prefix="xgb_vx_deep_") as temporary:
            bundle.extract(path=temporary, targets=targets)
            temporary_root = Path(temporary).resolve()
            for name in targets:
                extracted = (temporary_root / Path(name)).resolve()
                try:
                    extracted.relative_to(temporary_root)
                except ValueError:
                    collector.counts["unsafe_extracted_member"] += 1
                    continue
                try:
                    raw = extracted.read_bytes()
                except OSError:
                    failed_targets.append(name)
                    continue
                collector.consider(
                    raw=raw,
                    language=languages[name],
                    family=family,
                    file_path=f"{relative_archive}#{name}",
                    archive_path=relative_archive,
                )
        if failed_targets:
            try:
                bundle.reset()
                memory = BytesIOFactory(limit=MAX_SOURCE_BYTES)
                bundle.extract(targets=failed_targets, factory=memory)
            except Exception:
                collector.counts["sevenzip_memory_fallback_error"] += len(
                    failed_targets
                )
                return
            for name in failed_targets:
                try:
                    raw = memory.get(name).read()
                except (KeyError, OSError, ValueError):
                    collector.counts["sevenzip_member_read_error"] += 1
                    continue
                collector.counts["sevenzip_memory_recovered"] += 1
                collector.consider(
                    raw=raw,
                    language=languages[name],
                    family=family,
                    file_path=f"{relative_archive}#{name}",
                    archive_path=relative_archive,
                )


def _scan_archives(
    vx_root: Path,
    collector: DeepCollector,
    max_archive_bytes: int,
) -> None:
    archives = sorted(
        path
        for path in vx_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".7z", ".zip"}
    )
    collector.counts["archives_total"] = len(archives)
    for index, archive in enumerate(archives, 1):
        relative = archive.relative_to(vx_root).as_posix()
        try:
            if archive.stat().st_size > max_archive_bytes:
                collector.counts["archive_too_large"] += 1
                continue
            with archive.open("rb") as stream:
                signature = stream.read(6)
            if signature.startswith(b"PK\x03\x04"):
                if archive.suffix.lower() != ".zip":
                    collector.counts["extension_mismatch_zip"] += 1
                _scan_zip(archive, relative, collector)
            elif signature.startswith(b"7z\xbc\xaf\x27\x1c"):
                _scan_7z(archive, relative, collector)
            else:
                collector.counts["unsupported_archive_signature"] += 1
                continue
            collector.counts["archives_readable"] += 1
        except Exception as exc:  # corrupted/unsupported individual archive
            collector.counts[f"archive_error:{type(exc).__name__}"] += 1
        if index % 25 == 0 or index == len(archives):
            print(
                "[vx-deep] "
                f"archives={index}/{len(archives)} "
                f"members={collector.counts['archive_candidate_members']} "
                f"accepted={collector.counts['file_local_candidates']}",
                flush=True,
            )


def _cap_family_rows(
    collector: DeepCollector,
    max_per_family_language: int,
) -> dict[str, list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (family, _language), rows in collector.rows_by_family_language.items():
        selected = sorted(
            rows,
            key=lambda row: (
                -int(row.get("_signal_score") or 0),
                str(row.get("sample_hash") or ""),
            ),
        )[:max_per_family_language]
        collector.counts["family_cap_removed"] += len(rows) - len(selected)
        by_family[family].extend(selected)
    return by_family


def _assign_family_splits(
    by_family: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    totals = Counter()
    family_language_totals = Counter()
    for rows in by_family.values():
        totals.update(str(row["language"]) for row in rows)
        family_language_totals.update({
            str(row["language"]) for row in rows
        })
    target_rows = {
        (split, language): totals[language] * SPLIT_RATIOS[split]
        for split in SPLITS
        for language in LANGUAGES
    }
    target_families = {
        (split, language):
            family_language_totals[language] * SPLIT_RATIOS[split]
        for split in SPLITS
        for language in LANGUAGES
    }
    row_counts: Counter[tuple[str, str]] = Counter()
    family_counts: Counter[tuple[str, str]] = Counter()
    assignments: dict[str, str] = {}
    family_language_counts = {
        family: Counter(str(row["language"]) for row in rows)
        for family, rows in by_family.items()
    }

    def partition_cost() -> float:
        cost = 0.0
        for split in SPLITS:
            for language in LANGUAGES:
                row_target = max(1.0, target_rows[(split, language)])
                cost += (
                    (row_counts[(split, language)] - row_target)
                    / row_target
                ) ** 2
                family_target = max(
                    1.0,
                    target_families[(split, language)],
                )
                cost += 0.10 * (
                    (
                        family_counts[(split, language)]
                        - family_target
                    )
                    / family_target
                ) ** 2
        return cost

    def apply_assignment(
        family: str,
        old_split: str | None,
        new_split: str,
    ) -> None:
        language_counts = family_language_counts[family]
        if old_split is not None:
            for language, count in language_counts.items():
                row_counts[(old_split, language)] -= count
                family_counts[(old_split, language)] -= 1
        for language, count in language_counts.items():
            row_counts[(new_split, language)] += count
            family_counts[(new_split, language)] += 1
        assignments[family] = new_split

    ordered = sorted(
        by_family,
        key=lambda family: (
            -len(by_family[family]),
            hashlib.sha256(family.encode()).hexdigest(),
        ),
    )
    for family in ordered:
        language_counts = family_language_counts[family]
        scored: list[tuple[float, str]] = []
        for candidate_split in SPLITS:
            apply_assignment(family, None, candidate_split)
            cost = partition_cost()
            for language, count in language_counts.items():
                row_counts[(candidate_split, language)] -= count
                family_counts[(candidate_split, language)] -= 1
            assignments.pop(family, None)
            scored.append((cost, candidate_split))
        selected_split = min(
            scored,
            key=lambda item: (
                item[0],
                SPLITS.index(item[1]),
            ),
        )[1]
        apply_assignment(family, None, selected_split)

    # Greedy placement is sensitive to indivisible large families.  Refine it
    # with deterministic one-family moves until the global 70/15/15 objective
    # no longer improves.  Families remain atomic throughout.
    current_cost = partition_cost()
    for _ in range(len(ordered) * 3):
        best: tuple[float, str, str] | None = None
        for family in ordered:
            old_split = assignments[family]
            for new_split in SPLITS:
                if new_split == old_split:
                    continue
                apply_assignment(family, old_split, new_split)
                candidate_cost = partition_cost()
                apply_assignment(family, new_split, old_split)
                candidate = (candidate_cost, family, new_split)
                if (
                    candidate_cost + 1e-12 < current_cost
                    and (best is None or candidate < best)
                ):
                    best = candidate
        if best is None:
            break
        next_cost, family, new_split = best
        apply_assignment(
            family,
            assignments[family],
            new_split,
        )
        current_cost = next_cost
    return assignments


def _load_base_hashes(
    base_dataset: Path,
) -> tuple[dict[str, set[str]], int, int]:
    hashes: dict[str, set[str]] = defaultdict(set)
    kept = 0
    removed = 0
    with base_dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            replace_vx = (
                str(row.get("source") or "") == SOURCE
                and str(row.get("language") or "") in LANGUAGES
                and str(row.get("label") or "") == "malicious"
            )
            if replace_vx:
                removed += 1
                continue
            kept += 1
            digest = str(
                row.get("sample_hash")
                or row.get("artifact_sha256")
                or ""
            )
            if digest:
                hashes[digest].add(str(row.get("label") or ""))
    return hashes, kept, removed


def _write_dataset(
    *,
    base_dataset: Path,
    output_dataset: Path,
    rows: list[dict[str, Any]],
) -> tuple[
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dataset.with_suffix(output_dataset.suffix + ".tmp")
    all_counts: Counter[tuple[str, str, str]] = Counter()
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    output_rows = 0

    def record(row: dict[str, Any]) -> None:
        nonlocal output_rows
        output_rows += 1
        language = str(row.get("language") or "unknown")
        split = str(row.get("split") or "")
        label = str(row.get("label") or "")
        all_counts[(language, split, label)] += 1
        family = str(row.get("family") or "")
        source = str(row.get("source") or "")
        if family:
            family_splits[(source, family)].add(split)
        digest = str(
            row.get("sample_hash")
            or row.get("artifact_sha256")
            or ""
        )
        if digest:
            hash_splits[digest].add(split)

    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        with base_dataset.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                replace_vx = (
                    str(row.get("source") or "") == SOURCE
                    and str(row.get("language") or "") in LANGUAGES
                    and str(row.get("label") or "") == "malicious"
                )
                if replace_vx:
                    continue
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                record(row)
        for row in sorted(
            rows,
            key=lambda value: (
                SPLITS.index(str(value["split"])),
                str(value["language"]),
                str(value["family"]),
                str(value["sample_hash"]),
            ),
        ):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            record(row)
    os.replace(temporary, output_dataset)
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
    counts = [
        {
            "language": language,
            "split": split,
            "label": label,
            "rows": count,
        }
        for (language, split, label), count in sorted(all_counts.items())
    ]
    return output_rows, family_leaks, hash_leaks, counts


def rebuild(
    *,
    incoming_root: Path,
    base_dataset: Path,
    output_dataset: Path,
    report_path: Path,
    max_code_chars: int,
    max_per_family_language: int,
    max_archive_bytes: int,
) -> dict[str, Any]:
    vx_root = incoming_root / "MalwareSourceCode-main"
    if not vx_root.is_dir():
        raise SystemExit(f"VX source directory is missing: {vx_root}")
    base_hashes, base_kept, old_vx_removed = _load_base_hashes(base_dataset)
    collector = DeepCollector(
        base_hash_labels=base_hashes,
        max_code_chars=max_code_chars,
    )
    print("[vx-deep] scanning plain C/C++ sources", flush=True)
    _scan_plain(vx_root, collector)
    print("[vx-deep] scanning every readable .7z/.zip member", flush=True)
    _scan_archives(vx_root, collector, max_archive_bytes)
    by_family = _cap_family_rows(collector, max_per_family_language)
    assignments = _assign_family_splits(by_family)
    selected_rows: list[dict[str, Any]] = []
    for family, rows in by_family.items():
        split = assignments[family]
        for row in rows:
            row["split"] = split
            row["pair_id"] = family
            row["review_notes"] = (
                "Offline text-only deep VX scan; never executed or compiled; "
                "canonical malware family kept in one split."
            )
            row.pop("_signal_score", None)
            selected_rows.append(row)

    output_rows, family_leaks, hash_leaks, all_counts = _write_dataset(
        base_dataset=base_dataset,
        output_dataset=output_dataset,
        rows=selected_rows,
    )
    new_counts: Counter[tuple[str, str]] = Counter(
        (str(row["language"]), str(row["split"]))
        for row in selected_rows
    )
    new_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in selected_rows:
        new_families[(str(row["language"]), str(row["split"]))].add(
            str(row["family"])
        )
    report = {
        "base_dataset": str(base_dataset.resolve()),
        "output_dataset": str(output_dataset.resolve()),
        "incoming_root": str(incoming_root.resolve()),
        "offline_text_only": True,
        "samples_executed_or_compiled": False,
        "base_rows_kept": base_kept,
        "old_vx_c_cpp_rows_removed": old_vx_removed,
        "deep_vx_c_cpp_rows_added": len(selected_rows),
        "output_rows": output_rows,
        "max_per_family_language": max_per_family_language,
        "split_strategy": (
            "deterministic greedy 70/15/15 assignment of canonical malware "
            "families; one family cannot cross splits"
        ),
        "family_split_isolation_verified": not family_leaks,
        "hash_split_isolation_verified": not hash_leaks,
        "family_split_leaks": family_leaks,
        "hash_split_leaks": hash_leaks,
        "scan_counts": dict(sorted(collector.counts.items())),
        "selected_counts": [
            {
                "language": language,
                "split": split,
                "rows": new_counts[(language, split)],
                "families": len(new_families[(language, split)]),
            }
            for language in LANGUAGES
            for split in SPLITS
        ],
        "canonical_families": len(by_family),
        "all_counts": all_counts,
    }
    if family_leaks or hash_leaks:
        output_dataset.unlink(missing_ok=True)
        raise RuntimeError(
            "family/hash split leakage detected; output dataset removed"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-root", required=True, type=Path)
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-code-chars", type=int, default=12_000)
    parser.add_argument("--max-per-family-language", type=int, default=24)
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=MAX_ARCHIVE_BYTES,
    )
    args = parser.parse_args()
    report = rebuild(
        incoming_root=args.incoming_root.resolve(),
        base_dataset=args.base_dataset.resolve(),
        output_dataset=args.output_dataset.resolve(),
        report_path=args.report.resolve(),
        max_code_chars=max(1_000, args.max_code_chars),
        max_per_family_language=max(1, args.max_per_family_language),
        max_archive_bytes=max(MAX_SOURCE_BYTES, args.max_archive_bytes),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
