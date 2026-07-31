"""Build a family-isolated XGBoost dataset from the new practice-set downloads.

The builder is deliberately static-only.  It never extracts archives to disk,
imports, compiles, or executes sample code.  ZIP members and Parquet rows are
read with bounded memory, normalized, deduplicated, assigned by family to one
split, and appended to an existing reviewed JSONL dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import TRAINING_REVIEW_STATUSES
from attack_detection.features.high_confidence_behaviors import (
    ruby_high_confidence_behavior_count,
    rust_high_confidence_behavior_count,
)

from build_xgb_incoming_multilingual_dataset import (
    MAX_SOURCE_BYTES,
    MIN_CODE_CHARS,
    VENDORED_PARTS,
    _safe_member,
    _vx_signal_score,
)


SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
NORMAL_SPLIT_CAPS = {"train": 1000, "validation": 300, "test": 300}
MAX_CODE_CHARS = 12_000
NORMAL_FAMILY_CAP = 8
MALICIOUS_FAMILY_CAP = 24
NON_PRODUCTION_PARTS = {
    ".github",
    "benchmark",
    "benchmarks",
    "demo",
    "demos",
    "doc",
    "docs",
    "documentation",
    "example",
    "examples",
    "fixture",
    "fixtures",
    "test",
    "tests",
}
NORMAL_JSONL = {
    "batch": ("BatchCMD/data.json", "Batchfile"),
    "powershell": ("PowerShell/data(1).json", "PowerShell"),
    "rust": ("Rust/data(2).json", "Rust"),
    "scala": ("Scala/data(3).json", "Scala"),
    "sql": ("SQL/data(4).json", "SQL"),
    "lua": ("Lua/data(5).json", "Lua"),
}
MALICIOUS_ZIP_CONFIG = {
    "go": ("go", {".go"}),
    "rust": ("Rust", {".rs"}),
    "ruby": ("Ruby", {".rb"}),
    "lua": ("Lua", {".lua"}),
    "typescript": ("TypeScript", {".ts", ".tsx", ".mts", ".cts"}),
}
BENIGN_ARCHIVE_NAMES = {
    "Sophia-Script-for-Windows-main.zip",
    "PSScriptAnalyzer-main.zip",
    "containerd-main.zip",
    "kubernetes-master.zip",
    "terraform-main.zip",
    "coredns-master.zip",
    "grafana-main.zip",
    "moby-master.zip",
    "rust-analyzer-master.zip",
    "tokio-master.zip",
    "serde-master.zip",
    "cargo-master.zip",
    "tauri-dev.zip",
    "runtime-spec-main.zip",
    "runc-main.zip",
    "windows-rs-master.zip",
    "paho-mqtt-master.zip",
    "nats-server-main.zip",
    "caddy-master.zip",
    "miekg-dns-master.zip",
}
SOURCE_REPOSITORIES = {
    "AKILT": "https://github.com/Xart3mis/AKILT",
    "Coldfire": "https://github.com/redcode-labs/Coldfire",
    "emp3r0r": "https://github.com/jm33-m0/emp3r0r",
    "GoAT": "https://github.com/petercunha/GoAT",
    "GoBot2": "https://github.com/SaturnsVoid/GoBot2",
    "go-malware": "https://github.com/omaidf/go-malware",
    "gscript": "https://github.com/gen0cide/gscript",
    "maldev": "https://github.com/D3Ext/maldev",
    "neurax": "https://github.com/redcode-labs/neurax",
    "ransomware": "https://github.com/abhir98/ransomware",
    "ransomwhere": "https://github.com/hazcod/ransomwhere",
    "skuld": "https://github.com/hackirby/skuld",
    "variant": "https://github.com/C1ph3rX13/variant",
    "XMT": "https://github.com/iDigitalFlame/XMT",
    "GC2-sheet": "https://github.com/looCiprian/GC2-sheet",
    "Goasm-RAT": "https://github.com/Zhuagenborn/Goasm-RAT",
    "Discord-rat": "https://github.com/nw8g/Discord-rat",
    "BlackRAT": "https://github.com/fenix544/BlackRAT",
    "GoTokenTheft": "https://github.com/Aquilao/GoTokenTheft",
    "chromecookiestealer": (
        "https://github.com/magisterquis/chromecookiestealer"
    ),
    "Chrome-Password-Recovery": (
        "https://github.com/SaturnsVoid/Chrome-Password-Recovery"
    ),
    "Prince-Ransomware": "https://github.com/oakkaya/Prince-Ransomware",
    "go-crypt": "https://github.com/target111/go-crypt",
    "marmos-ransomware": "https://github.com/marmos91/ransomware",
    "ToRat": "https://github.com/luantak/ToRat",
    "OrcaC2": "https://github.com/Ptkatz/OrcaC2",
    "wintoken": "https://github.com/FourCoreLabs/wintoken",
    "browserpass": "https://github.com/rusq/browserpass",
    "GoStealer": "https://github.com/0xhades/GoStealer",
    "DPAPI-Session-Token-Stealer": (
        "https://github.com/yanard18/DPAPI-Session-Token-Stealer"
    ),
    "WindowsClipSpy": "https://github.com/PiterWeb/WindowsClipSpy",
    "cry-ransomware": "https://github.com/wille/cry",
    "rangoware": "https://github.com/LuanSilveiraSouza/rangoware",
    "ejserna-Ransomware": "https://github.com/ejserna/Ransomware",
    "gustavohenrique-ransomware": (
        "https://github.com/gustavohenrique/ransomware"
    ),
    "simple-golang-ransomware": (
        "https://github.com/knowlet/simple-golang-ransomware"
    ),
    "threeaccents-botnet": "https://github.com/threeaccents/botnet",
    "gobotnet": "https://github.com/andrewaeva/gobotnet",
    "BotnetGo": "https://github.com/1Birdo/BotnetGo",
    "2Pack": "https://github.com/xM0kht4r/2Pack",
    "Fe2O3": "https://github.com/guitmz/Fe2O3",
    "Hazard": "https://github.com/Jsmoreira02/Hazard",
    "hidden-vnc": "https://github.com/EduContin/hidden-vnc",
    "keylogger-rs": "https://github.com/JacobHin2/keylogger-rs",
    "keylogger.rs": "https://github.com/5nyper/keylogger.rs",
    "MalwareDevSeries": "https://github.com/darkarp/MalwareDevSeries",
    "Rust-Crypter": "https://github.com/Kerneldrop/Rust-Crypter",
    "Rust-Hells-Gate": "https://github.com/0xflux/Rust-Hells-Gate",
    "RustHollow": "https://github.com/Kudaes/RustHollow",
    "Rusty-Playground": "https://github.com/BlackSnufkin/Rusty-Playground",
    "self-modifying-malware": "https://github.com/SecSamDev/self-modifying-malware",
    "Simple-Rust-Malware": "https://github.com/cdong1012/Simple-Rust-Malware",
    "VEN0m-Ransomware": "https://github.com/samftggr/VEN0m-Ransomware",
    "XrMT": "https://github.com/iDigitalFlame/XrMT",
    "Hells-Hollow": "https://github.com/0xflux/Hells-Hollow",
    "nakitai": "https://github.com/giwiro/nakitai",
    "cordyceps": "https://github.com/lopes/cordyceps",
    "fc-ransomware": "https://github.com/fChristenson/fc-ransomware",
    "Rust-Ransomware": "https://github.com/amiroooamiran/Rust-Ransomware",
    "big-brogger": "https://github.com/rahzbob/big-brogger",
    "black-backdoorv1.3": "https://github.com/ghostdtdn/black-backdoorv1.3",
    "creds-harvester": "https://github.com/KINGSABRI/creds-harvester",
    "fake_ransomware": "https://github.com/ANorwell/fake_ransomware",
    "fanny.bmp": "https://github.com/loneicewolf/fanny.bmp",
    "ForceCannon": "https://github.com/Jsmoreira02/ForceCannon",
    "kopykat": "https://github.com/KINGSABRI/kopykat",
    "Meterpreter-BackDoor": "https://github.com/OsandaMalith/Meterpreter-BackDoor",
    "Ransome": "https://github.com/Lexterl33t/Ransome",
    "ransomware-mechanics": "https://github.com/prp-e/ransomware-mechanics",
    "ransomware_ability": "https://github.com/mekhalleh/ransomware_ability",
    "rcs-backdoor": "https://github.com/hackedteam/rcs-backdoor",
    "reverse-shell-windows": "https://github.com/DioBruh/reverse-shell-windows",
    "Ruby-Backdoor": "https://github.com/3x1t1um/Ruby-Backdoor",
    "win-reverseshell": "https://github.com/krishpranav/win-reverseshell",
    "browser-backdoor": "https://github.com/IMcPwn/browser-backdoor",
    "spinal_tap": "https://github.com/chadrem/spinal_tap",
    "CMD-Backdoor-Shell": "https://github.com/mrmnh/CMD-Backdoor-Shell",
    "Backdoor_Ruby": "https://github.com/dkhatkar/Backdoor_Ruby",
    "ruby-shells": "https://github.com/secjohn/ruby-shells",
    "Ruby-Bind-and-Reverse-Shells": (
        "https://github.com/Hood3dRob1n/Ruby-Bind-and-Reverse-Shells"
    ),
    "ruby-rootkit": "https://github.com/eVanilla/ruby-rootkit",
    "kai": "https://github.com/ShRP69/kai",
    "Reverse-Ruby": "https://github.com/LoliC0d3/Reverse-Ruby",
    "ruby-reverse-shell": "https://github.com/Matthiasclee/ruby-reverse-shell",
    "ruby-netcat-reverse-shell": (
        "https://github.com/networkdavit/ruby-netcat-reverse-shell"
    ),
    "reversfy": "https://github.com/Habib0x0/reversfy",
    "FUD-logger": "https://github.com/mileticluka1/FUD-logger",
    "green-hat-suite": "https://github.com/Green-m/green-hat-suite",
    "Jasurbek-Ruby-Reverse-Shell": (
        "https://github.com/Jasurbek-Masimov/Ruby-Reverse-Shell"
    ),
    "jennnia-Ruby-reverse-shell": (
        "https://github.com/jennnia/Ruby-reverse-shell"
    ),
    "RevShellRubRub": "https://github.com/ArthurFish6/-RevShellRubRub-",
    "RubyOnWorld-shells": "https://github.com/RubyOnWorld/shells",
    "mtpedro-ruby_reverse_shell": (
        "https://github.com/mtpedro/ruby_reverse_shell"
    ),
    "rb-reverse-shell": "https://github.com/code-developers/rb-reverse-shell",
    "krishpranav-ruby-reverse-shell": (
        "https://github.com/krishpranav/ruby-reverse-shell"
    ),
    "ruby-Reverse-tcp-shell": (
        "https://github.com/r3m0t3nu11/ruby-Reverse-tcp-shell"
    ),
    "simpleRubyReverseBackdoor": (
        "https://github.com/gillesdubois/simpleRubyReverseBackdoor"
    ),
    "FIVEM_CIPHER_MALWARES": "https://github.com/Yaya48/FIVEM_CIPHER_MALWARES",
    "fivem-deobf-malware": "https://github.com/5m1Ly/fivem-deobf-malware",
    "actual-malware": "https://github.com/qpwo/actual-malware",
    "Cacheract": "https://github.com/AdnaneKhan/Cacheract",
    "Memz.js": "https://github.com/SkwalExe/Memz.js",
    "vepar": "https://github.com/UsboKirishima/vepar",
}
TARGET_LANGUAGES = (
    "batch",
    "powershell",
    "go",
    "rust",
    "ruby",
    "kotlin",
    "lua",
    "scala",
    "sql",
    "typescript",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8", errors="ignore"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_code(code: str, max_chars: int = MAX_CODE_CHARS) -> tuple[str, bool]:
    value = code.replace("\x00", "").strip()
    if len(value) <= max_chars:
        return value, False
    head = max_chars * 2 // 3
    tail = max_chars - head
    return (
        value[:head]
        + "\n/* ... STATIC TRAINING SAMPLE TRUNCATED ... */\n"
        + value[-tail:],
        True,
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


def _source_name_from_zip(path: Path) -> str:
    return re.sub(r"-[0-9a-f]{40}$", "", path.stem, flags=re.IGNORECASE)


def _excluded_member(name: str, *, allow_project_lib: bool = False) -> bool:
    parts = {
        part.lower()
        for part in PurePosixPath(name.replace("\\", "/")).parts
    }
    vendored_parts = parts & VENDORED_PARTS
    if allow_project_lib:
        vendored_parts -= {"lib", "libs", "library", "libraries"}
    return bool(vendored_parts) or bool(parts & NON_PRODUCTION_PARTS)


def _make_row(
    *,
    code: str,
    label: str,
    language: str,
    family: str,
    source: str,
    file_path: str,
    source_url: str,
    label_basis: str,
    category: str,
    confidence: float,
    review_status: str,
    behavior_labels: Iterable[str] = (),
    artifact_sha256: str = "",
) -> dict[str, Any] | None:
    original = code.replace("\x00", "").strip()
    if len(original) < MIN_CODE_CHARS:
        return None
    bounded, truncated = _bounded_code(original)
    digest = _sha256_text(original)
    row: dict[str, Any] = {
        "code": bounded,
        "normalized_code": bounded,
        "label": label,
        "category": category,
        "language": language,
        "cwe": "",
        "source": source,
        "package_name": family,
        "version": "",
        "license": "",
        "sample_hash": digest,
        "family": family,
        "published_at": "",
        "split": "pending",
        "artifact_sha256": artifact_sha256 or digest,
        "source_url": source_url,
        "file_path": file_path,
        "paired_version": "",
        "label_basis": label_basis,
        "behavior_labels": list(behavior_labels),
        "cwe_labels": [],
        "label_confidence": confidence,
        "review_status": review_status,
        "parent_sample_hash": "",
        "pair_id": family,
        "pair_slot": label,
        "review_notes": (
            "Offline text-only ingestion; source was never extracted to an "
            "executable location, imported, compiled, or run."
        ),
        "line_labels": [],
        "label_scopes": ["malicious_intent"],
    }
    if truncated:
        row["original_code_length"] = len(original)
        row["training_truncated"] = True
    return row


def _desired_family_slots(
    count: int,
    *,
    min_validation: int,
    min_test: int,
) -> dict[str, int]:
    if count < 3:
        return {
            "train": max(1, count - 1),
            "validation": 0,
            "test": 1 if count > 1 else 0,
        }
    validation = max(min_validation, int(round(count * SPLIT_RATIOS["validation"])))
    test = max(min_test, int(round(count * SPLIT_RATIOS["test"])))
    while validation + test >= count:
        if validation > 1 and validation >= test:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            break
    return {"train": count - validation - test, "validation": validation, "test": test}


def _assign_families(
    rows_by_family: dict[str, list[dict[str, Any]]],
    *,
    min_validation: int = 1,
    min_test: int = 1,
) -> dict[str, str]:
    families = sorted(rows_by_family)
    slots = _desired_family_slots(
        len(families),
        min_validation=min_validation,
        min_test=min_test,
    )
    total_rows = sum(len(rows) for rows in rows_by_family.values())
    row_targets = {
        split: total_rows * (slots[split] / max(1, len(families)))
        for split in SPLITS
    }
    row_counts = Counter()
    family_counts = Counter()
    assignments: dict[str, str] = {}
    ordered = sorted(
        families,
        key=lambda family: (
            -len(rows_by_family[family]),
            _sha256_text(family),
        ),
    )
    for family in ordered:
        choices = [
            split
            for split in SPLITS
            if family_counts[split] < slots[split]
        ]
        if not choices:
            choices = list(SPLITS)
        split = min(
            choices,
            key=lambda name: (
                (row_counts[name] + len(rows_by_family[family]))
                / max(1.0, row_targets[name]),
                family_counts[name] / max(1, slots[name]),
                SPLITS.index(name),
            ),
        )
        assignments[family] = split
        row_counts[split] += len(rows_by_family[family])
        family_counts[split] += 1
    return assignments


def _dedupe_grouped(
    rows_by_family: dict[str, list[dict[str, Any]]],
    *,
    family_cap: int,
    counts: Counter[str],
) -> dict[str, list[dict[str, Any]]]:
    seen: set[str] = set()
    output: dict[str, list[dict[str, Any]]] = {}
    for family in sorted(rows_by_family):
        rows = sorted(
            rows_by_family[family],
            key=lambda row: (
                str(row["sample_hash"]),
                str(row["file_path"]),
            ),
        )
        selected = []
        for row in rows:
            digest = str(row["sample_hash"])
            if digest in seen:
                counts["within_source_exact_duplicate"] += 1
                continue
            seen.add(digest)
            selected.append(row)
            if len(selected) >= family_cap:
                break
        counts["family_cap_removed"] += max(0, len(rows) - len(selected))
        if selected:
            output[family] = selected
    return output


def _select_grouped_rows(
    rows_by_family: dict[str, list[dict[str, Any]]],
    *,
    family_cap: int,
    split_caps: dict[str, int] | None,
    min_validation_families: int,
    min_test_families: int,
    counts: Counter[str],
) -> list[dict[str, Any]]:
    grouped = _dedupe_grouped(
        rows_by_family,
        family_cap=family_cap,
        counts=counts,
    )
    assignments = _assign_families(
        grouped,
        min_validation=min_validation_families,
        min_test=min_test_families,
    )
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family, rows in grouped.items():
        split = assignments[family]
        for row in rows:
            row["split"] = split
            by_split[split].append(row)
    output = []
    for split in SPLITS:
        rows = sorted(
            by_split[split],
            key=lambda row: (
                _sha256_text(str(row["family"]) + str(row["sample_hash"])),
                str(row["sample_hash"]),
            ),
        )
        if split_caps is not None:
            rows = rows[: split_caps[split]]
        output.extend(rows)
    return output


def _collect_normal_jsonl(
    path: Path,
    *,
    language: str,
    counts: Counter[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            counts["normal_jsonl_rows_read"] += 1
            value = json.loads(line)
            code = str(value.get("content") or "")
            repository = str(value.get("repository_name") or "").strip()
            if not repository:
                repository = "unknown:" + _sha256_text(str(value.get("path") or ""))[:16]
            family = f"the_stack_smol:{language}:{repository}"
            row = _make_row(
                code=code,
                label="benign",
                language=language,
                family=family,
                source=f"the_stack_smol_{language}_expansion",
                file_path=str(value.get("path") or ""),
                source_url="https://huggingface.co/datasets/bigcode/the-stack-smol",
                label_basis=(
                    "Public repository code used as a benign candidate; "
                    "family is the source repository."
                ),
                category="benign_source",
                confidence=0.90,
                review_status="source_verified",
            )
            if row is None:
                counts["normal_short_or_empty"] += 1
                continue
            grouped[family].append(row)
    return _select_grouped_rows(
        grouped,
        family_cap=NORMAL_FAMILY_CAP,
        split_caps=NORMAL_SPLIT_CAPS,
        min_validation_families=1,
        min_test_families=1,
        counts=counts,
    )


def _push_ranked(
    heap: list[tuple[int, int]],
    *,
    rank: int,
    row_index: int,
    cap: int,
) -> None:
    item = (-rank, row_index)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _collect_kotlin_normal(path: Path, counts: Counter[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    metadata = pq.read_table(
        path,
        columns=["path", "owner", "name", "repo_id"],
    )
    family_heaps: dict[str, list[tuple[int, int]]] = defaultdict(list)
    paths = metadata["path"].to_pylist()
    owners = metadata["owner"].to_pylist()
    names = metadata["name"].to_pylist()
    repo_ids = metadata["repo_id"].to_pylist()
    counts["kotlin_normal_metadata_rows"] = len(paths)
    for index, (file_path, owner, name, repo_id) in enumerate(
        zip(paths, owners, names, repo_ids)
    ):
        suffix = PurePosixPath(str(file_path or "").lower()).suffix
        if suffix not in {".kt", ".kts"}:
            counts["kotlin_non_source_row"] += 1
            continue
        identity = f"{owner or ''}/{name or ''}".strip("/")
        if not identity:
            identity = f"repo_id:{repo_id}"
        family = f"kotlin_github:{identity}"
        rank = int(
            _sha256_text(f"{family}\0{file_path}\0{index}")[:16],
            16,
        )
        _push_ranked(
            family_heaps[family],
            rank=rank,
            row_index=index,
            cap=NORMAL_FAMILY_CAP,
        )
    pseudo_groups = {
        family: [{"sample_hash": f"{-rank:016x}", "file_path": str(index)} for rank, index in heap]
        for family, heap in family_heaps.items()
    }
    assignments = _assign_families(pseudo_groups)
    split_candidates: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for family, heap in family_heaps.items():
        split = assignments[family]
        for neg_rank, row_index in heap:
            split_candidates[split].append((-neg_rank, row_index, family))
    selected: dict[int, tuple[str, str]] = {}
    for split in SPLITS:
        for _rank, row_index, family in sorted(split_candidates[split])[
            : NORMAL_SPLIT_CAPS[split]
        ]:
            selected[row_index] = (split, family)

    rows: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    absolute_index = 0
    for batch in parquet.iter_batches(
        batch_size=256,
        columns=["path", "content", "owner", "name", "repo_id"],
    ):
        columns = [column.to_pylist() for column in batch.columns]
        for values in zip(*columns):
            selection = selected.get(absolute_index)
            absolute_index += 1
            if selection is None:
                continue
            file_path, code, _owner, _name, _repo_id = values
            split, family = selection
            row = _make_row(
                code=str(code or ""),
                label="benign",
                language="kotlin",
                family=family,
                source="kotlin_github_clean_shard",
                file_path=str(file_path or ""),
                source_url="local parquet shard 1 of 33; see practicesets README",
                label_basis=(
                    "Licensed public Kotlin repository code used as a benign "
                    "candidate; family is the source repository."
                ),
                category="benign_source",
                confidence=0.90,
                review_status="source_verified",
            )
            if row is None:
                counts["kotlin_normal_short_or_empty"] += 1
                continue
            row["split"] = split
            rows.append(row)
    counts["kotlin_normal_selected"] = len(rows)
    return rows


def _collect_typescript_clean(root: Path, counts: Counter[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".ts", ".tsx", ".mts", ".cts"}:
            continue
        relative = path.relative_to(root)
        if any(part.lower() in NON_PRODUCTION_PARTS for part in relative.parts):
            counts["typescript_clean_non_production"] += 1
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            counts["typescript_clean_read_error"] += 1
            continue
        code = _decode_source(raw)
        family_name = relative.parts[0] if relative.parts else path.parent.name
        family = f"paired_clean_typescript:{family_name}"
        row = _make_row(
            code=code,
            label="benign",
            language="typescript",
            family=family,
            source="paired_clean_typescript",
            file_path=relative.as_posix(),
            source_url="local paired clean NPM extraction; see paired_clean_manifest.json",
            label_basis="Analyst-paired clean NPM package source.",
            category="benign_package_source",
            confidence=0.95,
            review_status="source_verified",
        )
        if row is None:
            counts["typescript_clean_short_or_empty"] += 1
            continue
        grouped[family].append(row)
    return _select_grouped_rows(
        grouped,
        family_cap=MALICIOUS_FAMILY_CAP,
        split_caps=None,
        min_validation_families=1,
        min_test_families=1,
        counts=counts,
    )


def _sanitize_powershell(value: str) -> str:
    # The source corpus stores about 100 evasion variants per base script.
    # Some variants append hundreds of kilobytes of random alphanumeric
    # padding.  Static features are bounded to 12k characters, so scanning the
    # first 96k is sufficient to retain the script while avoiding billions of
    # Python-level character checks over irrelevant padding.
    value = value.replace("\x00", "")[: MAX_CODE_CHARS * 8]
    output = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[A-Za-z0-9+/=_-]{128,}", stripped):
            continue
        output.append(line.rstrip())
    return "\n".join(output).strip()


def _looks_like_powershell(code: str) -> bool:
    """Reject obvious Batch/log payloads stored under a .ps1 filename."""
    batch_markers = len(
        re.findall(
            r"(?im)^\s*(?:::|@?echo\b|setlocal\b|set\s+\"|goto\b|call\b|"
            r"if\s+errorlevel\b|cmd\s+/c\b)",
            code,
        )
    )
    powershell_markers = len(
        re.findall(
            r"(?i)(?:\$[A-Za-z_][\w:]*|\b[A-Z][a-z]+-[A-Z][A-Za-z]+\b|"
            r"\bparam\s*\(|\[(?:byte|string|int|object)[^\]]*\])",
            code,
        )
    )
    border_lines = len(re.findall(r"(?m)^\s*\|.{40,}\|\s*$", code))
    line_count = max(1, len(code.splitlines()))
    if batch_markers >= 3 and batch_markers > powershell_markers:
        return False
    if border_lines >= 5 and border_lines / line_count > 0.20:
        return False
    return powershell_markers > 0


def _collect_powershell_malicious(
    root: Path,
    counts: Counter[str],
    *,
    min_validation_families: int = 10,
    min_test_families: int = 10,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pattern = re.compile(r"sample_(\d+)_(\d+)\.ps1", re.IGNORECASE)
    for path in sorted(root.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        counts["powershell_parquet_files"] += 1
        for batch in parquet.iter_batches(
            batch_size=8,
            columns=["filename", "content", "label"],
        ):
            filenames, contents, labels = [
                column.to_pylist()
                for column in batch.columns
            ]
            for filename, raw_content, label in zip(filenames, contents, labels):
                counts["powershell_rows_read"] += 1
                if int(label) != 1:
                    counts["powershell_non_malicious_label"] += 1
                    continue
                match = pattern.fullmatch(str(filename))
                if not match:
                    counts["powershell_unrecognized_filename"] += 1
                    continue
                original = str(raw_content or "")
                code = _sanitize_powershell(original)
                if not _looks_like_powershell(code):
                    counts["powershell_non_powershell_content"] += 1
                    continue
                family = f"powershell_malicious_set_1:family_{int(match.group(1)):02d}"
                signal_score = _vx_signal_score(code)
                row = _make_row(
                    code=code,
                    label="malicious",
                    language="powershell",
                    family=family,
                    source="powershell_malicious_set_1",
                    file_path=f"{path.name}#{filename}",
                    source_url=(
                        "https://huggingface.co/datasets/rr4433/"
                        "powershell_malicious_set_1"
                    ),
                    label_basis=(
                        "Dataset label=1 plus filename-derived base-sample "
                        "family; generated variants remain in one split."
                    ),
                    category="malicious_powershell",
                    confidence=0.95,
                    review_status="source_verified",
                    behavior_labels=(f"static_signal_groups:{signal_score}",),
                    artifact_sha256=_sha256_text(original),
                )
                if row is None:
                    counts["powershell_short_after_sanitize"] += 1
                    continue
                grouped[family].append(row)
    # The source contains roughly 100 variants of each of 56 base samples.
    # Keep extra variants only for training; validation/test stay one row per
    # family so metrics are not multiplied by correlated mutations.
    deduped = _dedupe_grouped(grouped, family_cap=4, counts=counts)
    assignments = _assign_families(
        deduped,
        min_validation=min_validation_families,
        min_test=min_test_families,
    )
    rows = []
    for family, values in deduped.items():
        split = assignments[family]
        cap = 4 if split == "train" else 1
        for row in values[:cap]:
            row["split"] = split
            rows.append(row)
    return rows


def _collect_zip_malicious(
    root: Path,
    *,
    language: str,
    extensions: set[str],
    counts: Counter[str],
    train_only: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for archive in sorted(root.glob("*.zip")):
        if archive.name in BENIGN_ARCHIVE_NAMES:
            counts["github_allowlisted_benign_archive_skipped"] += 1
            continue
        source_name = _source_name_from_zip(archive)
        source_url = SOURCE_REPOSITORIES.get(source_name, "")
        family = f"github_malicious_candidate:{language}:{source_name}"
        counts["github_archives_seen"] += 1
        try:
            with zipfile.ZipFile(archive) as bundle:
                counts["github_archives_readable"] += 1
                for info in bundle.infolist():
                    name = info.filename.replace("\\", "/")
                    if (
                        info.is_dir()
                        or PurePosixPath(name.lower()).suffix not in extensions
                    ):
                        continue
                    counts["github_target_members"] += 1
                    if not _safe_member(name) or _excluded_member(
                        name,
                        allow_project_lib=(language == "ruby"),
                    ):
                        counts["github_excluded_member"] += 1
                        continue
                    if info.file_size > MAX_SOURCE_BYTES:
                        counts["github_oversize_member"] += 1
                        continue
                    try:
                        raw = bundle.read(info)
                    except (KeyError, OSError, RuntimeError, NotImplementedError):
                        counts["github_member_read_error"] += 1
                        continue
                    code = _decode_source(raw).replace("\x00", "").strip()
                    if len(code) < MIN_CODE_CHARS:
                        counts["github_short_or_binary_member"] += 1
                        continue
                    signal_score = _vx_signal_score(code)
                    # A repository-level malware label is insufficient at file
                    # level.  Rust's earlier one-signal policy admitted generic
                    # network, password, UI, and WinAPI helper modules.  Require
                    # two independent broad groups or a specific offensive
                    # behavior chain. Ruby uses its own code-local chains so
                    # generic login/configuration files from a malicious
                    # repository do not inherit a file-level malicious label.
                    rust_strong_behavior_count = (
                        rust_high_confidence_behavior_count(code)
                        if language == "rust"
                        else 0
                    )
                    ruby_strong_behavior_count = (
                        ruby_high_confidence_behavior_count(code)
                        if language == "ruby"
                        else 0
                    )
                    minimum_signal_score = 3 if language == "ruby" else 2
                    if (
                        signal_score < minimum_signal_score
                        and rust_strong_behavior_count == 0
                        and ruby_strong_behavior_count == 0
                    ):
                        counts["github_no_file_local_signal"] += 1
                        continue
                    row = _make_row(
                        code=code,
                        label="malicious",
                        language=language,
                        family=family,
                        source=f"github_{language}_malicious_candidates",
                        file_path=f"{archive.name}#{name}",
                        source_url=source_url,
                        label_basis=(
                            "Repository-level malicious/dual-use provenance "
                            "plus file-local behavior evidence: at least two "
                            "broad groups, or a high-confidence language-specific "
                            "offensive behavior chain."
                        ),
                        category=f"malicious_{language}_source_candidate",
                        confidence=0.90,
                        review_status="behavior_verified",
                        behavior_labels=(
                            f"file_local_signal_groups:{signal_score}",
                            (
                                "rust_high_confidence_behavior_groups:"
                                f"{rust_strong_behavior_count}"
                            ),
                            (
                                "ruby_high_confidence_behavior_groups:"
                                f"{ruby_strong_behavior_count}"
                            ),
                        ),
                        artifact_sha256=_sha256_bytes(raw),
                    )
                    if row is None:
                        counts["github_short_after_normalize"] += 1
                        continue
                    grouped[family].append(row)
        except (OSError, zipfile.BadZipFile):
            counts["github_archive_error"] += 1
    selected = _select_grouped_rows(
        grouped,
        family_cap=MALICIOUS_FAMILY_CAP,
        split_caps=None,
        min_validation_families=7 if language == "ruby" else 2,
        # Rust has only 13 behavior-bearing repositories and several contain
        # very few qualifying files.  Three test repositories are required to
        # keep the positive test row count above the project's minimum of 10.
        min_test_families=(
            7 if language == "ruby" else (3 if language == "rust" else 2)
        ),
        counts=counts,
    )
    if train_only:
        for row in selected:
            row["split"] = "train"
        counts["train_only_augmentation_rows"] += len(selected)
    return selected


def _collect_known_train_family_augmentation(
    root: Path,
    *,
    language: str,
    extensions: set[str],
    base_family_splits: dict[tuple[str, str], set[str]],
    counts: Counter[str],
) -> list[dict[str, Any]]:
    """Add behavior-bearing files only to families already fixed in train.

    Go's strict primary collector requires two local signal groups.  A single
    signal is still useful as lower-confidence training augmentation when the
    repository family is already assigned to train.  Rust/Ruby use the same
    one-signal floor here so files removed only by the regular family cap can
    be recovered without moving validation/test families into training.
    """

    source = f"github_{language}_malicious_candidates"
    allowed_families = {
        family
        for (candidate_source, family), splits in base_family_splits.items()
        if candidate_source == source and splits == {"train"}
    }
    if not allowed_families:
        counts["no_existing_train_families"] += 1
        return []

    rows: list[dict[str, Any]] = []
    for archive in sorted(root.glob("*.zip")):
        if archive.name in BENIGN_ARCHIVE_NAMES:
            continue
        source_name = _source_name_from_zip(archive)
        family = f"github_malicious_candidate:{language}:{source_name}"
        if family not in allowed_families:
            counts["non_train_family_skipped"] += 1
            continue
        source_url = SOURCE_REPOSITORIES.get(source_name, "")
        try:
            with zipfile.ZipFile(archive) as bundle:
                counts["archives_readable"] += 1
                for info in bundle.infolist():
                    name = info.filename.replace("\\", "/")
                    if (
                        info.is_dir()
                        or PurePosixPath(name.lower()).suffix not in extensions
                        or not _safe_member(name)
                        or _excluded_member(
                            name,
                            allow_project_lib=(language == "ruby"),
                        )
                        or info.file_size > MAX_SOURCE_BYTES
                    ):
                        continue
                    try:
                        raw = bundle.read(info)
                    except (KeyError, OSError, RuntimeError, NotImplementedError):
                        counts["member_read_error"] += 1
                        continue
                    code = _decode_source(raw).replace("\x00", "").strip()
                    if len(code) < MIN_CODE_CHARS:
                        continue
                    signal_score = _vx_signal_score(code)
                    if signal_score < 1:
                        counts["no_file_local_signal"] += 1
                        continue
                    rust_strong_behavior_count = (
                        rust_high_confidence_behavior_count(code)
                        if language == "rust"
                        else 0
                    )
                    ruby_strong_behavior_count = (
                        ruby_high_confidence_behavior_count(code)
                        if language == "ruby"
                        else 0
                    )
                    if (
                        language == "rust"
                        and signal_score < 2
                        and rust_strong_behavior_count == 0
                    ):
                        counts["rust_weak_file_local_signal"] += 1
                        continue
                    if (
                        language == "ruby"
                        and signal_score < 3
                        and ruby_strong_behavior_count == 0
                    ):
                        counts["ruby_weak_file_local_signal"] += 1
                        continue
                    # Strong Go rows are already handled by the primary
                    # collector.  Restrict this branch to its omitted
                    # one-signal files; duplicate hashes are removed later.
                    if language == "go" and signal_score != 1:
                        counts["go_primary_collector_signal_skipped"] += 1
                        continue
                    row = _make_row(
                        code=code,
                        label="malicious",
                        language=language,
                        family=family,
                        source=source,
                        file_path=f"{archive.name}#{name}",
                        source_url=source_url,
                        label_basis=(
                            "Known malicious/dual-use repository family already "
                            "isolated in train plus at least one file-local "
                            "behavior signal; train-only augmentation."
                        ),
                        category=f"malicious_{language}_train_augmentation",
                        confidence=0.80 if language == "go" else 0.90,
                        review_status="behavior_verified",
                        behavior_labels=(
                            f"file_local_signal_groups:{signal_score}",
                            (
                                "rust_high_confidence_behavior_groups:"
                                f"{rust_strong_behavior_count}"
                            ),
                            "train_only_known_family_augmentation",
                        ),
                        artifact_sha256=_sha256_bytes(raw),
                    )
                    if row is None:
                        continue
                    row["split"] = "train"
                    rows.append(row)
                    counts["candidate_rows"] += 1
        except (OSError, zipfile.BadZipFile):
            counts["archive_error"] += 1
    return rows


def _collect_zip_benign(
    archives: list[Path],
    *,
    language: str,
    extensions: set[str],
    counts: Counter[str],
    train_only: bool = False,
    family_cap: int = MALICIOUS_FAMILY_CAP,
) -> list[dict[str, Any]]:
    """Collect explicitly allow-listed normal repositories as benign hard-negatives."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for archive in sorted(archives):
        source_name = _source_name_from_zip(archive)
        family = f"github_benign_candidate:{language}:{source_name}"
        try:
            with zipfile.ZipFile(archive) as bundle:
                counts["benign_archives_readable"] += 1
                for info in bundle.infolist():
                    name = info.filename.replace("\\", "/")
                    if (
                        info.is_dir()
                        or PurePosixPath(name.lower()).suffix not in extensions
                        or not _safe_member(name)
                        or _excluded_member(name)
                        or info.file_size > MAX_SOURCE_BYTES
                    ):
                        continue
                    try:
                        raw = bundle.read(info)
                    except (KeyError, OSError, RuntimeError, NotImplementedError):
                        continue
                    code = _decode_source(raw).replace("\x00", "").strip()
                    if len(code) < MIN_CODE_CHARS:
                        continue
                    row = _make_row(
                        code=code,
                        label="benign",
                        language=language,
                        family=family,
                        source=f"github_{language}_benign_candidates",
                        file_path=f"{archive.name}#{name}",
                        source_url="local allow-listed benign repository archive",
                        label_basis="Explicitly allow-listed public repository used as benign hard-negative.",
                        category="benign_source",
                        confidence=0.90,
                        review_status="source_verified",
                        artifact_sha256=_sha256_bytes(raw),
                    )
                    if row is not None:
                        grouped[family].append(row)
        except (OSError, zipfile.BadZipFile):
            counts["benign_archive_error"] += 1
    selected = _select_grouped_rows(
        grouped,
        family_cap=family_cap,
        split_caps=None,
        min_validation_families=2,
        min_test_families=2,
        counts=counts,
    )
    if train_only:
        for row in selected:
            row["split"] = "train"
        counts["train_only_rows"] += len(selected)
    return selected


def _collect_kotlin_malicious(root: Path, counts: Counter[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for archive in sorted(root.rglob("*.zip")):
        family_name = archive.stem
        family = f"android_malware_kotlin:{family_name}"
        counts["kotlin_malware_archives_seen"] += 1
        try:
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    name = info.filename.replace("\\", "/")
                    if (
                        info.is_dir()
                        or PurePosixPath(name.lower()).suffix not in {".kt", ".kts"}
                    ):
                        continue
                    counts["kotlin_malware_members"] += 1
                    if not _safe_member(name) or _excluded_member(name):
                        counts["kotlin_malware_excluded"] += 1
                        continue
                    if info.file_size > MAX_SOURCE_BYTES:
                        counts["kotlin_malware_oversize"] += 1
                        continue
                    try:
                        raw = bundle.read(info, pwd=b"infected")
                    except RuntimeError:
                        raw = bundle.read(info)
                    code = _decode_source(raw).replace("\x00", "").strip()
                    signal_score = _vx_signal_score(code)
                    if signal_score < 1:
                        counts["kotlin_malware_no_file_local_signal"] += 1
                        continue
                    row = _make_row(
                        code=code,
                        label="malicious",
                        language="kotlin",
                        family=family,
                        source="android_malware_kotlin_candidates",
                        file_path=f"{archive.name}#{name}",
                        source_url=(
                            "https://github.com/d-Raco/"
                            "android-malware-source-code-samples"
                        ),
                        label_basis=(
                            "Android malware-family archive plus at least one "
                            "file-local behavior signal group."
                        ),
                        category="malicious_kotlin_source_candidate",
                        confidence=0.90,
                        review_status="behavior_verified",
                        behavior_labels=(f"file_local_signal_groups:{signal_score}",),
                        artifact_sha256=_sha256_bytes(raw),
                    )
                    if row is not None:
                        grouped[family].append(row)
        except (OSError, zipfile.BadZipFile):
            counts["kotlin_malware_archive_error"] += 1
    return _select_grouped_rows(
        grouped,
        family_cap=MALICIOUS_FAMILY_CAP,
        split_caps=None,
        min_validation_families=1,
        min_test_families=1,
        counts=counts,
    )


def _load_base_index(
    base_dataset: Path,
) -> tuple[
    dict[str, set[str]],
    dict[tuple[str, str], set[str]],
    dict[str, set[str]],
    int,
]:
    hash_labels: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    rows = 0
    with base_dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            digest = str(row.get("sample_hash") or row.get("artifact_sha256") or "")
            if digest:
                hash_labels[digest].add(str(row.get("label") or ""))
                hash_splits[digest].add(str(row.get("split") or ""))
            family = str(row.get("family") or "")
            source = str(row.get("source") or "")
            if family:
                family_splits[(source, family)].add(str(row.get("split") or ""))
    return hash_labels, family_splits, hash_splits, rows


def _merge_additions(
    additions: list[dict[str, Any]],
    *,
    base_hash_labels: dict[str, set[str]],
    base_family_splits: dict[tuple[str, str], set[str]],
    counts: Counter[str],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in additions:
        by_hash[str(row["sample_hash"])].append(row)
    output = []
    for digest in sorted(by_hash):
        rows = by_hash[digest]
        labels = {str(row["label"]) for row in rows}
        base_labels = base_hash_labels.get(digest, set())
        if len(labels | base_labels) > 1:
            counts["label_conflict_removed"] += len(rows)
            continue
        if base_labels:
            counts["base_duplicate_removed"] += len(rows)
            continue
        chosen = min(
            rows,
            key=lambda row: (
                str(row["source"]),
                str(row["family"]),
                str(row["file_path"]),
            ),
        )
        base_family_key = (str(chosen["source"]), str(chosen["family"]))
        inherited_splits = base_family_splits.get(base_family_key, set())
        if inherited_splits:
            chosen["split"] = sorted(inherited_splits)[0]
            counts["base_family_split_inherited"] += 1
        counts["cross_source_duplicate_removed"] += len(rows) - 1
        output.append(chosen)
    return output


def _write_dataset(
    *,
    base_dataset: Path,
    output_dataset: Path,
    additions: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dataset.with_suffix(output_dataset.suffix + ".tmp")
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str, str, str]] = Counter()
    output_rows = 0

    def record(row: dict[str, Any]) -> None:
        nonlocal output_rows
        output_rows += 1
        language = str(row.get("language") or "unknown")
        split = str(row.get("split") or "")
        label = str(row.get("label") or "")
        source = str(row.get("source") or "")
        counts[(language, split, label, source)] += 1
        family = str(row.get("family") or "")
        if family:
            family_splits[(source, family)].add(split)
        digest = str(row.get("sample_hash") or row.get("artifact_sha256") or "")
        if digest:
            hash_splits[digest].add(split)

    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        with base_dataset.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                record(row)
        for row in sorted(
            additions,
            key=lambda value: (
                SPLITS.index(str(value["split"])),
                str(value["language"]),
                str(value["label"]),
                str(value["family"]),
                str(value["sample_hash"]),
            ),
        ):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            record(row)
    os.replace(temporary, output_dataset)
    family_leaks = [
        {"source": source, "family": family, "splits": sorted(splits)}
        for (source, family), splits in family_splits.items()
        if len(splits) > 1
    ]
    hash_leaks = [
        {"sample_hash": digest, "splits": sorted(splits)}
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    ]
    table = [
        {
            "language": language,
            "split": split,
            "label": label,
            "source": source,
            "rows": rows,
        }
        for (language, split, label, source), rows in sorted(counts.items())
    ]
    return output_rows, family_leaks, hash_leaks, table


def _readiness(dataset: Path) -> dict[str, Any]:
    row_counts: Counter[tuple[str, str, str]] = Counter()
    families: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                confidence = float(row.get("label_confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if (
                confidence < 0.8
                or str(row.get("review_status") or "") not in TRAINING_REVIEW_STATUSES
            ):
                continue
            language = str(row.get("language") or "unknown")
            split = str(row.get("split") or "")
            label = str(row.get("label") or "")
            if label not in {"benign", "malicious"}:
                continue
            row_counts[(language, split, label)] += 1
            family = str(row.get("family") or "")
            source = str(row.get("source") or "")
            if family:
                families[(language, split, label)].add((source, family))
    minimum_rows = {"train": 20, "validation": 5, "test": 10}
    minimum_malicious_families = {"train": 5, "validation": 2, "test": 2}
    report = {}
    for language in TARGET_LANGUAGES:
        splits = {}
        passed = True
        for split in SPLITS:
            benign = row_counts[(language, split, "benign")]
            malicious = row_counts[(language, split, "malicious")]
            benign_families = len(families[(language, split, "benign")])
            malicious_families = len(families[(language, split, "malicious")])
            row = {
                "benign": benign,
                "malicious": malicious,
                "benign_families": benign_families,
                "malicious_families": malicious_families,
                "minimum_rows_per_class": minimum_rows[split],
                "minimum_malicious_families": minimum_malicious_families[split],
            }
            row["passed"] = (
                benign >= minimum_rows[split]
                and malicious >= minimum_rows[split]
                and malicious_families >= minimum_malicious_families[split]
            )
            passed = passed and bool(row["passed"])
            splits[split] = row
        report[language] = {
            "eligible_for_strict_candidate": passed,
            "splits": splits,
        }
    return report


def _inventory(paths: Iterable[Path], practices_root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(practices_root).as_posix()
        except ValueError:
            relative = str(path)
        output.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "modified_at_local": path.stat().st_mtime,
        })
    return output


def build(
    *,
    practices_root: Path,
    base_dataset: Path,
    output_dataset: Path,
    report_path: Path,
    powershell_min_validation_families: int = 10,
    powershell_min_test_families: int = 10,
    augment_known_train_families: bool = False,
) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    input_paths: list[Path] = []
    base_hash_labels, base_families, base_hash_splits, base_rows = _load_base_index(
        base_dataset
    )
    preexisting_family_leaks = [
        (key, splits)
        for key, splits in base_families.items()
        if len(splits) > 1
    ]
    preexisting_hash_leaks = [
        (digest, splits)
        for digest, splits in base_hash_splits.items()
        if len(splits) > 1
    ]
    if preexisting_family_leaks or preexisting_hash_leaks:
        raise RuntimeError("base dataset already contains family/hash split leakage")

    additions: list[dict[str, Any]] = []
    for language, (relative, expected_language) in NORMAL_JSONL.items():
        path = practices_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        input_paths.append(path)
        print(f"[v11] collecting benign {language}: {path.name}", flush=True)
        additions.extend(
            _collect_normal_jsonl(
                path,
                language=language,
                counts=counters[f"normal:{language}"],
            )
        )

    kotlin_normal = practices_root / "Kotlin" / "train-00000-of-00033.parquet"
    input_paths.append(kotlin_normal)
    print("[v11] selecting benign kotlin rows", flush=True)
    additions.extend(_collect_kotlin_normal(kotlin_normal, counters["normal:kotlin"]))

    typescript_clean = practices_root / "other" / "paired_clean_static"
    print("[v11] collecting paired-clean typescript", flush=True)
    additions.extend(
        _collect_typescript_clean(
            typescript_clean,
            counters["normal:typescript"],
        )
    )

    benign_archives = {
        "powershell": [
            practices_root / "PowerShell" / "Sophia-Script-for-Windows-main.zip",
            practices_root / "PowerShell" / "PSScriptAnalyzer-main.zip",
        ],
        "go": [
            practices_root / "go" / "containerd-main.zip",
            practices_root / "go" / "kubernetes-master.zip",
            practices_root / "go" / "terraform-main.zip",
            practices_root / "go" / "coredns-master.zip",
            practices_root / "go" / "grafana-main.zip",
            practices_root / "go" / "moby-master.zip",
        ],
        "rust": [
            practices_root / "Rust" / "rust-analyzer-master.zip",
            practices_root / "Rust" / "tokio-master.zip",
            practices_root / "Rust" / "serde-master.zip",
            practices_root / "Rust" / "cargo-master.zip",
            practices_root / "Rust" / "tauri-dev.zip",
        ],
    }
    benign_extensions = {
        "powershell": {".ps1", ".psm1", ".psd1"},
        "go": {".go"},
        "rust": {".rs"},
    }
    for language, archives in benign_archives.items():
        present = [archive for archive in archives if archive.is_file()]
        input_paths.extend(present)
        print(f"[v13] collecting allow-listed benign {language} archives", flush=True)
        additions.extend(
            _collect_zip_benign(
                present,
                language=language,
                extensions=benign_extensions[language],
                counts=counters[f"normal_zip:{language}"],
            )
        )

    train_only_benign_archives = {
        "go": [
            practices_root / "go" / "runtime-spec-main.zip",
            # Independent official network/service repositories used only as
            # training hard-negatives.  They teach the detector that MQTT,
            # DNS, HTTP/TLS, authentication configuration, and long-running
            # server loops are not malicious without a stronger behavior
            # chain.  Validation and test routes remain unchanged.
            practices_root / "go" / "paho-mqtt-master.zip",
            practices_root / "go" / "nats-server-main.zip",
            practices_root / "go" / "caddy-master.zip",
            practices_root / "go" / "miekg-dns-master.zip",
        ],
        "rust": [practices_root / "Rust" / "windows-rs-master.zip"],
    }
    for language, archives in train_only_benign_archives.items():
        present = [archive for archive in archives if archive.is_file()]
        if not present:
            continue
        input_paths.extend(present)
        print(
            f"[v23] collecting official train-only benign {language} archives",
            flush=True,
        )
        additions.extend(
            _collect_zip_benign(
                present,
                language=language,
                extensions=benign_extensions[language],
                counts=counters[f"train_only_benign:{language}"],
                train_only=True,
                family_cap=96,
            )
        )

    powershell_root = practices_root / "PowerShell"
    input_paths.extend(sorted(powershell_root.glob("*.parquet")))
    print("[v11] reading malicious PowerShell Parquet shards", flush=True)
    additions.extend(
        _collect_powershell_malicious(
            powershell_root,
            counters["malicious:powershell"],
            min_validation_families=max(1, powershell_min_validation_families),
            min_test_families=max(1, powershell_min_test_families),
        )
    )
    powershell_malicious_archives = [
        powershell_root / "Empire-master.zip",
        powershell_root / "PowerSploit-master.zip",
    ]
    input_paths.extend(
        archive for archive in powershell_malicious_archives if archive.is_file()
    )
    print("[v15] scanning explicit malicious PowerShell ZIP members", flush=True)
    additions.extend(
        _collect_zip_malicious(
            powershell_root,
            language="powershell",
            extensions={".ps1", ".psm1", ".psd1"},
            counts=counters["malicious:powershell_zip"],
            train_only=True,
        )
    )

    for language, (folder, extensions) in MALICIOUS_ZIP_CONFIG.items():
        source_root = practices_root / folder
        input_paths.extend(sorted(source_root.glob("*.zip")))
        print(f"[v11] scanning {language} ZIP members", flush=True)
        additions.extend(
            _collect_zip_malicious(
                source_root,
                language=language,
                extensions=extensions,
                counts=counters[f"malicious:{language}"],
            )
        )
        if augment_known_train_families and language in {"go", "ruby", "rust"}:
            print(
                f"[v22] adding train-only known-family {language} rows",
                flush=True,
            )
            additions.extend(
                _collect_known_train_family_augmentation(
                    source_root,
                    language=language,
                    extensions=extensions,
                    base_family_splits=base_families,
                    counts=counters[f"train_augmentation:{language}"],
                )
            )

    kotlin_malicious = practices_root / "java" / "android_malware_java"
    input_paths.extend(sorted(kotlin_malicious.rglob("*.zip")))
    print("[v11] scanning existing Android Kotlin members", flush=True)
    additions.extend(
        _collect_kotlin_malicious(
            kotlin_malicious,
            counters["malicious:kotlin"],
        )
    )

    merged = _merge_additions(
        additions,
        base_hash_labels=base_hash_labels,
        base_family_splits=base_families,
        counts=counters["merge"],
    )
    output_rows, family_leaks, hash_leaks, all_counts = _write_dataset(
        base_dataset=base_dataset,
        output_dataset=output_dataset,
        additions=merged,
    )
    if family_leaks or hash_leaks:
        output_dataset.unlink(missing_ok=True)
        raise RuntimeError("family/hash split leakage detected; output dataset removed")
    route_readiness = _readiness(output_dataset)
    added_counts = Counter(
        (
            str(row["language"]),
            str(row["split"]),
            str(row["label"]),
            str(row["source"]),
        )
        for row in merged
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "local_timezone": "Asia/Shanghai",
        "objective": "expand malicious-intent XGBoost routes from practicesets",
        "offline_text_only": True,
        "samples_executed_or_compiled": False,
        "base_dataset": str(base_dataset.resolve()),
        "base_dataset_sha256": _sha256_file(base_dataset),
        "base_rows": base_rows,
        "output_dataset": str(output_dataset.resolve()),
        "output_dataset_sha256": _sha256_file(output_dataset),
        "output_rows": output_rows,
        "added_rows": len(merged),
        "family_split_isolation_verified": True,
        "hash_split_isolation_verified": True,
        "family_split_leaks": [],
        "hash_split_leaks": [],
        "split_strategy": (
            "source/repository/base-sample family is atomic; deterministic "
            "family-balanced train/validation/test assignment"
        ),
        "github_label_policy": (
            "repository-level malicious/dual-use provenance plus at least one "
            "file-local behavior signal; vendored/test/demo/docs paths excluded"
        ),
        "powershell_label_policy": (
            "source label=1; filename sample_N_variant establishes 56 base "
            "families; validation/test retain at most one row per family"
        ),
        "added_counts": [
            {
                "language": language,
                "split": split,
                "label": label,
                "source": source,
                "rows": rows,
            }
            for (language, split, label, source), rows in sorted(added_counts.items())
        ],
        "all_counts": all_counts,
        "route_readiness": route_readiness,
        "collector_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(counters.items())
        },
        "input_inventory": _inventory(input_paths, practices_root),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practices-root", required=True, type=Path)
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--powershell-min-validation-families", type=int, default=10)
    parser.add_argument("--powershell-min-test-families", type=int, default=10)
    parser.add_argument(
        "--augment-known-train-families",
        action="store_true",
        help="append lower-confidence behavior rows only to existing train families",
    )
    args = parser.parse_args()
    report = build(
        practices_root=args.practices_root.resolve(),
        base_dataset=args.base_dataset.resolve(),
        output_dataset=args.output_dataset.resolve(),
        report_path=args.report.resolve(),
        powershell_min_validation_families=args.powershell_min_validation_families,
        powershell_min_test_families=args.powershell_min_test_families,
        augment_known_train_families=args.augment_known_train_families,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
