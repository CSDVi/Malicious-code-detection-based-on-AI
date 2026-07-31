"""Safe acquisition and preparation pipeline for source-code risk datasets.

The pipeline never executes package code. Archives remain under ``quarantine``
and allowed source files are read directly into JSONL records in memory.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

from .dataset import CodeSample, DATA_ROOT, DEFAULT_DATASET, ensure_data_directories, load_dataset
from .languages import SOURCE_EXTENSIONS, language_from_path


ALLOWED_HOSTS = {
    "api.github.com",
    "raw.githubusercontent.com",
    "codeload.github.com",
    "github.com",
    "registry.npmjs.org",
    "pypi.org",
    "files.pythonhosted.org",
    "samate.nist.gov",
}
ALLOWED_SOURCE_EXTENSIONS = set(SOURCE_EXTENSIONS) - {".txt"}
NESTED_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
BLOCKED_PARTS = {
    "node_modules", "vendor", "dist", "build", "coverage", ".git", "docs", "documentation",
    "fixtures", "examples", "example", "testdata", "__pycache__",
}
MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
MAX_ARCHIVE_FILES = 12_000
MAX_FILE_BYTES = 768 * 1024
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200

DATADOG_MANIFESTS = {
    "npm": "https://raw.githubusercontent.com/DataDog/malicious-software-packages-dataset/main/samples/npm/manifest.json",
    "pypi": "https://raw.githubusercontent.com/DataDog/malicious-software-packages-dataset/main/samples/pypi/manifest.json",
}
DATADOG_TREE_URL = "https://api.github.com/repos/DataDog/malicious-software-packages-dataset/git/trees/main?recursive=1"
OWASP_URL = "https://codeload.github.com/OWASP-Benchmark/BenchmarkJava/zip/refs/heads/master"
NIST_PHP_URL = "https://samate.nist.gov/SARD/downloads/test-suites/2022-08-02-php-test-suite-xss-v1-0-0.zip"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SafeDownloader:
    def __init__(self, data_root: Path = DATA_ROOT) -> None:
        self.data_root = data_root
        self.log_path = data_root / "manifests" / "downloads.jsonl"

    def download(self, url: str, target: Path, source: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> dict[str, object]:
        self._validate_url(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size:
            record = self._record(url, target, source, "reused")
            self._append(record)
            return record

        request = urllib.request.Request(url, headers={"User-Agent": "XiezhiCodeGuard/1.0 (static dataset research)"})
        partial = target.with_suffix(target.suffix + ".partial")
        digest = hashlib.sha256()
        total = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                self._validate_url(response.geturl())
                declared = int(response.headers.get("Content-Length") or 0)
                if declared > max_bytes:
                    raise ValueError(f"download exceeds limit: {declared} > {max_bytes}")
                with partial.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > max_bytes:
                            raise ValueError(f"download exceeded {max_bytes} bytes")
                        digest.update(block)
                        output.write(block)
            os.replace(partial, target)
        finally:
            if partial.exists() and not target.exists():
                partial.unlink(missing_ok=True)
        record = {
            "source": source,
            "url": url,
            "downloaded_at": utc_now(),
            "path": str(target.relative_to(self.data_root)),
            "size": total,
            "sha256": digest.hexdigest(),
            "status": "downloaded",
        }
        self._append(record)
        return record

    @staticmethod
    def fetch_json(url: str, max_bytes: int = 50 * 1024 * 1024) -> dict[str, object]:
        SafeDownloader._validate_url(url)
        request = urllib.request.Request(url, headers={"User-Agent": "XiezhiCodeGuard/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            SafeDownloader._validate_url(response.geturl())
            raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("metadata response exceeds limit")
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in ALLOWED_HOSTS:
            raise ValueError(f"URL is outside the acquisition allowlist: {url}")
        if parsed.username or parsed.password:
            raise ValueError("credentials in URLs are not allowed")

    def _record(self, url: str, target: Path, source: str, status: str) -> dict[str, object]:
        return {
            "source": source,
            "url": url,
            "downloaded_at": utc_now(),
            "path": str(target.relative_to(self.data_root)),
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
            "status": status,
        }

    def _append(self, record: dict[str, object]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts and not re.match(r"^[A-Za-z]:", name)


def _wanted_source(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    suffix = path.suffix.lower()
    lowered_parts = {part.lower() for part in path.parts}
    if suffix in NESTED_ARCHIVE_EXTENSIONS or suffix not in ALLOWED_SOURCE_EXTENSIONS:
        return False
    if lowered_parts & BLOCKED_PARTS:
        return False
    lowered_name = path.name.lower()
    return not any(marker in lowered_name for marker in (".min.js", ".bundle.js", "generated", "package-lock", "yarn.lock"))


def iter_archive_sources(path: Path, password: bytes | None = None) -> Iterator[tuple[str, str]]:
    """Yield allowed text files without extracting them to the filesystem."""

    if path.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"archive is too large: {path}")
    count = 0
    total = 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not _safe_member(info.filename) or not _wanted_source(info.filename):
                    continue
                count += 1
                total += info.file_size
                if count > MAX_ARCHIVE_FILES or info.file_size > MAX_FILE_BYTES or total > MAX_PACKAGE_BYTES:
                    raise ValueError(f"archive safety limit exceeded: {path.name}")
                if info.compress_size and info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
                    raise ValueError(f"suspicious compression ratio: {info.filename}")
                with archive.open(info, pwd=password) as source:
                    raw = source.read(MAX_FILE_BYTES + 1)
                if len(raw) > MAX_FILE_BYTES or b"\x00" in raw[:4096]:
                    continue
                yield info.filename, raw.decode("utf-8", errors="replace")
        return

    try:
        archive = tarfile.open(path, mode="r:*")
    except tarfile.TarError as exc:
        raise ValueError(f"unsupported archive: {path}") from exc
    with archive:
        for member in archive:
            if not member.isfile() or not _safe_member(member.name) or not _wanted_source(member.name):
                continue
            count += 1
            total += member.size
            if count > MAX_ARCHIVE_FILES or member.size > MAX_FILE_BYTES or total > MAX_PACKAGE_BYTES:
                raise ValueError(f"archive safety limit exceeded: {path.name}")
            source = archive.extractfile(member)
            if source is None:
                continue
            raw = source.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES or b"\x00" in raw[:4096]:
                continue
            yield member.name, raw.decode("utf-8", errors="replace")


def normalize_code(code: str) -> str:
    lines = [re.sub(r"[ \t]+$", "", line) for line in code.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def make_sample(code: str, **metadata: object) -> CodeSample:
    normalized = normalize_code(code)
    category = str(metadata.get("category") or "unknown")
    cwe = str(metadata.get("cwe") or "")
    behavior_labels = _labels(metadata.get("behavior_labels"))
    cwe_labels = _labels(metadata.get("cwe_labels") or cwe)
    return CodeSample(
        code=code.strip(),
        normalized_code=normalized,
        label=str(metadata.get("label") or "benign"),
        category=category,
        language=str(metadata.get("language") or "unknown"),
        cwe=cwe,
        source=str(metadata.get("source") or "unknown"),
        package_name=str(metadata.get("package_name") or ""),
        version=str(metadata.get("version") or ""),
        license=str(metadata.get("license") or ""),
        sample_hash=hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest(),
        family=str(metadata.get("family") or metadata.get("package_name") or ""),
        published_at=str(metadata.get("published_at") or ""),
        split=str(metadata.get("split") or ""),
        artifact_sha256=str(metadata.get("artifact_sha256") or ""),
        source_url=str(metadata.get("source_url") or ""),
        file_path=str(metadata.get("file_path") or ""),
        paired_version=str(metadata.get("paired_version") or ""),
        label_basis=str(metadata.get("label_basis") or ""),
        behavior_labels=tuple(behavior_labels),
        cwe_labels=tuple(cwe_labels),
        label_confidence=_bounded_float(metadata.get("label_confidence")),
        review_status=str(metadata.get("review_status") or "unreviewed"),
        parent_sample_hash=str(metadata.get("parent_sample_hash") or ""),
        review_notes=str(metadata.get("review_notes") or ""),
        line_labels=tuple(metadata.get("line_labels") or ()),
        label_scopes=tuple(_labels(metadata.get("label_scopes"))),
    )


def _labels(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _bounded_float(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def ingest_nist_php(path: Path, limit: int = 5_000) -> list[CodeSample]:
    files = dict(iter_archive_sources(path))
    samples: list[CodeSample] = []
    artifact_hash = sha256_file(path)
    with zipfile.ZipFile(path) as archive:
        manifests = [name for name in archive.namelist() if name.endswith("manifest.sarif") and _safe_member(name)]
        for manifest_name in manifests:
            if len(samples) >= limit:
                break
            metadata = json.loads(archive.read(manifest_name).decode("utf-8"))
            run = metadata.get("runs", [{}])[0]
            properties = run.get("properties", {})
            state = str(properties.get("state", "")).lower()
            label = "vulnerable" if state == "bad" else "benign"
            cwe = ""
            results = run.get("results") or []
            if results:
                cwe = str(results[0].get("ruleId") or "")
            test_root = str(PurePosixPath(manifest_name).parent)
            source_uri = str((run.get("artifacts") or [{}])[0].get("location", {}).get("uri", "src/sample.php"))
            source_name = f"{test_root}/{source_uri}"
            code = files.get(source_name)
            if not code:
                continue
            description = str(properties.get("description") or "")
            category = "XSS" if "xss" in description.lower() or cwe == "CWE-79" else (cwe or "NIST SARD")
            case_id = str(properties.get("id") or PurePosixPath(test_root).name)
            samples.append(make_sample(
                code,
                label=label,
                category=category,
                language="php",
                cwe=cwe,
                source="nist_sard",
                package_name=f"sard-{case_id}",
                version=str(properties.get("version") or "1.0.0"),
                license="NIST dataset terms",
                family=f"nist:{case_id}",
                published_at=str(properties.get("submissionDate") or ""),
                artifact_sha256=artifact_hash,
                source_url=NIST_PHP_URL,
            ))
    return samples


def ingest_owasp(path: Path, limit: int = 5_000) -> list[CodeSample]:
    files = dict(iter_archive_sources(path))
    expected_name = next((name for name in files if "expectedresults" in name.lower() and name.lower().endswith(".csv")), "")
    if not expected_name:
        return []
    expected: dict[str, tuple[str, str, str]] = {}
    for row in csv.reader(io.StringIO(files[expected_name])):
        if len(row) < 4 or not row[0].strip().startswith("BenchmarkTest"):
            continue
        expected[row[0].strip()] = (row[1].strip(), row[2].strip().lower(), row[3].strip())
    artifact_hash = sha256_file(path)
    samples: list[CodeSample] = []
    for name, code in files.items():
        test_name = PurePosixPath(name).stem
        if test_name not in expected or not name.lower().endswith(".java"):
            continue
        category, real, cwe_number = expected[test_name]
        cwe = f"CWE-{cwe_number}" if cwe_number and not cwe_number.upper().startswith("CWE-") else cwe_number.upper()
        samples.append(make_sample(
            code,
            label="vulnerable" if real in {"true", "1", "yes"} else "benign",
            category=category or cwe,
            language="java",
            cwe=cwe,
            source="owasp_benchmark_java",
            package_name=test_name,
            version="1.2",
            license="GNU GPL v2",
            family=f"owasp:{test_name}",
            artifact_sha256=artifact_hash,
            source_url=OWASP_URL,
        ))
        if len(samples) >= limit:
            break
    return samples


def _datadog_path_metadata(path: str) -> dict[str, str]:
    parts = PurePosixPath(path).parts
    if len(parts) < 7:
        return {}
    ecosystem = parts[1]
    category = parts[2]
    version = parts[-2]
    package_name = "/".join(parts[3:-2])
    return {"ecosystem": ecosystem, "category": category, "version": version, "package_name": package_name}


def ingest_datadog_archive(path: Path, repository_path: str, source_url: str) -> list[CodeSample]:
    metadata = _datadog_path_metadata(repository_path)
    if not metadata:
        return []
    language_default = "javascript" if metadata["ecosystem"] == "npm" else "python"
    artifact_hash = sha256_file(path)
    published = PurePosixPath(path.name).name[:10] if re.match(r"\d{4}-\d{2}-\d{2}", path.name) else ""
    samples = []
    for name, code in iter_archive_sources(path, password=b"infected"):
        language = language_from_path(name, language_default)
        samples.append(make_sample(
            code,
            label="malicious",
            category="compromised_package" if "comprom" in metadata["category"] else "malicious_package",
            language=language,
            source="datadog_malicious_packages",
            package_name=metadata["package_name"],
            version=metadata["version"],
            license="See package metadata; dataset Apache-2.0",
            family=f"{metadata['ecosystem']}:{metadata['package_name']}",
            published_at=published,
            artifact_sha256=artifact_hash,
            source_url=source_url,
        ))
    return samples


def _version_key(value: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"[._+\-]", value)
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def choose_clean_version(versions: Iterable[str], affected: Iterable[str], malicious_version: str) -> str:
    bad = {str(value) for value in affected}
    candidates = [str(value) for value in versions if str(value) not in bad]
    if not candidates:
        return ""
    prior = [value for value in candidates if _version_key(value) < _version_key(malicious_version)]
    return sorted(prior or candidates, key=_version_key)[-1]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def acquire_datadog(data_root: Path, limit_per_ecosystem: int, downloader: SafeDownloader) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata_dir = data_root / "raw_metadata"
    for ecosystem, url in DATADOG_MANIFESTS.items():
        downloader.download(url, metadata_dir / f"datadog_{ecosystem}_manifest.json", f"datadog_{ecosystem}_manifest", 5 * 1024 * 1024)
    tree_path = metadata_dir / "datadog_tree.json"
    try:
        tree = _load_json(tree_path)
    except (OSError, json.JSONDecodeError):
        downloader.download(DATADOG_TREE_URL, tree_path, "datadog_repository_tree", 50 * 1024 * 1024)
        tree = _load_json(tree_path)

    selected: list[dict[str, str]] = []
    for ecosystem in ("npm", "pypi"):
        entries = []
        for item in tree.get("tree", []):
            repository_path = str(item.get("path") or "")
            if item.get("type") != "blob" or not repository_path.endswith(".zip") or not repository_path.startswith(f"samples/{ecosystem}/"):
                continue
            meta = _datadog_path_metadata(repository_path)
            if meta:
                entries.append((0 if "comprom" in meta["category"] else 1, int(item.get("size") or 0), repository_path, meta))
        entries.sort(key=lambda value: (value[0], value[1], value[2]))
        used_packages: set[str] = set()
        for _, size, repository_path, meta in entries:
            if len([item for item in selected if item["ecosystem"] == ecosystem]) >= limit_per_ecosystem:
                break
            if size > 8 * 1024 * 1024 or meta["package_name"] in used_packages:
                continue
            used_packages.add(meta["package_name"])
            url = "https://raw.githubusercontent.com/DataDog/malicious-software-packages-dataset/main/" + urllib.parse.quote(repository_path, safe="/@")
            target = data_root / "quarantine" / "datadog" / repository_path.removeprefix("samples/")
            downloader.download(url, target, "datadog_malicious_archive", 10 * 1024 * 1024)
            selected.append({**meta, "repository_path": repository_path, "url": url, "local_path": str(target)})

    pairs = acquire_clean_pairs(data_root, selected, downloader)
    (metadata_dir / "datadog_selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (metadata_dir / "paired_clean_versions.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected, pairs


def acquire_clean_pairs(data_root: Path, selected: list[dict[str, str]], downloader: SafeDownloader) -> list[dict[str, str]]:
    manifests = {
        ecosystem: _load_json(data_root / "raw_metadata" / f"datadog_{ecosystem}_manifest.json")
        for ecosystem in ("npm", "pypi")
    }
    pairs: list[dict[str, str]] = []
    for item in selected:
        ecosystem = item["ecosystem"]
        package_name = item["package_name"]
        affected = manifests[ecosystem].get(package_name)
        if not isinstance(affected, list):
            continue
        try:
            if ecosystem == "npm":
                metadata_url = "https://registry.npmjs.org/" + urllib.parse.quote(package_name, safe="@")
                metadata = SafeDownloader.fetch_json(metadata_url)
                clean_version = choose_clean_version(metadata.get("versions", {}).keys(), affected, item["version"])
                version_data = metadata.get("versions", {}).get(clean_version, {})
                archive_url = str(version_data.get("dist", {}).get("tarball") or "")
                license_name = str(version_data.get("license") or metadata.get("license") or "")
                published_at = str(metadata.get("time", {}).get(clean_version) or "")
            else:
                metadata_url = f"https://pypi.org/pypi/{urllib.parse.quote(package_name, safe='')}/json"
                metadata = SafeDownloader.fetch_json(metadata_url)
                clean_version = choose_clean_version(metadata.get("releases", {}).keys(), affected, item["version"])
                release_files = metadata.get("releases", {}).get(clean_version, [])
                package_file = next((entry for entry in release_files if entry.get("packagetype") == "sdist"), release_files[0] if release_files else {})
                archive_url = str(package_file.get("url") or "")
                license_name = str(metadata.get("info", {}).get("license") or "")
                published_at = str(package_file.get("upload_time_iso_8601") or "")
            if not clean_version or not archive_url:
                continue
            suffix = ".tgz" if ecosystem == "npm" else Path(urllib.parse.urlparse(archive_url).path).suffix
            if archive_url.endswith(".tar.gz"):
                suffix = ".tar.gz"
            target = data_root / "quarantine" / "paired_clean" / ecosystem / _safe_package_name(package_name) / f"{clean_version}{suffix}"
            downloader.download(archive_url, target, f"{ecosystem}_official_clean_version", 50 * 1024 * 1024)
            pairs.append({
                "ecosystem": ecosystem,
                "package_name": package_name,
                "malicious_version": item["version"],
                "clean_version": clean_version,
                "url": archive_url,
                "local_path": str(target),
                "license": license_name,
                "published_at": published_at,
            })
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            continue
    return pairs


def _safe_package_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "package"


def ingest_clean_pair(item: dict[str, str]) -> list[CodeSample]:
    path = Path(item["local_path"])
    artifact_hash = sha256_file(path)
    default_language = "javascript" if item["ecosystem"] == "npm" else "python"
    samples = []
    for name, code in iter_archive_sources(path):
        language = language_from_path(name, default_language)
        samples.append(make_sample(
            code,
            label="benign",
            category="paired_clean_version",
            language=language,
            source=f"{item['ecosystem']}_official_registry",
            package_name=item["package_name"],
            version=item["clean_version"],
            license=item.get("license", ""),
            family=f"{item['ecosystem']}:{item['package_name']}",
            published_at=item.get("published_at", ""),
            artifact_sha256=artifact_hash,
            source_url=item["url"],
        ))
    return samples


TOKEN_PATTERN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+|[^\s]", re.UNICODE)


def _minhash(code: str, permutations: int = 16) -> tuple[int, ...]:
    tokens = TOKEN_PATTERN.findall(normalize_code(code).lower())
    # Large generated/vendor-like files previously produced every possible
    # shingle before the 512-shingle cap was applied.  Bound the token stream
    # first while retaining the beginning, middle, and end of the file.
    if len(tokens) > 4_096:
        middle = len(tokens) // 2
        tokens = tokens[:1_536] + tokens[middle - 512:middle + 512] + tokens[-1_536:]
    shingles = {"\x1f".join(tokens[index:index + 5]) for index in range(max(1, len(tokens) - 4))}
    if not shingles:
        shingles = {normalize_code(code).lower()}
    if len(shingles) > 512:
        ordered = sorted(shingles, key=lambda value: hashlib.sha1(value.encode("utf-8")).digest())
        shingles = set(ordered[:512])
    values = [int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") for value in shingles]
    prime = 18_446_744_073_709_551_557
    return tuple(
        min(((2 * seed + 1) * value + (seed + 1) * 1_000_003) % prime for value in values)
        for seed in range(permutations)
    )


def deduplicate(samples: list[CodeSample], threshold: float = 0.9) -> tuple[list[CodeSample], dict[str, object]]:
    by_hash: dict[str, list[CodeSample]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.sample_hash].append(sample)
    kept: list[CodeSample] = []
    exact_removed = 0
    label_conflicts = []
    conflict_samples = 0
    for sample_hash, matches in by_hash.items():
        labels = {sample.label for sample in matches}
        if len(labels) > 1:
            label_conflicts.append({"sha256": sample_hash, "labels": sorted(labels), "samples": len(matches)})
            conflict_samples += len(matches)
            continue
        kept.append(matches[0])
        exact_removed += len(matches) - 1

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    signatures: list[tuple[int, ...]] = []
    output: list[CodeSample] = []
    near_removed = 0
    for sample in kept:
        signature = _minhash(sample.normalized_code or sample.code)
        candidates: set[int] = set()
        band_count = len(signature) // 4
        for band in range(band_count):
            key = (band, signature[band * 4:(band + 1) * 4])
            candidates.update(buckets[key])
        duplicate = False
        for index in candidates:
            other = output[index]
            if other.label != sample.label:
                continue
            similarity = sum(a == b for a, b in zip(signature, signatures[index])) / len(signature)
            if similarity >= threshold:
                duplicate = True
                near_removed += 1
                break
        if duplicate:
            continue
        index = len(output)
        output.append(sample)
        signatures.append(signature)
        for band in range(band_count):
            buckets[(band, signature[band * 4:(band + 1) * 4])].append(index)
    report = {
        "input": len(samples),
        "output": len(output),
        "exact_removed": exact_removed,
        "conflict_samples_quarantined": conflict_samples,
        "near_removed": near_removed,
        "near_threshold": threshold,
        "label_conflicts": label_conflicts,
    }
    return output, report


def assign_splits(samples: list[CodeSample], heldout_sources: set[str] | None = None) -> list[CodeSample]:
    heldout_sources = heldout_sources or {"nist_sard"}
    group_split: dict[str, str] = {}
    output = []
    for sample in samples:
        group = sample.family or sample.package_name or f"{sample.source}:{sample.sample_hash[:16]}"
        if sample.source == "evasion_suite" or sample.source in heldout_sources:
            split = "test"
        elif group in group_split:
            split = group_split[group]
        else:
            bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
            split = "test" if bucket < 20 else ("validation" if bucket < 35 else "train")
            group_split[group] = split
        output.append(replace(sample, split=split))
    return output


def generate_evasion_suite(samples: list[CodeSample], limit: int = 300) -> list[CodeSample]:
    suite: list[CodeSample] = []
    for sample in samples:
        if sample.label == "benign" or len(suite) >= limit:
            continue
        transforms = [
            ("identifier_rename", _rename_identifiers(sample.code)),
            ("string_split", _split_strings(sample.code, sample.language)),
            ("base64_literal", _encode_one_literal(sample.code, sample.language)),
            ("dead_code", _insert_dead_code(sample.code, sample.language)),
        ]
        for method, code in transforms:
            if code == sample.code or len(suite) >= limit:
                continue
            suite.append(make_sample(
                code,
                label=sample.label,
                category=f"evasion:{method}:{sample.category}",
                language=sample.language,
                cwe=sample.cwe,
                source="evasion_suite",
                package_name=sample.package_name,
                version=sample.version,
                license=sample.license,
                family=sample.family or sample.package_name,
                published_at=sample.published_at,
                split=sample.split,
                artifact_sha256=sample.artifact_sha256,
                source_url=sample.source_url,
                file_path=sample.file_path,
                paired_version=sample.paired_version,
                label_basis=f"generated_from:{sample.sample_hash}:{method}",
                behavior_labels=sample.behavior_labels,
                cwe_labels=sample.cwe_labels,
                label_confidence=max(0.0, sample.label_confidence - 0.05),
                review_status="generated_variant",
                parent_sample_hash=sample.sample_hash,
                line_labels=sample.line_labels if method != "dead_code" else (),
            ))
    return suite


def _rename_identifiers(code: str) -> str:
    keywords = {"if", "else", "for", "while", "return", "class", "def", "function", "new", "public", "private", "true", "false", "null", "none", "import", "from"}
    names = []
    for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", code):
        if name.lower() not in keywords and name not in names:
            names.append(name)
        if len(names) >= 4:
            break
    output = code
    for index, name in enumerate(names):
        output = re.sub(rf"\b{re.escape(name)}\b", f"v_{index:02d}", output)
    return output


def _split_strings(code: str, language: str) -> str:
    operator = "." if language == "php" else "+"
    pattern = re.compile(r"(['\"])([^'\"\n]{8,})\1")
    match = pattern.search(code)
    if not match:
        return code
    value = match.group(2)
    middle = len(value) // 2
    replacement = f"{match.group(1)}{value[:middle]}{match.group(1)} {operator} {match.group(1)}{value[middle:]}{match.group(1)}"
    return code[:match.start()] + replacement + code[match.end():]


def _encode_one_literal(code: str, language: str) -> str:
    pattern = re.compile(r"(['\"])([^'\"\n]{8,})\1")
    match = pattern.search(code)
    if not match:
        return code
    encoded = base64.b64encode(match.group(2).encode()).decode()
    if language == "php":
        replacement = f"base64_decode('{encoded}')"
    elif language == "python":
        replacement = f"__import__('base64').b64decode('{encoded}').decode()"
    elif language in {"javascript", "typescript"}:
        replacement = f"atob('{encoded}')"
    else:
        return code
    return code[:match.start()] + replacement + code[match.end():]


def _insert_dead_code(code: str, language: str) -> str:
    if language == "python":
        return "if False:\n    marker = 'unused'\n" + code
    if language == "php":
        return "<?php if (false) { $marker = 'unused'; } ?>\n" + code
    return "if (false) { const marker = 'unused'; }\n" + code


def write_jsonl(path: Path, samples: Iterable[CodeSample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(data_root: Path = DATA_ROOT, datadog_limit: int = 5, owasp_limit: int = 5_000, nist_limit: int = 5_000) -> dict[str, object]:
    ensure_data_directories(data_root)
    downloader = SafeDownloader(data_root)
    owasp_path = data_root / "quarantine" / "owasp_benchmark_java.zip"
    nist_path = data_root / "quarantine" / "nist_php_xss_v1.zip"
    downloader.download(OWASP_URL, owasp_path, "owasp_benchmark_java", MAX_DOWNLOAD_BYTES)
    downloader.download(NIST_PHP_URL, nist_path, "nist_sard_php_xss", 10 * 1024 * 1024)
    selected, pairs = acquire_datadog(data_root, datadog_limit, downloader)

    samples = list(load_dataset(DEFAULT_DATASET))
    samples.extend(ingest_owasp(owasp_path, owasp_limit))
    samples.extend(ingest_nist_php(nist_path, nist_limit))
    for item in selected:
        samples.extend(ingest_datadog_archive(Path(item["local_path"]), item["repository_path"], item["url"]))
    for item in pairs:
        samples.extend(ingest_clean_pair(item))

    deduped, dedupe_report = deduplicate(samples)
    split_samples = assign_splits(deduped)
    evasions = generate_evasion_suite(split_samples)
    all_samples = split_samples + evasions
    processed_path = data_root / "processed" / "phase1_dataset.jsonl"
    write_jsonl(processed_path, all_samples)
    write_jsonl(data_root / "processed" / "evasion_tests.jsonl", evasions)
    for split in ("train", "validation", "test"):
        write_jsonl(data_root / "splits" / f"{split}.jsonl", (sample for sample in all_samples if sample.split == split))

    manifest = {
        "schema_version": 2,
        "created_at": utc_now(),
        "dataset_path": str(processed_path),
        "samples": len(all_samples),
        "labels": dict(Counter(sample.label for sample in all_samples)),
        "languages": dict(Counter(sample.language for sample in all_samples)),
        "sources": dict(Counter(sample.source for sample in all_samples)),
        "splits": dict(Counter(sample.split for sample in all_samples)),
        "packages": len({sample.package_name for sample in all_samples if sample.package_name}),
        "families": len({sample.family for sample in all_samples if sample.family}),
        "datadog_archives": len(selected),
        "paired_clean_archives": len(pairs),
        "evasion_samples": len(evasions),
        "deduplication": dedupe_report,
        "heldout_sources": ["nist_sard"],
        "safety": {
            "executed_samples": False,
            "archive_mode": "read-only in-memory source extraction",
            "allowed_extensions": sorted(ALLOWED_SOURCE_EXTENSIONS),
            "max_file_bytes": MAX_FILE_BYTES,
            "max_package_bytes": MAX_PACKAGE_BYTES,
            "max_compression_ratio": MAX_COMPRESSION_RATIO,
        },
        "artifacts": _artifact_manifest(data_root),
    }
    (data_root / "manifests" / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_root / "manifests" / "dedupe_report.json").write_text(json.dumps(dedupe_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _artifact_manifest(data_root: Path) -> list[dict[str, object]]:
    artifacts = []
    for path in sorted((data_root / "quarantine").rglob("*")):
        if path.is_file() and path.name != ".gitkeep" and not path.name.endswith(".partial"):
            artifacts.append({
                "path": str(path.relative_to(data_root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and prepare the isolated phase-one dataset")
    parser.add_argument("--datadog-limit", type=int, default=5, help="malicious archives per ecosystem")
    parser.add_argument("--owasp-limit", type=int, default=5000)
    parser.add_argument("--nist-limit", type=int, default=5000)
    args = parser.parse_args()
    manifest = build_dataset(datadog_limit=max(0, args.datadog_limit), owasp_limit=args.owasp_limit, nist_limit=args.nist_limit)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
