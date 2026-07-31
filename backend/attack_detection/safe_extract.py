"""Safely extract static text from quarantined dataset archives.

This utility does not execute package code and does not preserve executable
permissions. It rejects path traversal, symlinks, nested archives, oversized
members, suspicious compression ratios, binary content, and generated/vendor
trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ALLOWED_EXTENSIONS = {
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".json", ".yml", ".yaml", ".sh", ".py", ".java", ".php",
}
BLOCKED_PARTS = {
    ".git", "node_modules", "vendor", "dist", "build", "coverage",
    "docs", "documentation", "examples", "example", "testdata", "__pycache__",
}
NESTED_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
MAX_FILE_BYTES = 768 * 1024
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_FILES_PER_ARCHIVE = 12_000
MAX_COMPRESSION_RATIO = 200
SUPPORTED_ARCHIVE_SUFFIXES = (".zip", ".tgz", ".tar.gz", ".tar")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not path.parts or path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", normalized):
        return None
    return path


def _wanted_member(path: PurePosixPath) -> bool:
    suffix = path.suffix.lower()
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    if suffix in NESTED_ARCHIVE_EXTENSIONS or suffix not in ALLOWED_EXTENSIONS:
        return False
    if lowered_parts & BLOCKED_PARTS:
        return False
    return not any(marker in lowered_name for marker in (".min.js", ".bundle.js", "package-lock", "yarn.lock", "generated"))


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _archive_output_name(archive_path: Path, input_root: Path) -> str:
    relative = archive_path.relative_to(input_root).as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9@._-]+", "_", archive_path.stem).strip("._") or "archive"
    path_digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem}__{path_digest}"


def _store_static_member(
    package_dir: Path,
    output_root: Path,
    member_name: str,
    raw: bytes,
    extracted: list[dict[str, object]],
) -> None:
    content_hash = _sha256_bytes(raw)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", PurePosixPath(member_name).name)[-80:]
    destination = (package_dir / "files" / f"{len(extracted):05d}_{content_hash[:12]}_{safe_name}").resolve()
    package_resolved = package_dir.resolve()
    if destination != package_resolved and package_resolved not in destination.parents:
        raise ValueError(f"unsafe output path for member: {member_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(raw.decode("utf-8", errors="replace"), encoding="utf-8", newline="\n")
    try:
        os.chmod(destination, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    extracted.append({
        "source_member": member_name,
        "output": str(destination.relative_to(output_root)),
        "bytes": len(raw),
        "sha256": content_hash,
    })


def extract_archive(
    archive_path: Path,
    output_root: Path,
    password: bytes,
    input_root: Path | None = None,
) -> dict[str, object]:
    output_root = output_root.resolve()
    archive_path = archive_path.resolve()
    archive_root = (input_root or archive_path.parent).resolve()
    package_dir = output_root / _archive_output_name(archive_path, archive_root)
    package_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    skipped: dict[str, int] = {}
    total_bytes = 0

    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) > MAX_FILES_PER_ARCHIVE:
            raise ValueError(f"too many entries: {archive_path.name}")
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = _safe_member(info.filename)
            reason = ""
            if member is None:
                reason = "unsafe_path"
            elif _is_symlink(info):
                reason = "symlink"
            elif not _wanted_member(member):
                reason = "not_allowlisted"
            elif info.file_size > MAX_FILE_BYTES:
                reason = "file_too_large"
            elif info.compress_size and info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
                reason = "compression_ratio"
            elif total_bytes + info.file_size > MAX_PACKAGE_BYTES:
                reason = "package_too_large"
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue

            try:
                raw = archive.read(info, pwd=password)
            except (RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
                raise ValueError(f"cannot decrypt/read {archive_path.name}:{info.filename}: {exc}") from exc
            if len(raw) > MAX_FILE_BYTES:
                skipped["file_too_large"] = skipped.get("file_too_large", 0) + 1
                continue
            if b"\x00" in raw[:4096]:
                skipped["binary"] = skipped.get("binary", 0) + 1
                continue

            _store_static_member(package_dir, output_root, info.filename, raw, extracted)
            total_bytes += len(raw)

    return {
        "archive": archive_path.relative_to(archive_root).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "output_directory": str(package_dir.relative_to(output_root)),
        "extracted_files": len(extracted),
        "extracted_bytes": total_bytes,
        "skipped": skipped,
        "files": extracted,
    }


def extract_tar_archive(
    archive_path: Path,
    output_root: Path,
    input_root: Path,
) -> dict[str, object]:
    output_root = output_root.resolve()
    archive_path = archive_path.resolve()
    archive_root = input_root.resolve()
    package_dir = output_root / _archive_output_name(archive_path, archive_root)
    package_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, object]] = []
    skipped: dict[str, int] = {}
    total_bytes = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        if len(members) > MAX_FILES_PER_ARCHIVE:
            raise ValueError(f"too many entries: {archive_path.name}")
        for info in members:
            if not info.isfile():
                if info.issym() or info.islnk():
                    skipped["symlink"] = skipped.get("symlink", 0) + 1
                continue
            member = _safe_member(info.name)
            reason = ""
            if member is None:
                reason = "unsafe_path"
            elif not _wanted_member(member):
                reason = "not_allowlisted"
            elif info.size > MAX_FILE_BYTES:
                reason = "file_too_large"
            elif total_bytes + info.size > MAX_PACKAGE_BYTES:
                reason = "package_too_large"
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            source = archive.extractfile(info)
            if source is None:
                skipped["unreadable"] = skipped.get("unreadable", 0) + 1
                continue
            raw = source.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                skipped["file_too_large"] = skipped.get("file_too_large", 0) + 1
                continue
            if b"\x00" in raw[:4096]:
                skipped["binary"] = skipped.get("binary", 0) + 1
                continue
            _store_static_member(package_dir, output_root, info.name, raw, extracted)
            total_bytes += len(raw)
    return {
        "archive": archive_path.relative_to(archive_root).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "output_directory": str(package_dir.relative_to(output_root)),
        "extracted_files": len(extracted),
        "extracted_bytes": total_bytes,
        "skipped": skipped,
        "files": extracted,
    }


def extract_directory(input_dir: Path, output_dir: Path, password: str = "infected") -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    archives = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.name.lower().endswith(SUPPORTED_ARCHIVE_SUFFIXES)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    for archive in archives:
        try:
            if zipfile.is_zipfile(archive):
                results.append(extract_archive(archive, output_dir, password.encode("utf-8"), input_dir))
            else:
                results.append(extract_tar_archive(archive, output_dir, input_dir))
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            errors.append({"archive": archive.relative_to(input_dir).as_posix(), "error": str(exc)})

    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "executed_samples": False,
        "archives": len(archives),
        "successful_archives": len(results),
        "failed_archives": len(errors),
        "extracted_files": sum(int(item["extracted_files"]) for item in results),
        "extracted_bytes": sum(int(item["extracted_bytes"]) for item in results),
        "limits": {
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            "max_file_bytes": MAX_FILE_BYTES,
            "max_package_bytes": MAX_PACKAGE_BYTES,
            "max_compression_ratio": MAX_COMPRESSION_RATIO,
        },
        "errors": errors,
        "results": results,
    }
    manifest_path = output_dir.parent / f"{output_dir.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract allowlisted static text from encrypted ZIP datasets")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--password", default="infected")
    args = parser.parse_args()
    print(json.dumps(extract_directory(args.input_dir, args.output_dir, args.password), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
