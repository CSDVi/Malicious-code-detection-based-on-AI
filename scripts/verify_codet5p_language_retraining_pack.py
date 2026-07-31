"""Verify the per-language CodeT5+ continuation-training handoff."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from attack_detection.dataset import is_training_eligible, load_dataset  # noqa: E402
from attack_detection.training.codet5p_classifier_trainer import (  # noqa: E402
    TASKS,
    _select_languages,
    _validate_family_isolation,
    _validate_partitions,
)
from build_codet5p_handoff import _sha256  # noqa: E402
from web.routes.attack_routes import _validate_training_dataset  # noqa: E402


DEFAULT_ROOT = ROOT / "artifacts" / "codet5p_language_retraining_20260724"


def verify(output_root: Path) -> dict[str, object]:
    data_root = output_root / "training_data"
    manifest_path = data_root / "language_retraining_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = []
    for job in manifest["jobs"]:
        path = data_root / str(job["file"])
        task = str(job["task"])
        language = str(job["target_language"])
        positive = TASKS[task]["positive"]
        negative = TASKS[task]["negative"]
        _validate_training_dataset(path, "codet5p")
        rows = [
            sample
            for sample in load_dataset(path)
            if is_training_eligible(sample) and sample.label in {positive, negative}
        ]
        supported = _select_languages(rows, language, positive, negative)
        selected = [sample for sample in rows if sample.language in supported]
        partitions = {
            split: [sample for sample in selected if sample.split == split]
            for split in ("train", "validation", "test")
        }
        _validate_partitions(partitions, positive, negative)
        _validate_family_isolation(selected)
        if path.stat().st_size != int(job["bytes"]):
            raise RuntimeError(f"size mismatch: {path.name}")
        if _sha256(path) != str(job["sha256"]):
            raise RuntimeError(f"SHA-256 mismatch: {path.name}")
        reports.append({
            "file": path.name,
            "language": language,
            "records": len(rows),
            "supported_languages": supported,
            "splits": {name: len(values) for name, values in partitions.items()},
        })

    guide = (data_root / "README_LANGUAGE_RETRAINING.md").read_text(encoding="utf-8")
    if "\ufffd" in guide or "����" in guide:
        raise RuntimeError("training guide contains replacement characters")

    checksums = json.loads((output_root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    zip_reports = {}
    for archive_name, expected_sha256 in checksums.items():
        archive_path = output_root / archive_name
        if _sha256(archive_path) != expected_sha256:
            raise RuntimeError(f"archive SHA-256 mismatch: {archive_name}")
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"bad ZIP member in {archive_name}: {bad}")
            if any(_unsafe_archive_name(name) for name in names):
                raise RuntimeError(f"unsafe path found in {archive_name}")
            zip_reports[archive_name] = {
                "bytes": archive_path.stat().st_size,
                "sha256": expected_sha256,
                "members": len(names),
            }

    project_name = "Xiezhi_CodeT5_Language_Retraining_Project.zip"
    with zipfile.ZipFile(output_root / project_name) as archive:
        names = set(archive.namelist())
        required = {
            "backend/models/codet5p_registry.json",
            "TRAINING_HANDOFF.md",
        }
        if not required.issubset(names):
            raise RuntimeError(f"project archive is missing: {sorted(required - names)}")
        if any("_in_progress" in PurePosixPath(name).name for name in names):
            raise RuntimeError("project archive contains an in-progress checkpoint")
        final_weights = [
            name for name in names
            if name.endswith("/codet5p_classifier.safetensors")
        ]
        if len(final_weights) < 3:
            raise RuntimeError("project archive is missing continuation/runtime model weights")
        zip_reports[project_name]["final_weight_files"] = sorted(final_weights)

    data_name = "Xiezhi_CodeT5_Language_Retraining_Data.zip"
    with zipfile.ZipFile(output_root / data_name) as archive:
        jsonls = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if len(jsonls) != len(manifest["jobs"]):
            raise RuntimeError("data archive does not contain one JSONL per job")

    return {
        "status": "passed",
        "jobs": len(reports),
        "dataset_preflight": reports,
        "archives": zip_reports,
        "published_javascript_version": manifest["already_published"]["version"],
    }


def _unsafe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
