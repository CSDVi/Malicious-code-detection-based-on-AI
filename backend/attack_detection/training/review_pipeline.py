"""Evidence-based review of package files without promoting package-level labels blindly."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from attack_detection.data_pipeline import write_jsonl
from attack_detection.dataset import CodeSample, load_dataset

BEHAVIOR_PATTERNS = {
    "Command Execution": r"\b(eval|exec|system|popen|subprocess|child_process|powershell|cmd\.exe|/bin/sh)\b",
    "Network Access": r"(https?://|requests\.|urllib|fetch\s*\(|socket\.|curl\b|wget\b)",
    "Credential Collection": r"(\.ssh|\.aws|\.npmrc|\.pypirc|id_rsa|credentials|wallet|process\.env|os\.environ)",
    "Install Hook Execution": r"\b(preinstall|postinstall|postInstall|cmdclass|setup_requires|build_ext)\b",
    "Obfuscated Payload": r"(base64|b64decode|fromcharcode|\\x[0-9a-f]{2}|[A-Za-z0-9+/]{100,}={0,2})",
    "Sensitive File Access": r"(passwd|shadow|\.env\b|\.git-credentials|browser|cookies|keychain)",
    "Persistence": r"(startup|crontab|systemd|registry|\.pth\b|site-packages)",
    "Dynamic Download": r"(download|urlretrieve|writefile|response\.content|Invoke-WebRequest)",
}
DECISIVE_BEHAVIORS = {
    "Command Execution", "Credential Collection", "Sensitive File Access", "Persistence", "Dynamic Download",
}


def review_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    decisions_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    samples = load_dataset(dataset_path)
    clean_by_family: dict[str, list[CodeSample]] = defaultdict(list)
    for sample in samples:
        if sample.label == "benign" and sample.source in {"npm_official_registry", "pypi_official_registry"}:
            clean_by_family[sample.family].append(sample)
    decisions = []
    reviewed = []
    for sample in samples:
        if sample.review_status != "needs_review" or sample.label != "malicious":
            reviewed.append(sample)
            continue
        decision = _review_sample(sample, clean_by_family.get(sample.family, []))
        decisions.append(decision)
        if decision["decision"] == "differentially_verified":
            reviewed.append(replace(
                sample,
                review_status="differentially_verified",
                label_confidence=float(decision["confidence"]),
                label_basis=str(decision["label_basis"]),
                behavior_labels=tuple(decision["behavior_labels"]),
                line_labels=tuple(decision["line_labels"]),
                review_notes=str(decision["notes"]),
            ))
        elif decision["decision"] == "behavior_verified":
            reviewed.append(replace(
                sample,
                review_status="behavior_verified",
                label_confidence=float(decision["confidence"]),
                label_basis=str(decision["label_basis"]),
                behavior_labels=tuple(decision["behavior_labels"]),
                line_labels=tuple(decision["line_labels"]),
                review_notes=str(decision["notes"]),
            ))
        else:
            reviewed.append(sample)
    write_jsonl(Path(output_path), reviewed)
    decision_file = Path(decisions_path)
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    with decision_file.open("w", encoding="utf-8", newline="\n") as stream:
        for item in decisions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = Counter(str(item["decision"]) for item in decisions)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(Path(dataset_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "reviewed_records": len(decisions),
        "decisions": dict(counts),
        "training_promoted": counts["differentially_verified"] + counts["behavior_verified"],
        "remaining_human_review": counts["needs_human_review"],
        "policy": {
            "differential": "confirmed compromised package plus malicious behavior introduced relative to an official clean file",
            "intent": "confirmed malicious-intent package plus a multi-stage decisive behavior chain in the file",
            "not_sufficient": ["package membership alone", "version-only diff", "single generic API", "rule hit alone"],
        },
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _review_sample(sample: CodeSample, clean_candidates: list[CodeSample]) -> dict[str, Any]:
    clean = _matching_clean(sample, clean_candidates)
    if sample.source == "datadog_compromised_package_diff" and clean is not None:
        added = _added_lines(clean.code, sample.code)
        line_labels, behaviors = _line_evidence(added, "verified_version_diff")
        decisive = DECISIVE_BEHAVIORS & behaviors
        if decisive and (len(behaviors) >= 2 or "Install Hook Execution" in behaviors):
            return _decision(
                sample,
                "differentially_verified",
                0.88,
                behaviors,
                line_labels,
                "confirmed_compromise_plus_official_clean_version_malicious_delta",
                f"Verified {len(added)} added lines against official clean version {clean.version}.",
            )
    if sample.source == "datadog_malicious_intent":
        numbered = list(enumerate(sample.code.splitlines(), 1))
        line_labels, behaviors = _line_evidence(numbered, "verified_behavior_chain")
        decisive = DECISIVE_BEHAVIORS & behaviors
        entrypoint = PurePosixPath(sample.file_path.replace("\\", "/")).name.lower() in {
            "setup.py", "__init__.py", "package.json", "index.js", "main.js",
        }
        if entrypoint and len(decisive) >= 2 and len(behaviors) >= 3:
            return _decision(
                sample,
                "behavior_verified",
                0.84,
                behaviors,
                line_labels,
                "human_vetted_malicious_package_plus_decisive_multistage_behavior_chain",
                "Package-level human triage is supported by a multi-stage behavior chain in an execution entrypoint.",
            )
    return _decision(
        sample,
        "needs_human_review",
        sample.label_confidence,
        set(sample.behavior_labels),
        list(sample.line_labels),
        sample.label_basis,
        "No independently verifiable file-level malicious behavior chain was found.",
    )


def _matching_clean(sample: CodeSample, candidates: list[CodeSample]) -> CodeSample | None:
    target = _canonical_path(sample.file_path)
    exact = [item for item in candidates if _canonical_path(item.file_path) == target]
    if len(exact) == 1:
        return exact[0]
    basename = PurePosixPath(target).name.lower()
    same_name = [item for item in candidates if PurePosixPath(_canonical_path(item.file_path)).name.lower() == basename]
    return same_name[0] if len(same_name) == 1 else None


def _canonical_path(value: str) -> str:
    parts = list(PurePosixPath(value.replace("\\", "/")).parts)
    lowered = [part.lower() for part in parts]
    if "package" in lowered:
        return "/".join(parts[lowered.index("package") + 1:]).lower()
    while parts and (parts[0].lower().startswith("tmp") or parts[0].lower().endswith(('.tar.gz', '.zip'))):
        parts.pop(0)
    if len(parts) > 1 and re.search(r"[-_]\d", parts[0]):
        parts.pop(0)
    return "/".join(parts).lower()


def _added_lines(clean: str, infected: str) -> list[tuple[int, str]]:
    matcher = difflib.SequenceMatcher(a=clean.splitlines(), b=infected.splitlines(), autojunk=False)
    output = []
    infected_lines = infected.splitlines()
    for tag, _, _, start, end in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            output.extend((index + 1, infected_lines[index]) for index in range(start, end))
    return output


def _line_evidence(
    numbered_lines: list[tuple[int, str]],
    evidence_source: str,
) -> tuple[list[dict[str, object]], set[str]]:
    labels = []
    behaviors = set()
    for line_number, line in numbered_lines:
        matched = [name for name, pattern in BEHAVIOR_PATTERNS.items() if re.search(pattern, line, re.IGNORECASE)]
        for behavior in matched:
            behaviors.add(behavior)
            labels.append({
                "start_line": line_number,
                "end_line": line_number,
                "label": behavior,
                "risk_type": "malicious",
                "cwe": "",
                "source": evidence_source,
                "confidence": 0.88,
            })
    return labels, behaviors


def _decision(
    sample: CodeSample,
    decision: str,
    confidence: float,
    behaviors: set[str],
    line_labels: list[dict[str, object]],
    label_basis: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "sample_hash": sample.sample_hash,
        "family": sample.family,
        "package_name": sample.package_name,
        "version": sample.version,
        "file_path": sample.file_path,
        "source": sample.source,
        "split": sample.split,
        "decision": decision,
        "confidence": round(float(confidence), 3),
        "behavior_labels": sorted(behaviors),
        "line_labels": line_labels,
        "label_basis": label_basis,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review malicious package files using independent differential evidence")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(json.dumps(review_dataset(args.dataset, args.output, args.decisions, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
