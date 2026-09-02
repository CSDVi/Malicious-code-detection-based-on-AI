"""Cross-modal source-to-binary software-genome verification.

The runtime is deliberately read-only: it parses a bounded source archive and
a PE-family artifact, builds two graphs, and compares their observable
evidence. A separately quality-gated twin GAT may replace the transparent
baseline when its manifest and weights are installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .binary_analysis import SUSPICIOUS_IMPORTS, parse_pe
from .data_pipeline import make_sample
from .features.graph_builder import build_lightweight_graph, build_project_graph
from .languages import (
    SOURCE_EXTENSIONS,
    decode_source_payload,
    detect_source_language,
    is_generic_text_path,
    is_probably_text_payload,
)
from .static_analysis.strings_ioc import printable_strings


MAX_SOURCE_FILES = 160
MAX_SOURCE_MEMBER_BYTES = 4 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_SOURCE_MEMBERS = 20_000
MAX_COMPRESSION_RATIO = 100
MAX_SOURCE_ANALYSIS_BYTES = 512 * 1024
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
STRING_PATTERN = re.compile(r"(?P<quote>['\"])(?P<value>[^'\"\r\n]{5,160})(?P=quote)")
WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$@?-]{3,}")
GENERIC_SYMBOLS = {
    "main", "init", "true", "false", "none", "null", "this", "self",
    "string", "object", "return", "public", "private", "static", "class",
    "function", "system", "windows", "program", "application", "error",
}


class GenomeAnalysisError(ValueError):
    """Raised when a source/artifact pair cannot be safely analyzed."""


def analyze_software_genome(
    source_archive: str | Path | bytes | BinaryIO,
    artifact_payload: bytes,
    *,
    source_name: str = "source.zip",
    artifact_name: str = "artifact.exe",
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build source/binary graphs and return a conservative integrity verdict."""

    if not artifact_payload:
        raise GenomeAnalysisError("构建产物为空。")
    parsed = parse_pe(artifact_payload)
    if not parsed.get("is_pe"):
        raise GenomeAnalysisError("构建产物不是有效的 PE、EXE、DLL、SYS 或 OCX 文件。")

    source_records, warnings = _read_source_archive(source_archive, source_name)
    if not source_records:
        raise GenomeAnalysisError("源码包中没有可分析的源代码文件。")
    samples = [
        make_sample(
            record["content"],
            language=record["language"],
            file_path=record["path"],
            package_name=Path(source_name).stem,
            family=Path(source_name).stem,
            version="uploaded",
            source="software_genome_upload",
        )
        for record in source_records
    ]
    source_graph = build_project_graph(samples)
    artifact_graph = build_artifact_graph(parsed, artifact_name, artifact_payload)
    baseline = _baseline_compare(source_records, parsed, artifact_payload)
    model = _run_twin_model(
        source_graph,
        artifact_graph,
        Path(model_dir) if model_dir else MODEL_DIR,
    )

    if model.get("status") == "completed":
        verdict = str(model["decision"])
        verdict_label = "疑似被替换或注入" if verdict == "review" else "跨模态一致"
        decision_basis = "twin_gat"
        verdict_reason = (
            f"孪生 GAT 给出的篡改概率为 {model['tamper_probability']}，"
            f"发布阈值为 {model['threshold']}。"
        )
    else:
        verdict = str(baseline["decision"])
        verdict_label = {
            "consistent": "发现一致性证据",
            "review": "发现需复核的产物侧增量",
            "inconclusive": "证据不足",
        }[verdict]
        decision_basis = "structural_baseline"
        verdict_reason = str(baseline["reason"])

    return {
        "source_name": Path(source_name).name,
        "artifact_name": Path(artifact_name).name,
        "source_sha256": _sha256_source(source_archive),
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "verdict": verdict,
        "verdict_label": verdict_label,
        "verdict_reason": verdict_reason,
        "decision_basis": decision_basis,
        "source_file_count": len(source_records),
        "source_languages": sorted({str(record["language"]) for record in source_records}),
        "warnings": warnings,
        "source_graph": _graph_summary(source_graph),
        "artifact_graph": _graph_summary(artifact_graph),
        "pe": {
            "machine": parsed.get("machine"),
            "section_count": parsed.get("section_count", 0),
            "import_library_count": len(parsed.get("imports") or []),
            "import_function_count": sum(
                len(library.get("functions") or [])
                for library in parsed.get("imports") or []
                if isinstance(library, dict)
            ),
            "high_entropy_sections": parsed.get("high_entropy_sections") or [],
            "overlay_bytes": parsed.get("overlay_bytes", 0),
        },
        "baseline": baseline,
        "model": model,
    }


def build_artifact_graph(
    parsed: dict[str, Any], artifact_name: str, artifact_payload: bytes
) -> dict[str, Any]:
    """Represent PE structure, imports, and selected strings as a bounded graph."""

    digest = hashlib.sha256(artifact_payload).hexdigest()
    root_id = f"binary:{digest}"
    nodes: list[dict[str, Any]] = [{
        "id": root_id,
        "type": "binary",
        "name": Path(artifact_name).name,
    }]
    edges: list[dict[str, str]] = []
    known = {root_id}
    for index, section in enumerate((parsed.get("sections") or [])[:96]):
        section_id = f"section:{index}:{section.get('name', '')}"
        nodes.append({
            "id": section_id,
            "type": "section",
            "name": str(section.get("name") or f"section-{index}"),
            "entropy": section.get("entropy", 0.0),
            "size": section.get("raw_size", 0),
            "suspicious": float(section.get("entropy", 0.0) or 0.0) >= 7.2,
        })
        known.add(section_id)
        edges.append({"source": root_id, "target": section_id, "type": "has_section"})
    for library_index, library in enumerate((parsed.get("imports") or [])[:256]):
        if not isinstance(library, dict):
            continue
        library_name = str(library.get("dll") or f"library-{library_index}")
        library_id = f"library:{library_index}:{_normalized_symbol(library_name)}"
        if library_id not in known:
            nodes.append({"id": library_id, "type": "import_library", "name": library_name})
            known.add(library_id)
        edges.append({"source": root_id, "target": library_id, "type": "imports"})
        for function_index, function in enumerate((library.get("functions") or [])[:512]):
            function_name = str(function)
            function_id = f"native:{library_index}:{function_index}:{_normalized_symbol(function_name)}"
            nodes.append({
                "id": function_id,
                "type": "native_api",
                "name": function_name,
                "suspicious": _normalized_symbol(function_name) in SUSPICIOUS_IMPORTS,
            })
            edges.append({"source": library_id, "target": function_id, "type": "imports"})
    for index, value in enumerate((parsed.get("suspicious_strings") or [])[:80]):
        signal_id = f"string:{index}:{hashlib.sha256(str(value).encode('utf-8', errors='ignore')).hexdigest()[:12]}"
        nodes.append({
            "id": signal_id,
            "type": "string_signal",
            "name": str(value)[:160],
            "suspicious": True,
        })
        edges.append({"source": root_id, "target": signal_id, "type": "exposes_string"})
    return {
        "graph_id": root_id,
        "modality": "artifact",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": sorted({str(node["type"]) for node in nodes}),
        "edge_types": sorted({str(edge["type"]) for edge in edges}),
        "nodes": nodes,
        "edges": edges,
    }


def _read_source_archive(
    source_archive: str | Path | bytes | BinaryIO,
    source_name: str,
) -> tuple[list[dict[str, str]], list[str]]:
    archive_source: Any
    if isinstance(source_archive, (str, Path)):
        archive_source = Path(source_archive)
    elif isinstance(source_archive, bytes):
        archive_source = BytesIO(source_archive)
    else:
        archive_source = source_archive
        try:
            archive_source.seek(0)
        except (AttributeError, OSError):
            pass
    records: list[dict[str, str]] = []
    warnings: list[str] = []
    total_bytes = 0
    try:
        with zipfile.ZipFile(archive_source) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_SOURCE_MEMBERS:
                raise GenomeAnalysisError(f"源码包成员超过 {MAX_SOURCE_MEMBERS} 个。")
            for member in members:
                name = member.filename.replace("\\", "/")
                path = PurePosixPath(name)
                mode = member.external_attr >> 16
                if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                    warnings.append(f"已跳过不安全路径：{member.filename}")
                    continue
                if member.flag_bits & 0x1:
                    warnings.append(f"已跳过加密文件：{member.filename}")
                    continue
                if (
                    Path(name).suffix.lower() not in SOURCE_EXTENSIONS
                    and not is_generic_text_path(name)
                ):
                    continue
                if len(records) >= MAX_SOURCE_FILES:
                    warnings.append(f"源码文件超过 {MAX_SOURCE_FILES} 个，剩余文件已跳过。")
                    break
                if member.file_size > MAX_SOURCE_MEMBER_BYTES:
                    warnings.append(f"已跳过超过 4 MB 的源码文件：{member.filename}")
                    continue
                if member.compress_size and member.file_size / member.compress_size > MAX_COMPRESSION_RATIO:
                    warnings.append(f"已跳过压缩比异常的源码文件：{member.filename}")
                    continue
                if total_bytes + member.file_size > MAX_SOURCE_TOTAL_BYTES:
                    warnings.append("源码解压总量超过 64 MB，剩余文件已跳过。")
                    break
                with archive.open(member) as stream:
                    payload = stream.read(MAX_SOURCE_MEMBER_BYTES + 1)
                if (
                    len(payload) > MAX_SOURCE_MEMBER_BYTES
                    or not is_probably_text_payload(payload)
                ):
                    warnings.append(f"已跳过非文本或超限文件：{member.filename}")
                    continue
                content = decode_source_payload(payload)
                analysis_content = _bounded_text(content, MAX_SOURCE_ANALYSIS_BYTES)
                records.append({
                    "path": name,
                    "content": analysis_content,
                    "language": detect_source_language(name, analysis_content),
                })
                total_bytes += len(payload)
    except (zipfile.BadZipFile, OSError) as exc:
        raise GenomeAnalysisError(f"无法读取源码 ZIP：{exc}") from exc
    return records, warnings[:100]


def _baseline_compare(
    source_records: list[dict[str, str]],
    parsed: dict[str, Any],
    artifact_payload: bytes,
) -> dict[str, Any]:
    claims = {"functions": set(), "dependencies": set(), "apis": set(), "literals": set()}
    for record in source_records:
        graph = build_lightweight_graph(record["content"], record["language"])
        claims["functions"].update(_claim_symbols(graph.get("functions") or []))
        claims["dependencies"].update(_claim_symbols(graph.get("imports") or []))
        claims["apis"].update(_claim_symbols(graph.get("dangerous_apis") or []))
        claims["literals"].update(_source_literals(record["content"]))

    imported_functions = {
        _normalized_symbol(function)
        for library in parsed.get("imports") or []
        if isinstance(library, dict)
        for function in library.get("functions") or []
        if _normalized_symbol(function)
    }
    imported_libraries = {
        _normalized_symbol(str(library.get("dll") or "").rsplit(".", 1)[0])
        for library in parsed.get("imports") or []
        if isinstance(library, dict)
    }
    binary_strings = [value for _, value in printable_strings(artifact_payload, minimum=5, limit=4000)]
    binary_text = "\n".join(binary_strings).lower()
    binary_symbols = {
        _normalized_symbol(value)
        for text in binary_strings
        for value in WORD_PATTERN.findall(text)
        if _normalized_symbol(value)
    }
    binary_symbols.update(imported_functions)
    binary_symbols.update(imported_libraries)

    matched: dict[str, list[str]] = {}
    matched["functions"] = sorted(value for value in claims["functions"] if value in binary_symbols)
    matched["dependencies"] = sorted(value for value in claims["dependencies"] if value in binary_symbols or value in imported_libraries)
    matched["apis"] = sorted(value for value in claims["apis"] if value in binary_symbols or value in imported_functions)
    matched["literals"] = sorted(value for value in claims["literals"] if value.lower() in binary_text)

    weights = {"functions": 0.25, "dependencies": 0.2, "apis": 0.35, "literals": 0.2}
    active_weight = sum(weights[name] for name, values in claims.items() if values)
    weighted = sum(
        weights[name] * len(matched[name]) / max(1, len(values))
        for name, values in claims.items()
        if values
    )
    similarity = round(100 * weighted / max(active_weight, 1e-9), 1) if active_weight else 0.0
    source_api_claims = set(claims["apis"])
    suspicious_imports = {
        value for value in imported_functions if value in SUSPICIOUS_IMPORTS
    }
    suspicious_imports.update(_normalized_symbol(value) for value in parsed.get("import_indicators") or [])
    unmatched_risk = sorted(value for value in suspicious_imports if value not in source_api_claims)
    matched_count = sum(len(values) for values in matched.values())
    if unmatched_risk:
        decision = "review"
        reason = "产物出现源码侧未观察到的高风险原生能力，需要核对构建链。"
    elif matched_count >= 3 and similarity >= 45:
        decision = "consistent"
        reason = "源码声明与产物可见符号形成多类交叉印证。"
    else:
        decision = "inconclusive"
        reason = "编译优化或符号剥离后可比证据不足，基线不作一致性断言。"
    evidence = [
        {"category": name, "value": value}
        for name in ("functions", "dependencies", "apis", "literals")
        for value in matched[name][:12]
    ][:32]
    return {
        "decision": decision,
        "similarity": similarity,
        "reason": reason,
        "matched_count": matched_count,
        "source_claim_count": sum(len(values) for values in claims.values()),
        "matched_evidence": evidence,
        "unmatched_risk_signals": unmatched_risk[:32],
        "claim_counts": {name: len(values) for name, values in claims.items()},
    }


def _run_twin_model(
    source_graph: dict[str, Any],
    artifact_graph: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    manifest_path = model_dir / "genome_twin_manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "unavailable",
            "mode": "baseline",
            "reason": "尚未安装通过发布门禁的孪生 GAT 权重。",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "mode": "baseline", "reason": f"模型清单不可读：{exc}"}
    if manifest.get("runtime_ready") is not True:
        return {
            "status": "unavailable",
            "mode": "baseline",
            "model_version": manifest.get("model_version"),
            "reason": "孪生 GAT 尚未通过独立测试集发布门禁。",
        }
    if int(manifest.get("feature_dimension") or 0) != _current_feature_dimension():
        return {"status": "unavailable", "mode": "baseline", "reason": "孪生 GAT 特征模式与当前运行时不兼容。"}
    files = manifest.get("files") or []
    if not isinstance(files, list):
        files = []
    weights_path = model_dir / str(files[0] if files else "genome_twin_classifier.pt")
    if not weights_path.is_file():
        return {"status": "unavailable", "mode": "baseline", "reason": "孪生 GAT 权重文件缺失。"}
    try:
        import torch
        from torch_geometric.data import Batch
        from .training.genome_features import EDGE_TYPES, feature_dimension, graph_to_pyg
        from .training.genome_twin_model import TwinGATVerifier

        training = manifest.get("training") or {}
        if not isinstance(training, dict):
            raise ValueError("invalid training configuration")
        model = TwinGATVerifier(
            feature_dimension(),
            len(EDGE_TYPES),
            hidden=int(training.get("hidden", 96)),
            heads=int(training.get("heads", 4)),
            dropout=float(training.get("dropout", 0.2)),
            embedding_dim=int(training.get("embedding_dim", 96)),
        )
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        source_data = graph_to_pyg(source_graph, "source", torch)
        artifact_data = graph_to_pyg(artifact_graph, "artifact", torch)
        with torch.no_grad():
            logit = model(
                Batch.from_data_list([source_data]),
                Batch.from_data_list([artifact_data]),
            )
            probability = float(torch.sigmoid(logit)[0].item())
        threshold = float(manifest.get("threshold", 0.5))
        return {
            "status": "completed",
            "mode": "twin_gat",
            "decision": "review" if probability >= threshold else "consistent",
            "tamper_probability": round(probability, 6),
            "threshold": threshold,
            "model_version": manifest.get("model_version"),
        }
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError) as exc:
        return {"status": "unavailable", "mode": "baseline", "reason": f"孪生 GAT 运行失败：{exc}"}


def _current_feature_dimension() -> int:
    from .training.genome_features import feature_dimension

    return feature_dimension()


def _claim_symbols(values: list[Any]) -> set[str]:
    result = set()
    for raw in values:
        text = str(raw or "")
        candidates = [text, *re.split(r"[./:$@?\\-]+", text)]
        for candidate in candidates:
            normalized = _normalized_symbol(candidate)
            if len(normalized) >= 4 and normalized not in GENERIC_SYMBOLS:
                result.add(normalized)
    return result


def _source_literals(content: str) -> set[str]:
    values = set()
    for match in STRING_PATTERN.finditer(content[:MAX_SOURCE_ANALYSIS_BYTES]):
        value = match.group("value").strip()
        if (
            5 <= len(value) <= 96
            and not value.isspace()
            and not value.startswith(("http://", "https://"))
            and sum(character.isalnum() for character in value) >= 4
        ):
            values.add(value.lower())
        if len(values) >= 120:
            break
    return values


def _normalized_symbol(value: object) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(value or "").strip().lower())


def _bounded_text(content: str, maximum_bytes: int) -> str:
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return content
    marker = "\n/* ... software-genome middle omitted ... */\n"
    budget = max(2, maximum_bytes - len(marker.encode("utf-8")))
    head = encoded[:budget // 2].decode("utf-8", errors="ignore")
    tail = encoded[-(budget - budget // 2):].decode("utf-8", errors="ignore")
    return head + marker + tail


def _graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_count": int(graph.get("node_count") or len(graph.get("nodes") or [])),
        "edge_count": int(graph.get("edge_count") or len(graph.get("edges") or [])),
        "node_types": list(graph.get("node_types") or []),
        "edge_types": list(graph.get("edge_types") or []),
    }


def _sha256_source(source: str | Path | bytes | BinaryIO) -> str:
    if isinstance(source, bytes):
        return hashlib.sha256(source).hexdigest()
    if isinstance(source, (str, Path)):
        digest = hashlib.sha256()
        with Path(source).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    try:
        position = source.tell()
        source.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        source.seek(position)
        return digest.hexdigest()
    except (AttributeError, OSError):
        return ""
