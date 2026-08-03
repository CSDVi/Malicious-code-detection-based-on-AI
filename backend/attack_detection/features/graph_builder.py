"""Feature extraction helpers."""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from collections import Counter
from typing import Iterable

from attack_detection.dataset import CodeSample


LEXICAL_BUCKETS = 64
PROJECT_RELATIONSHIP_NODE_LIMIT = 22
PROJECT_RELATIONSHIP_EDGE_LIMIT = 48


def build_lightweight_graph(content: str, language: str) -> dict[str, object]:
    imports = _imports_and_dependencies(content, language)
    functions = _functions(content)
    dangerous = _dangerous_apis(content)
    nodes = set(["file"])
    nodes.update(f"import:{item}" for item in imports)
    nodes.update(f"function:{item}" for item in functions)
    nodes.update(f"api:{item.lower()}" for item in dangerous)
    edge_count = len(imports) + len(functions) + len(dangerous)
    return {
        "language": language,
        "node_count": len(nodes),
        "edge_count": edge_count,
        "imports": imports[:20],
        "functions": functions[:20],
        "dangerous_apis": sorted(set(dangerous)),
    }


def build_project_graph(samples: Iterable[CodeSample]) -> dict[str, object]:
    records = list(samples)
    if not records:
        raise ValueError("project graph requires at least one file")

    first = records[0]
    package_id = f"package:{first.family or first.package_name}:{first.version}"
    nodes = [
        {
            "id": package_id,
            "type": "package",
            "name": first.package_name or first.family,
            "version": first.version,
        }
    ]
    edges: list[dict[str, object]] = []
    known_nodes = {package_id}

    for sample in records[:200]:
        file_id = f"file:{sample.sample_hash}"
        _node(nodes, known_nodes, {
            "id": file_id,
            "type": "file",
            "name": sample.file_path or sample.sample_hash[:12],
            "language": sample.language,
            "label": sample.label,
            "line_labels": list(sample.line_labels),
            "lexical_buckets": _source_token_buckets(sample.code),
        })
        edges.append({"source": package_id, "target": file_id, "type": "contains"})

        for name in _functions(sample.code)[:50]:
            function_id = f"function:{sample.sample_hash}:{name}"
            _node(nodes, known_nodes, {
                "id": function_id,
                "type": "function",
                "name": name,
            })
            edges.append({"source": file_id, "target": function_id, "type": "declares"})

        for dependency in _imports_and_dependencies(sample.code, sample.language)[:50]:
            dependency_id = f"package:dependency:{dependency}"
            _node(nodes, known_nodes, {
                "id": dependency_id,
                "type": "package",
                "name": dependency,
            })
            edges.append({"source": file_id, "target": dependency_id, "type": "import"})
            edges.append({"source": package_id, "target": dependency_id, "type": "dependency"})

        semantic_signals = list(dict.fromkeys(
            _dangerous_apis(sample.code) + _context_signals(sample.file_path, sample.code)
        ))
        for api in semantic_signals[:30]:
            api_id = f"api:{api}"
            _node(nodes, known_nodes, {
                "id": api_id,
                "type": "dangerous_api",
                "name": api,
            })
            edges.append({"source": file_id, "target": api_id, "type": "call"})

        if sample.paired_version:
            prior_id = f"package:{sample.family or sample.package_name}:{sample.paired_version}"
            _node(nodes, known_nodes, {
                "id": prior_id,
                "type": "package",
                "name": sample.package_name or sample.family,
                "version": sample.paired_version,
                "role": "clean_pair",
            })
            edges.append({"source": package_id, "target": prior_id, "type": "version_diff"})

    label_counts = Counter(sample.label for sample in records)
    graph_label = max(label_counts, key=lambda label: label_counts[label])
    return {
        "graph_id": package_id,
        "family": first.family,
        "package_name": first.package_name,
        "version": first.version,
        "split": first.split,
        "label": graph_label,
        "source": first.source,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": sorted({str(node["type"]) for node in nodes}),
        "edge_types": sorted({str(edge["type"]) for edge in edges}),
        "nodes": nodes,
        "edges": edges,
    }


def build_project_relationship_graph(
    samples: Iterable[CodeSample],
    risk_by_path: dict[str, dict[str, object]] | None = None,
    *,
    node_limit: int = PROJECT_RELATIONSHIP_NODE_LIMIT,
    edge_limit: int = PROJECT_RELATIONSHIP_EDGE_LIMIT,
) -> dict[str, object]:
    """Build a bounded, report-only graph of resolved local file references.

    This graph is deliberately separate from the GATv2 feature graph.  It only
    draws an edge when an import/include/source reference resolves to one
    concrete file in the uploaded project, so the report never invents calls.
    """

    records = list(samples)[:200]
    normalized_records = [
        (_normalize_project_path(sample.file_path or sample.sample_hash), sample)
        for sample in records
    ]
    normalized_records = [item for item in normalized_records if item[0]]
    project_paths = [path for path, _sample in normalized_records]
    path_lookup = {path.casefold(): path for path in project_paths}
    risk_lookup = {
        _normalize_project_path(path).casefold(): dict(value)
        for path, value in (risk_by_path or {}).items()
        if _normalize_project_path(path) and isinstance(value, dict)
    }

    relationships: set[tuple[str, str, str]] = set()
    for source_path, sample in normalized_records:
        for relation, reference in _local_file_references(
            sample.code,
            sample.language,
        ):
            target_path = _resolve_project_reference(
                source_path,
                reference,
                sample.language,
                project_paths,
                path_lookup,
            )
            if target_path and target_path != source_path:
                relationships.add((source_path, target_path, relation))

    degree = Counter()
    for source, target, _relation in relationships:
        degree.update((source, target))

    def risk_score(path: str) -> int:
        value = risk_lookup.get(path.casefold(), {}).get("risk_score")
        try:
            return max(0, min(100, int(float(value or 0))))
        except (TypeError, ValueError):
            return 0

    prioritized_edges = sorted(
        relationships,
        key=lambda item: (
            -max(risk_score(item[0]), risk_score(item[1])),
            -(degree[item[0]] + degree[item[1]]),
            item[0].casefold(),
            item[1].casefold(),
            item[2],
        ),
    )
    selected_paths: set[str] = set()
    selected_edges: list[tuple[str, str, str]] = []
    for source, target, relation in prioritized_edges:
        new_paths = {source, target} - selected_paths
        if len(selected_paths) + len(new_paths) > max(2, int(node_limit)):
            continue
        selected_paths.update((source, target))
        selected_edges.append((source, target, relation))
        if len(selected_edges) >= max(1, int(edge_limit)):
            break

    for path in sorted(
        project_paths,
        key=lambda item: (-risk_score(item), -degree[item], item.casefold()),
    ):
        if len(selected_paths) >= max(1, int(node_limit)):
            break
        selected_paths.add(path)

    selected = sorted(
        selected_paths,
        key=lambda item: (-degree[item], -risk_score(item), item.casefold()),
    )
    node_ids = {path: f"file-{index + 1}" for index, path in enumerate(selected)}
    nodes = []
    for path in selected:
        risk = risk_lookup.get(path.casefold(), {})
        nodes.append({
            "id": node_ids[path],
            "path": path,
            "name": posixpath.basename(path) or path,
            "language": str(risk.get("language") or "unknown"),
            "risk_score": risk_score(path),
            "risk_level": str(risk.get("risk_level") or "unknown"),
            "degree": degree[path],
        })
    edges = [
        {
            "source": node_ids[source],
            "target": node_ids[target],
            "relation": relation,
        }
        for source, target, relation in selected_edges
        if source in node_ids and target in node_ids
    ]
    return {
        "node_count": len(project_paths),
        "edge_count": len(relationships),
        "displayed_node_count": len(nodes),
        "displayed_edge_count": len(edges),
        "truncated": len(nodes) < len(project_paths) or len(edges) < len(relationships),
        "nodes": nodes,
        "edges": edges,
    }


def _node(nodes: list[dict[str, object]], known: set[str], value: dict[str, object]) -> None:
    identifier = str(value["id"])
    if identifier in known:
        return
    known.add(identifier)
    nodes.append(value)


def _normalize_project_path(value: str) -> str:
    normalized = posixpath.normpath(str(value or "").replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return "" if normalized in {"", ".", ".."} else normalized.lstrip("/")


def _local_file_references(content: str, language: str) -> list[tuple[str, str]]:
    language = str(language or "").lower()
    patterns: list[tuple[str, str]] = []
    if language == "python":
        patterns.extend((
            ("import", r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\b"),
            ("import", r"^\s*import\s+([A-Za-z0-9_.]+)"),
        ))
    if language in {"javascript", "typescript"}:
        patterns.extend((
            ("import", r"(?:import|export)[^;\n]*?\bfrom\s*['\"]([^'\"]+)['\"]"),
            ("import", r"^\s*import\s*['\"]([^'\"]+)['\"]"),
            ("require", r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
            ("import", r"\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        ))
    if language in {"c", "cpp"}:
        patterns.append(("include", r"^\s*#\s*include\s*\"([^\"]+)\""))
    if language in {"php", "ruby"}:
        patterns.extend((
            ("include", r"\b(?:include|include_once|require|require_once)\s*\(?\s*['\"]([^'\"]+)['\"]"),
            ("require", r"\brequire_relative\s*\(?\s*['\"]([^'\"]+)['\"]"),
        ))
    if language in {"java", "kotlin", "scala"}:
        patterns.append(("import", r"^\s*import\s+([A-Za-z0-9_.]+)\s*;?"))
    if language in {"bash", "powershell"}:
        patterns.extend((
            ("source", r"^\s*(?:source|\.)\s+['\"]?([^'\"\s;]+)"),
            ("import", r"^\s*Import-Module\s+['\"]?([^'\"\s;]+)"),
        ))
    if language == "html":
        patterns.append(("load", r"<script\b[^>]*\bsrc\s*=\s*['\"]([^'\"]+)['\"]"))

    references = []
    for relation, pattern in patterns:
        for value in re.findall(pattern, content, re.IGNORECASE | re.MULTILINE):
            reference = str(value or "").strip()
            if reference:
                references.append((relation, reference))
    return list(dict.fromkeys(references))


def _resolve_project_reference(
    source_path: str,
    reference: str,
    language: str,
    project_paths: list[str],
    path_lookup: dict[str, str],
) -> str | None:
    raw = str(reference or "").strip().replace("\\", "/")
    raw = raw.split("?", 1)[0].split("#", 1)[0]
    if (
        not raw
        or raw.startswith(("http://", "https://", "//"))
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw)
    ):
        return None

    source_dir = posixpath.dirname(source_path)
    candidates: list[str] = []
    if language == "python" and raw.startswith("."):
        leading = len(raw) - len(raw.lstrip("."))
        base = source_dir
        for _index in range(max(0, leading - 1)):
            base = posixpath.dirname(base)
        module = raw[leading:].replace(".", "/")
        candidates.append(posixpath.join(base, module))
    else:
        candidates.extend((
            posixpath.join(source_dir, raw),
            raw.lstrip("/"),
        ))
        if "." in raw and "/" not in raw:
            module = raw.replace(".", "/")
            candidates.extend((module, posixpath.join(source_dir, module)))

    extensions = (
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".php", ".rb", ".sh",
        ".bash", ".ps1", ".psm1", ".html", ".htm",
    )
    expanded: list[str] = []
    for candidate in candidates:
        normalized = _normalize_project_path(candidate)
        if not normalized:
            continue
        expanded.append(normalized)
        if not posixpath.splitext(normalized)[1]:
            expanded.extend(normalized + extension for extension in extensions)
            expanded.extend(
                posixpath.join(normalized, "index" + extension)
                for extension in (".js", ".jsx", ".ts", ".tsx", ".py")
            )

    for candidate in dict.fromkeys(expanded):
        exact = path_lookup.get(candidate.casefold())
        if exact:
            return exact

    suffix_matches = {
        path
        for candidate in expanded
        for path in project_paths
        if path.casefold().endswith("/" + candidate.casefold())
    }
    if len(suffix_matches) == 1:
        return next(iter(suffix_matches))

    reference_name = posixpath.basename(raw)
    reference_stem = posixpath.splitext(reference_name)[0].casefold()
    if reference_stem:
        stem_matches = {
            path
            for path in project_paths
            if posixpath.splitext(posixpath.basename(path))[0].casefold()
            == reference_stem
        }
        if len(stem_matches) == 1:
            return next(iter(stem_matches))
    return None


def _functions(code: str) -> list[str]:
    patterns = (
        r"\b(?:def|function|func|fun)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\b(?:public|private|protected|internal|static|async|virtual|override|final)\s+(?:[A-Za-z_$][A-Za-z0-9_$<>\[\],.?]*\s+)+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
        r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{",
    )
    return list(dict.fromkeys(
        name
        for pattern in patterns
        for name in re.findall(pattern, code, re.MULTILINE)
    ))


def _dangerous_apis(code: str) -> list[str]:
    values = {value.lower() for value in re.findall(
        r"(?<![A-Za-z0-9_])(eval|exec|system|popen|subprocess|child_process|shell_exec|passthru|proc_open|pcntl_exec|assert|processbuilder|runtime\.getruntime|os\.exec|requests\.get|urllib\.request|webclient|fetch|curl|wget|socket|base64_decode|gzinflate|gzuncompress|str_rot13|preg_replace|create_function|rawurldecode|urldecode|hex2bin|convert_uudecode|call_user_func|call_user_func_array|file_put_contents|file_get_contents|fwrite|fopen|readfile|move_uploaded_file|chmod|unlink|rename|copy|scandir|glob|opendir|readdir|fsockopen|stream_socket_client|php://input|\$_post|\$_get|\$_request|\$_cookie|\$_files|\$_server|\$_env|invoke-expression|iex|start-process|new-object|downloadstring|downloadfile|invoke-webrequest|invoke-restmethod|frombase64string|encodedcommand|add-mppreference|set-mppreference|register-scheduledtask|schtasks|certutil|bitsadmin|mshta|rundll32|regsvr32|wmic|cmd|powershell|bash|sh|netcat|nc|chattr|crontab|nohup|winexec|shellexecute|createprocess|virtualalloc|virtualprotect|writeprocessmemory|createremotethread|loadlibrary|getprocaddress|internetopenurl|urldownloadtofile|winhttpopen|regsetvalue|openprocess|adjusttokenprivileges|process\.start|assembly\.load|powershell\.create|registry\.setvalue|dllimport|connect|recv|send)(?![A-Za-z0-9_])",
        code,
        re.IGNORECASE,
    )}
    execution = {
        "eval", "exec", "system", "popen", "subprocess", "child_process",
        "shell_exec", "passthru", "proc_open", "pcntl_exec", "assert",
        "processbuilder", "runtime.getruntime", "os.exec", "requests.get",
        "urllib.request", "webclient", "fetch", "curl", "wget", "iex",
        "start-process", "new-object", "downloadstring", "downloadfile",
        "invoke-webrequest", "invoke-restmethod", "frombase64string",
        "encodedcommand", "create_function", "call_user_func",
        "call_user_func_array", "powershell", "bash", "sh", "cmd",
        "winexec", "shellexecute", "createprocess", "process.start",
    }
    inputs = {"$_post", "$_get", "$_request", "$_cookie", "$_files", "$_server", "$_env", "php://input"}
    decoders = {"base64_decode", "frombase64string", "gzinflate", "hex2bin", "convert_uudecode", "certutil", "rawurldecode", "str_rot13", "gzuncompress", "urldecode"}
    writes = {"bitsadmin", "move_uploaded_file", "downloadfile", "copy", "unlink", "fopen", "fwrite", "rename", "file_put_contents"}
    remote_load = {"invoke-webrequest", "bitsadmin", "winhttpopen", "downloadfile", "invoke-restmethod", "internetopenurl", "urldownloadtofile", "fsockopen", "curl", "socket", "wget", "downloadstring", "fetch", "stream_socket_client"}

    if values.intersection(inputs) and values.intersection(execution):
        values.add("behavior_input_execution_chain")
    if values.intersection(decoders) and values.intersection(execution):
        values.add("behavior_decode_execution_chain")
    if values.intersection(inputs) and values.intersection(writes):
        values.add("behavior_input_file_write_chain")
    if values.intersection(remote_load) and values.intersection(execution):
        values.add("behavior_remote_execution_chain")
    if {"writeprocessmemory", "createremotethread", "virtualalloc"}.issubset(values):
        values.add("behavior_process_injection_chain")
    if "encodedcommand" not in values and "frombase64string" in values and values.intersection(execution):
        values.add("behavior_encoded_command")
    if values.intersection({"add-mppreference", "set-mppreference"}) and values.intersection(execution):
        values.add("behavior_defense_evasion_chain")
    if values.intersection({"schtasks", "register-scheduledtask", "crontab"}) and values.intersection(execution):
        values.add("behavior_persistence_chain")
    if (
        re.search(r"[A-Za-z0-9+/]{240,}={0,2}", code)
        or re.search(r"(?:\\x[0-9a-fA-F]{2}){24,}", code)
        or re.search(r"(?:%[0-9a-fA-F]{2}){32,}", code)
    ):
        values.add("behavior_encoded_payload")
    if re.search(r"\$[A-Za-z_][A-Za-z0-9_]*\s*\(", code):
        values.add("behavior_variable_function_call")

    lowered = code.lower()
    if re.search(r"\b(?:reverse[\s_-]*shell|bind[\s_-]*shell|meterpreter|interactive[\s_-]*shell|shellcode)\b", lowered):
        values.add("behavior_reverse_shell")
    if re.search(r"\b(?:credential(?:s)?|creds|password|passwd|mimikatz|lsass|kerberos|requestfakedelegticket|hashdump)\b", lowered):
        values.add("behavior_credential_access")
    if re.search(r"\b(?:phish(?:ing)?|gophish|socialfish|hiddeneye|blackeye|credsniper)\b", lowered):
        values.add("behavior_phishing")
    if re.search(r"\b(?:anti[\s_-]?virus|antivirus|defender|amsi|edr|security[\s_-]*bypass|bypass[\s_-]*(?:security|detection))\b", lowered):
        values.add("behavior_security_evasion")
    if re.search(r"\buac[\s_-]*bypass\b", lowered) or re.search(r"\b(?:fodhelper|eventvwr|sdclt|computerdefaults)\b", lowered):
        values.add("behavior_uac_bypass")
    if re.search(r"\b(?:command[\s_-]*(?:and|&)[\s_-]*control|c2[\s_-]*server|c&c|cnc|beacon(?:ing)?)\b", lowered):
        values.add("behavior_command_and_control")
    if re.search(r"(?<![A-Za-z0-9+/])TVqQ[A-Za-z0-9+/]{400,}={0,2}", code) or re.search(r"(?:\\x4d\\x5a|\\x5a\\x4d)(?:\\x[0-9a-fA-F]{2}){64,}", code):
        values.add("behavior_embedded_executable")
    if re.search(r"\b(?:hping3|packet[\s_-]*flood|syn[\s_-]*flood|ip[\s_-]*flood|ddos|denial[\s_-]*of[\s_-]*service)\b", lowered):
        values.add("behavior_network_flood")
    if re.search(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:", code) or re.search(r"\binfinite number of processes\b", lowered):
        values.add("behavior_fork_bomb")
    if re.search(r"\b(?:curl|wget)\b[^\n|]{0,240}\|\s*(?:/usr/bin/)?(?:ba)?sh\b", lowered):
        values.add("behavior_download_execute_pipe")

    return sorted(values)


def _context_signals(file_path: str, code: str) -> list[str]:
    normalized_path = str(file_path or "").replace("\\", "/").lower()
    base_name = normalized_path.rsplit("/", 1)[-1]
    lowered = code.lower()
    values: set[str] = set()

    if re.search(r"(?:reverse|rev)[_-]?shell", normalized_path):
        values.add("behavior_reverse_shell")

    if (
        re.search(r"^(?:ci|ci[_-]build|build|build[_-]via[_-]cmake|test|test[_-]script|entrypoint)\.(?:sh|bash|cmd|bat|ps1)$", base_name)
        or "/.github/workflows/" in f"/{normalized_path}"
        or (re.search(r"\b(?:cmake|pytest|docker\s+build|pip\s+install)\b", lowered) and re.search(r"\b(?:build|test|ci|pipeline)\b", lowered))
    ):
        values.add("context_ci_or_build")

    if "redistribution and use in source and binary forms" in lowered and re.search(r"\busage\s*:", lowered):
        values.add("context_documented_system_utility")

    return sorted(values)


def _source_token_buckets(code: str) -> list[float]:
    counts = [0] * LEXICAL_BUCKETS
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{1,}|--?[A-Za-z][A-Za-z0-9_-]*", code.lower())
    for token in tokens[:12000]:
        bucket = int(hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % LEXICAL_BUCKETS
        counts[bucket] += 1
    maximum = max(counts, default=0)
    if maximum <= 0:
        return [0.0] * LEXICAL_BUCKETS
    scale = math.log1p(maximum)
    return [math.log1p(value) / scale for value in counts]


def _imports_and_dependencies(code: str, language: str) -> list[str]:
    patterns = (
        r"^\s*(?:from|import|using|use)\s+([A-Za-z0-9_./:@-]+)",
        r"require\s*\(\s*['\"]([^'\"]+)",
        r"(?:include|require|require_once|include_once)\s*\(?\s*['\"]([^'\"]+)",
        r"^\s*#\s*include\s*[<\"]([^>\"]+)",
        r"^\s*import\s*[\"`]([^\"`]+)[\"`]",
        r"^\s*Import-Module\s+['\"]?([A-Za-z0-9_.:/@-]+)",
        r"^\s*(?:source|\.)\s+['\"]?([A-Za-z0-9_./@-]+)",
    )
    output: list[str] = []
    for pattern in patterns:
        for value in re.findall(pattern, code, re.MULTILINE):
            if value:
                output.append(value.split(".")[0])

    if language == "config":
        try:
            manifest = json.loads(code)
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                if isinstance(manifest.get(section), dict):
                    output.extend(str(name) for name in manifest[section])
        except (json.JSONDecodeError, AttributeError):
            pass

    return sorted({value for value in output if value})
