"""Resolved cross-file call graph and bounded source-to-sink analysis.

The production GATv2 artifacts were trained on the historical project graph.
This module therefore keeps the richer resolved graph and taint evidence in a
separate, explainable channel until a supplemental GATv2 candidate passes the
existing per-language release gates.  No source code is executed.
"""

from __future__ import annotations

import ast
import hashlib
import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


MAX_ANALYSIS_FILES = 200
MAX_FUNCTIONS = 4000
MAX_CALLS = 12000
MAX_TAINT_ROUNDS = 12
MAX_CHAINS = 80

_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_CALL = re.compile(
    r"(?<![A-Za-z0-9_$])([A-Za-z_$][A-Za-z0-9_$.:-]{0,160})\s*\("
)
_ASSIGN = re.compile(
    r"^\s*(?:const\s+|let\s+|var\s+|my\s+|local\s+|"
    r"[A-Za-z_$][A-Za-z0-9_$<>,.?\[\]]*\s+)?"
    r"((?:this\.|self\.)?[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\+|-|\*|/|\|)?=\s*(.+)$"
)

_SOURCE_TOKENS = (
    "os.getenv", "environ.get", "getenv", "getpass", "keyring.get_password",
    "browser_cookie3", "cookie", "credential", "password", "passwd", "secret",
    "token", "api_key", "queryvalueex", "winreg", ".ssh", "requests.get",
    "request.get", "urlopen", "fetch", "downloadstring", "downloadfile",
    "invoke-webrequest", "invoke-restmethod", "socket.recv", ".recv", "stdin",
    "request.args", "request.form", "request.json", "$_post", "$_get", "$_request",
    "calllog.calls", "content://sms", "contactscontract", "browser.bookmarkcolumns",
    "telephonymanager.extra_incoming_number", "telephonymanager.extra_phone_number",
)
_TRANSFORM_TOKENS = (
    "base64", "b64encode", "b64decode", "decode", "encode", "hexlify",
    "unhexlify", "json.dumps", "json.loads", "serialize", "marshal", "pickle",
    "pack", "compress", "decompress", "gzip", "zlib", "encrypt", "decrypt",
    "cipher", "archive", "tar", "zip", "join", "format_payload", "prepare_payload",
    "getbytes",
)
_SINK_TOKENS = (
    "requests.post", "requests.put", "session.post", "http.post", "httpclient.post",
    "webhook", "discord", "upload", "exfil", "sendall", "socket.send", ".send",
    "storbinary", "put_object", "invoke-restmethod", "invoke-webrequest", "curl",
    "wget", "urlopen", "fetch", "postasync", "sendasync", "net.http.post",
)
_CALL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "typeof",
    "function", "def", "class", "new", "print", "echo", "with", "assert",
}


@dataclass(frozen=True)
class SourceUnit:
    path: str
    content: str
    language: str
    rows: tuple[str, ...]


@dataclass
class FunctionInfo:
    identifier: str
    path: str
    name: str
    qualified_name: str
    line: int
    end_line: int
    parameters: tuple[str, ...]
    language: str


@dataclass
class ImportBinding:
    path: str
    module: str
    symbol: str | None
    alias: str
    level: int = 0
    relation: str = "import"


@dataclass
class Event:
    kind: str
    path: str
    function_id: str
    line: int
    snippet: str
    target: str | None = None
    names: tuple[str, ...] = ()
    callee: str | None = None
    arguments: tuple[tuple[str, ...], ...] = ()
    stage: str | None = None
    resolved_target: str | None = None
    resolution: str | None = None
    confidence: float = 0.0


@dataclass
class ParsedProject:
    units: list[SourceUnit]
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    imports: list[ImportBinding] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    parse_errors: list[dict[str, object]] = field(default_factory=list)


def analyze_cross_file_project(
    records: Iterable[object],
    *,
    max_files: int = MAX_ANALYSIS_FILES,
) -> dict[str, object]:
    """Return a JSON-ready call graph, taint chains and component ranking."""

    parsed = _parse_project(records, max_files=max_files)
    relationships = _resolve_imports(parsed)
    call_edges = _resolve_calls(parsed, relationships)
    chains = _trace_taint(parsed)
    complete = [chain for chain in chains if chain.get("complete")]
    ranking = _rank_components(chains, call_edges)
    languages = sorted({unit.language for unit in parsed.units})
    return {
        "schema_version": 1,
        "status": "completed",
        "analysis_method": "python_ast_plus_multilanguage_static_resolution",
        "execution_policy": "static_only_never_execute_uploaded_code",
        "file_count": len(parsed.units),
        "language_count": len(languages),
        "languages": languages,
        "call_graph": _shape_call_graph(parsed, relationships, call_edges),
        "taint_chains": chains,
        "complete_chains": complete,
        "complete_chain_count": len(complete),
        "component_ranking": ranking,
        "most_suspicious_component": ranking[0] if ranking else None,
        "findings": [_chain_finding(chain) for chain in complete],
        "parse_errors": parsed.parse_errors[:50],
        "limitations": [
            "Python uses AST resolution; other languages currently use bounded static symbol resolution.",
            "Reflection, dynamic imports, generated code, runtime dispatch and native aliases may be unresolved.",
            "A reported chain is static evidence and still requires analyst review.",
        ],
    }


def _parse_project(records: Iterable[object], *, max_files: int) -> ParsedProject:
    units: list[SourceUnit] = []
    for raw in list(records)[:max(1, int(max_files))]:
        path = _normalize_path(_value(raw, "filename") or _value(raw, "file_path"))
        content = str(_value(raw, "content") or _value(raw, "code") or "")
        language = str(_value(raw, "language") or "unknown").lower()
        if not path or not content or language == "binary":
            continue
        units.append(SourceUnit(path, content, language, tuple(content.splitlines())))
    parsed = ParsedProject(units=units)
    for unit in units:
        module = _module_function(unit)
        parsed.functions[module.identifier] = module
        if unit.language == "python":
            _parse_python(unit, parsed)
        else:
            _parse_generic(unit, parsed)
        if len(parsed.functions) >= MAX_FUNCTIONS or len(parsed.events) >= MAX_CALLS:
            break
    return parsed


def _parse_python(unit: SourceUnit, parsed: ParsedProject) -> None:
    try:
        tree = ast.parse(unit.content, filename=unit.path)
    except (SyntaxError, ValueError) as exc:
        parsed.parse_errors.append({
            "path": unit.path,
            "language": unit.language,
            "error": f"python_ast_parse_failed: {exc}",
        })
        _parse_generic(unit, parsed)
        return
    extractor = _PythonExtractor(unit, parsed)
    extractor.visit(tree)


class _PythonExtractor(ast.NodeVisitor):
    def __init__(self, unit: SourceUnit, parsed: ParsedProject) -> None:
        self.unit = unit
        self.parsed = parsed
        self.stack = [_module_id(unit.path)]
        self.qualified: list[str] = []
        self.call_targets: dict[int, str] = {}

    @property
    def current_function(self) -> str:
        return self.stack[-1]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.parsed.imports.append(ImportBinding(
                path=self.unit.path,
                module=alias.name,
                symbol=None,
                alias=alias.asname or alias.name.split(".", 1)[0],
            ))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                continue
            self.parsed.imports.append(ImportBinding(
                path=self.unit.path,
                module=module,
                symbol=alias.name,
                alias=alias.asname or alias.name,
                level=int(node.level or 0),
            ))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join((*self.qualified, node.name))
        identifier = _function_id(self.unit.path, qualified, int(node.lineno))
        parameters = tuple(
            argument.arg
            for argument in (
                *getattr(node.args, "posonlyargs", []),
                *node.args.args,
                *node.args.kwonlyargs,
            )
        )
        self.parsed.functions[identifier] = FunctionInfo(
            identifier=identifier,
            path=self.unit.path,
            name=node.name,
            qualified_name=qualified,
            line=int(node.lineno),
            end_line=int(getattr(node, "end_lineno", node.lineno) or node.lineno),
            parameters=parameters,
            language="python",
        )
        self.stack.append(identifier)
        self.qualified.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.qualified.pop()
        self.stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        target = _python_target_name(node.targets[0]) if node.targets else None
        if target and isinstance(node.value, ast.Call):
            self.call_targets[id(node.value)] = target
        self._assignment_event(node, target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = _python_target_name(node.target)
        if target and isinstance(node.value, ast.Call):
            self.call_targets[id(node.value)] = target
        if node.value is not None:
            self._assignment_event(node, target, node.value)
        self.generic_visit(node)

    def _assignment_event(self, node: ast.AST, target: str | None, value: ast.AST) -> None:
        if not target:
            return
        callee = _python_call_name(value) if isinstance(value, ast.Call) else ""
        stage = _classify_stage(callee, _source_segment(self.unit, node))
        self.parsed.events.append(Event(
            kind="transform" if stage == "transform" else "source" if stage == "source" else "assign",
            path=self.unit.path,
            function_id=self.current_function,
            line=int(getattr(node, "lineno", 1) or 1),
            snippet=_line(self.unit, int(getattr(node, "lineno", 1) or 1)),
            target=target,
            names=tuple(sorted(_python_names(value) - {callee.split(".", 1)[0]})),
            callee=callee or None,
            stage=stage if stage in {"source", "transform"} else None,
            confidence=0.94,
        ))

    def visit_Call(self, node: ast.Call) -> None:
        callee = _python_call_name(node)
        stage = _classify_stage(callee, _source_segment(self.unit, node))
        arguments = tuple(
            tuple(sorted(_python_names(argument)))
            for argument in (*node.args, *(keyword.value for keyword in node.keywords))
        )
        self.parsed.events.append(Event(
            kind="sink" if stage == "sink" else "call",
            path=self.unit.path,
            function_id=self.current_function,
            line=int(getattr(node, "lineno", 1) or 1),
            snippet=_line(self.unit, int(getattr(node, "lineno", 1) or 1)),
            target=self.call_targets.get(id(node)),
            names=tuple(sorted({name for values in arguments for name in values})),
            callee=callee or None,
            arguments=arguments,
            stage=stage,
            confidence=0.96,
        ))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.parsed.events.append(Event(
            kind="return",
            path=self.unit.path,
            function_id=self.current_function,
            line=int(getattr(node, "lineno", 1) or 1),
            snippet=_line(self.unit, int(getattr(node, "lineno", 1) or 1)),
            names=tuple(sorted(_python_names(node.value))) if node.value else (),
            confidence=0.96,
        ))
        self.generic_visit(node)


def _parse_generic(unit: SourceUnit, parsed: ParsedProject) -> None:
    definitions = _generic_definitions(unit)
    for function in definitions:
        parsed.functions.setdefault(function.identifier, function)
    for relation, reference, symbol, alias, line in _generic_imports(unit):
        parsed.imports.append(ImportBinding(
            path=unit.path,
            module=reference,
            symbol=symbol,
            alias=alias,
            relation=relation,
        ))
    for line_number, row in enumerate(unit.rows, 1):
        function = _enclosing_function(unit.path, line_number, parsed.functions)
        function_id = function.identifier if function else _module_id(unit.path)
        assignment = _ASSIGN.match(row)
        target = assignment.group(1) if assignment else None
        expression = assignment.group(2) if assignment else row
        names = tuple(sorted(set(_IDENTIFIER.findall(expression))))
        calls = [
            value for value in _CALL.findall(expression)
            if value.lower().split(".")[-1] not in _CALL_KEYWORDS
            and not (
                function is not None
                and function.line == line_number
                and value.replace("::", ".").rsplit(".", 1)[-1]
                == function.name
            )
        ]
        primary = calls[0] if calls else ""
        stage = _classify_stage(primary, row)
        if target:
            parsed.events.append(Event(
                kind="source" if stage == "source" else "transform" if stage == "transform" else "assign",
                path=unit.path, function_id=function_id, line=line_number,
                snippet=row.strip()[:240], target=target, names=names,
                callee=primary or None, stage=stage if stage != "sink" else None,
                confidence=0.72,
            ))
        for callee in calls:
            call_stage = _classify_stage(callee, row)
            if _is_network_write(callee, function_id, unit, parsed.functions):
                call_stage = "sink"
            parsed.events.append(Event(
                kind="sink" if call_stage == "sink" else "call",
                path=unit.path, function_id=function_id, line=line_number,
                snippet=row.strip()[:240], target=target, names=names,
                callee=callee,
                arguments=_generic_call_arguments(expression, callee) or (names,),
                stage=call_stage,
                confidence=0.68,
            ))
        if re.search(r"\breturn\b", row, re.IGNORECASE):
            parsed.events.append(Event(
                kind="return", path=unit.path, function_id=function_id,
                line=line_number, snippet=row.strip()[:240], names=names,
                confidence=0.68,
            ))


def _resolve_imports(parsed: ParsedProject) -> list[dict[str, object]]:
    paths = [unit.path for unit in parsed.units]
    output: list[dict[str, object]] = []
    for binding in parsed.imports:
        target = _resolve_module_path(
            binding.path, binding.module, paths, level=binding.level,
        )
        if target and target != binding.path:
            output.append({
                "source": binding.path,
                "target": target,
                "type": binding.relation,
                "module": binding.module,
                "symbol": binding.symbol,
                "alias": binding.alias,
                "confidence": 0.98 if binding.path.endswith(".py") else 0.82,
            })
    return _dedupe_dicts(output, ("source", "target", "type", "symbol", "alias"))


def _resolve_calls(
    parsed: ParsedProject,
    relationships: list[dict[str, object]],
) -> list[dict[str, object]]:
    functions_by_path_name: dict[tuple[str, str], list[FunctionInfo]] = defaultdict(list)
    functions_by_name: dict[str, list[FunctionInfo]] = defaultdict(list)
    for function in parsed.functions.values():
        functions_by_path_name[(function.path, function.name)].append(function)
        functions_by_name[function.name].append(function)
    bindings: dict[tuple[str, str], tuple[str, str | None]] = {}
    for relationship in relationships:
        source = str(relationship["source"])
        alias = str(relationship.get("alias") or "")
        if alias:
            bindings[(source, alias)] = (
                str(relationship["target"]),
                str(relationship.get("symbol")) if relationship.get("symbol") else None,
            )
    related = defaultdict(set)
    for relationship in relationships:
        related[str(relationship["source"])].add(str(relationship["target"]))
    edges = []
    for event in parsed.events:
        if event.kind not in {"call", "sink"} or not event.callee:
            continue
        callee = event.callee.replace("::", ".").replace(":", ".")
        parts = [part for part in callee.split(".") if part]
        candidates: list[FunctionInfo] = []
        resolution = ""
        if len(parts) >= 2 and (event.path, parts[0]) in bindings:
            target_path, imported_symbol = bindings[(event.path, parts[0])]
            symbol = parts[-1] if imported_symbol is None else imported_symbol
            candidates = functions_by_path_name.get((target_path, symbol), [])
            resolution = "module_alias"
        elif parts and (event.path, parts[-1]) in bindings:
            target_path, imported_symbol = bindings[(event.path, parts[-1])]
            symbol = imported_symbol or parts[-1]
            candidates = functions_by_path_name.get((target_path, symbol), [])
            resolution = "imported_symbol"
        elif parts:
            local = functions_by_path_name.get((event.path, parts[-1]), [])
            if local:
                candidates = local
                resolution = "same_file_symbol"
            else:
                candidates = [
                    item for item in functions_by_name.get(parts[-1], [])
                    if item.path in related[event.path]
                ]
                if len(candidates) == 1:
                    resolution = "unique_related_symbol"
                elif not candidates:
                    project_candidates = functions_by_name.get(parts[-1], [])
                    if len({item.path for item in project_candidates}) == 1:
                        candidates = project_candidates
                        resolution = "unique_project_file_symbol"
        if len(candidates) > 1 and event.arguments:
            same_arity = [
                candidate for candidate in candidates
                if len(candidate.parameters) == len(event.arguments)
            ]
            if same_arity:
                candidates = same_arity
                resolution += "_arity"
        if len(candidates) > 1 and len({item.path for item in candidates}) == 1:
            # Overloaded methods in the same component share the same component-level
            # attribution.  Pick deterministically; dataflow remains conservative.
            candidates = sorted(candidates, key=lambda item: (item.line, item.identifier))[:1]
            resolution += "_overload"
        if len(candidates) != 1:
            continue
        target = candidates[0]
        event.resolved_target = target.identifier
        event.resolution = resolution
        event.confidence = min(event.confidence or 1.0, 0.98 if "symbol" in resolution or "alias" in resolution else 0.78)
        edges.append({
            "source": event.function_id,
            "target": target.identifier,
            "type": "resolved_call",
            "caller_file": event.path,
            "caller_function": parsed.functions[event.function_id].qualified_name,
            "callee_file": target.path,
            "callee_function": target.qualified_name,
            "line": event.line,
            "snippet": event.snippet,
            "assigned_variable": event.target,
            "argument_variables": [list(values) for values in event.arguments],
            "resolution": resolution,
            "confidence": round(event.confidence, 3),
        })
    return _dedupe_dicts(edges, ("source", "target", "line"))


def _trace_taint(parsed: ParsedProject) -> list[dict[str, object]]:
    events_by_function: dict[str, list[Event]] = defaultdict(list)
    for event in parsed.events:
        events_by_function[event.function_id].append(event)
    for events in events_by_function.values():
        events.sort(key=lambda item: (item.line, _event_order(item.kind)))

    parameter_traces: dict[tuple[str, str], list[dict[str, object]]] = {}
    return_traces: dict[str, list[dict[str, object]]] = {}
    field_traces: dict[tuple[str, str], list[dict[str, object]]] = {}
    found: list[list[dict[str, object]]] = []
    for _round in range(MAX_TAINT_ROUNDS):
        changed = False
        for function_id, events in events_by_function.items():
            function = parsed.functions.get(function_id)
            if function is None:
                continue
            local = {
                parameter: list(parameter_traces[(function_id, parameter)])
                for parameter in function.parameters
                if (function_id, parameter) in parameter_traces
            }
            local.update({
                field: list(trace)
                for (path, field), trace in field_traces.items()
                if path == function.path
            })
            for event in events:
                if event.kind == "source" and event.target:
                    trace = [_trace_step(event, "source", event.target)]
                    if _set_trace(local, event.target, trace):
                        changed = True
                    continue
                if event.kind in {"assign", "transform"} and event.target:
                    parent = _best_named_trace(local, event.names)
                    if parent:
                        trace = list(parent)
                        if event.kind == "transform":
                            trace.append(_trace_step(event, "transform", event.target))
                        if _set_trace(local, event.target, trace):
                            changed = True
                        field = _field_name(event.target)
                        if field and _set_trace(field_traces, (event.path, field), trace):
                            changed = True
                if event.kind in {"call", "sink"}:
                    if event.stage == "transform" and function.language != "python":
                        parent = _best_named_trace(local, event.names)
                        variable = _first_tainted_name(local, event.names)
                        if parent and variable:
                            trace = list(parent) + [
                                _trace_step(event, "transform", variable)
                            ]
                            if _set_trace(local, variable, trace):
                                changed = True
                    if _is_collection_mutation(event.callee):
                        parent = _best_named_trace(local, event.names)
                        receiver = _call_receiver(event.callee)
                        if parent and receiver:
                            trace = list(parent) + [
                                _trace_step(
                                    event,
                                    "transform",
                                    receiver,
                                    detail="tainted value aggregated into a collection",
                                )
                            ]
                            if _set_trace(local, receiver, trace):
                                changed = True
                    if event.resolved_target:
                        target_function = parsed.functions.get(event.resolved_target)
                        if target_function:
                            for index, names in enumerate(event.arguments):
                                parent = _best_named_trace(local, names)
                                if parent and index < len(target_function.parameters):
                                    parameter = target_function.parameters[index]
                                    trace = list(parent) + [
                                        _trace_step(
                                            event, "transfer", parameter,
                                            detail=(
                                                f"{function.qualified_name} -> "
                                                f"{target_function.qualified_name}"
                                            ),
                                        )
                                    ]
                                    if _set_trace(
                                        parameter_traces,
                                        (target_function.identifier, parameter),
                                        trace,
                                    ):
                                        changed = True
                            if event.target and target_function.identifier in return_traces:
                                trace = list(return_traces[target_function.identifier]) + [
                                    _trace_step(
                                        event, "transfer", event.target,
                                        detail=(
                                            f"{target_function.qualified_name} -> "
                                            f"{function.qualified_name}"
                                        ),
                                    )
                                ]
                                if _set_trace(local, event.target, trace):
                                    changed = True
                    if event.kind == "sink":
                        parent = _best_named_trace(local, event.names)
                        if parent:
                            found.append(list(parent) + [
                                _trace_step(event, "sink", _first_name(event.names))
                            ])
                if event.kind == "return":
                    parent = _best_named_trace(local, event.names)
                    if parent and _set_trace(return_traces, function_id, list(parent)):
                        changed = True
        if not changed:
            break

    shaped = []
    seen: set[tuple[tuple[str, int, str], ...]] = set()
    for trace in found:
        compact = _compact_trace(trace)
        signature = tuple(
            (str(step["file"]), int(step["line"]), str(step["stage"]))
            for step in compact
        )
        if signature in seen:
            continue
        seen.add(signature)
        stages = {str(step["stage"]) for step in compact}
        files = list(dict.fromkeys(str(step["file"]) for step in compact))
        complete = {"source", "transform", "sink"}.issubset(stages) and len(files) >= 2
        shaped.append({
            "chain_id": "cross-file-" + hashlib.sha256(
                repr(signature).encode("utf-8")
            ).hexdigest()[:12],
            "category": "Cross-file Source Transform Exfiltration",
            "risk_type": "malicious" if complete else "context",
            "severity": 9 if complete else 5,
            "complete": complete,
            "cross_file": len(files) >= 2,
            "files": files,
            "variables": list(dict.fromkeys(
                str(step.get("variable") or "") for step in compact
                if step.get("variable")
            )),
            "trace_steps": compact,
            "source": next((step for step in compact if step["stage"] == "source"), None),
            "sink": next((step for step in reversed(compact) if step["stage"] == "sink"), None),
            "confidence": round(min(float(step.get("confidence") or 0.0) for step in compact), 3),
            "evidence": (
                "静态解析到跨文件变量传递，并形成来源、处理、外传三个阶段。"
                if complete else
                "静态解析到来源与外传关联，但处理阶段或跨文件证据不完整。"
            ),
        })
        if len(shaped) >= MAX_CHAINS:
            break
    shaped.sort(key=lambda item: (not bool(item["complete"]), -float(item["confidence"]), str(item["chain_id"])))
    return shaped


def _shape_call_graph(
    parsed: ParsedProject,
    relationships: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[str, object]:
    file_nodes = [
        {"id": f"file:{unit.path}", "type": "file", "path": unit.path, "language": unit.language}
        for unit in parsed.units
    ]
    function_nodes = [
        {
            "id": item.identifier,
            "type": "function",
            "path": item.path,
            "name": item.name,
            "qualified_name": item.qualified_name,
            "line": item.line,
            "end_line": item.end_line,
            "parameters": list(item.parameters),
            "language": item.language,
        }
        for item in parsed.functions.values()
    ]
    return {
        "node_count": len(file_nodes) + len(function_nodes),
        "edge_count": len(edges),
        "file_relationship_count": len(relationships),
        "nodes": [*file_nodes, *function_nodes],
        "edges": edges,
        "file_relationships": relationships,
        "resolved_edge_count": len(edges),
        "unresolved_call_count": sum(
            event.kind in {"call", "sink"} and bool(event.callee) and not event.resolved_target
            for event in parsed.events
        ),
    }


def _rank_components(
    chains: list[dict[str, object]],
    call_edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    scores: Counter[str] = Counter()
    reasons: dict[str, list[str]] = defaultdict(list)
    weights = {"source": 18, "transfer": 8, "transform": 32, "sink": 55}
    for chain in chains:
        multiplier = 1.0 if chain.get("complete") else 0.35
        for step in chain.get("trace_steps") or []:
            path = str(step.get("file") or "")
            stage = str(step.get("stage") or "")
            if not path:
                continue
            scores[path] += round(weights.get(stage, 2) * multiplier)
            reason = f"{stage}@{step.get('line')}"
            if reason not in reasons[path]:
                reasons[path].append(reason)
    for edge in call_edges:
        scores[str(edge["caller_file"])] += 1
        scores[str(edge["callee_file"])] += 2
    maximum = max(scores.values(), default=0)
    return [
        {
            "rank": index + 1,
            "path": path,
            "score": min(100, round(value / max(1, maximum) * 100)),
            "raw_score": value,
            "reasons": reasons[path][:12],
            "method": "cross_file_chain_stage_weighting",
        }
        for index, (path, value) in enumerate(
            sorted(scores.items(), key=lambda item: (-item[1], item[0].casefold()))
        )
    ]


def _chain_finding(chain: Mapping[str, object]) -> dict[str, object]:
    sink = chain.get("sink") if isinstance(chain.get("sink"), Mapping) else {}
    return {
        "source": "cross_file_taint",
        "rule_id": "PROJECT-CROSS-FILE-TAINT",
        "category": chain.get("category"),
        "risk_type": chain.get("risk_type"),
        "severity": chain.get("severity"),
        "line": sink.get("line"),
        "file": sink.get("file"),
        "function": sink.get("function"),
        "snippet": sink.get("snippet"),
        "description": "跨文件静态污点链：来源数据经过处理后到达外传接口。",
        "harm": "敏感或外部输入可能在跨文件传递和处理后离开受信任边界。",
        "evidence": chain.get("evidence"),
        "repair_advice": "复核数据来源、跨函数变量传递和外联目的地；对敏感数据实施最小化、脱敏、出站白名单和审计。",
        "repair_suggestions": [
            "限制敏感数据读取范围并验证调用者身份。",
            "对跨模块参数传递保留数据分类和用途信息。",
            "为外联目标设置显式白名单、最小化载荷并记录审计日志。",
        ],
        "remediation_references": [],
        "owasp_category": None,
        "api_security_category": None,
        "risk_domains": ["恶意行为", "跨文件数据流", "数据外传"],
        "cwe": None,
        "code_context": [],
        "suspicion_score": 90,
        "suspicion_basis": "解析后的跨文件来源-处理-外传静态证据链",
        "confidence": chain.get("confidence"),
        "trace_steps": chain.get("trace_steps"),
        "chain_id": chain.get("chain_id"),
        "evidence_basis": "cross_file_static_dataflow",
        "basis_text": "该证据由跨文件调用解析和变量传递分析生成；最终恶意结论仍由统一AI结果通道决定。",
    }


def _generic_definitions(unit: SourceUnit) -> list[FunctionInfo]:
    patterns = [
        re.compile(r"^\s*(?:async\s+)?(?:def|function|func|fun)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)"),
        re.compile(r"^\s*function\s+([A-Za-z_$][A-Za-z0-9_$-]*)\s*\{?"),
        re.compile(r"^\s*([A-Za-z_$][A-Za-z0-9_$-]*)\s*\(\s*\)\s*\{"),
        re.compile(r"^\s*(?:(?:public|private|protected|internal)\s+)?([A-Z][A-Za-z0-9_$]*)\s*\(([^)]*)\)\s*\{"),
        re.compile(r"^\s*(?:public|private|protected|internal|static|async|virtual|override|final|native|synchronized|\s)+\s*[A-Za-z_$][A-Za-z0-9_$<>,.?\[\]]*\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)\s*(?:throws\s+[^\{]+)?\{"),
        re.compile(r"^\s*def\s+([A-Za-z_$][A-Za-z0-9_$!?=]*)\s*(?:\(([^)]*)\)|\s+([^#]+))?"),
    ]
    hits: list[tuple[int, str, tuple[str, ...]]] = []
    for line_number, row in enumerate(unit.rows, 1):
        for pattern in patterns:
            match = pattern.search(row)
            if not match:
                continue
            raw_params = next((value for value in match.groups()[1:] if value), "")
            parameters = _generic_parameters(raw_params)
            hits.append((line_number, match.group(1), parameters))
            break
    output = []
    for index, (line, name, parameters) in enumerate(hits):
        end_line = hits[index + 1][0] - 1 if index + 1 < len(hits) else len(unit.rows)
        identifier = _function_id(unit.path, name, line)
        output.append(FunctionInfo(
            identifier, unit.path, name, name, line, end_line,
            parameters, unit.language,
        ))
    return output


def _generic_parameters(raw: str) -> tuple[str, ...]:
    parameters = []
    for segment in _split_arguments(raw):
        tokens = [
            token for token in _IDENTIFIER.findall(segment)
            if token.lower() not in {
                "const", "ref", "out", "in", "self", "this", "final",
                "public", "private", "protected", "static",
            }
        ]
        if tokens:
            parameters.append(tokens[-1])
    return tuple(parameters)


def _generic_call_arguments(expression: str, callee: str) -> tuple[tuple[str, ...], ...]:
    marker = expression.find(callee)
    if marker < 0:
        return ()
    opening = expression.find("(", marker + len(callee))
    if opening < 0:
        return ()
    depth = 0
    quote = ""
    escaped = False
    for index in range(opening, len(expression)):
        char = expression[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                raw = expression[opening + 1:index]
                return tuple(
                    tuple(sorted(set(_IDENTIFIER.findall(argument))))
                    for argument in _split_arguments(raw)
                )
    return ()


def _split_arguments(raw: str) -> list[str]:
    output = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    for index, char in enumerate(raw):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{<":
            depth += 1
        elif char in ")]}>" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            output.append(raw[start:index].strip())
            start = index + 1
    tail = raw[start:].strip()
    if tail or raw.strip():
        output.append(tail)
    return output


def _is_network_write(
    callee: str,
    function_id: str,
    unit: SourceUnit,
    functions: Mapping[str, FunctionInfo],
) -> bool:
    if callee.lower().rsplit(".", 1)[-1] not in {"write", "writebytes", "send", "sendall"}:
        return False
    function = functions.get(function_id)
    if function is None:
        return False
    context = "\n".join(unit.rows[max(0, function.line - 1):function.end_line]).lower()
    return any(token in context for token in (
        "httpurlconnection", "urlconnection", "socket", "getoutputstream",
        "networkstream", "httpclient", "webrequest",
    ))


def _generic_imports(unit: SourceUnit) -> list[tuple[str, str, str | None, str, int]]:
    patterns = [
        ("import", re.compile(r"^\s*(?:import|export).*?\bfrom\s*['\"]([^'\"]+)['\"]", re.I), None),
        ("require", re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.I), None),
        ("include", re.compile(r"^\s*#\s*include\s*\"([^\"]+)\"", re.I), None),
        ("include", re.compile(r"\b(?:include|include_once|require|require_once)\s*\(?\s*['\"]([^'\"]+)['\"]", re.I), None),
        ("source", re.compile(r"^\s*(?:source|\.)\s+['\"]?([^'\"\s;]+)", re.I), None),
        ("import", re.compile(r"^\s*Import-Module\s+['\"]?([^'\"\s;]+)", re.I), None),
        ("load", re.compile(r"<script\b[^>]*\bsrc\s*=\s*['\"]([^'\"]+)['\"]", re.I), None),
    ]
    output = []
    for line_number, row in enumerate(unit.rows, 1):
        for relation, pattern, symbol in patterns:
            for match in pattern.finditer(row):
                reference = match.group(1)
                alias = posixpath.splitext(posixpath.basename(reference))[0]
                output.append((relation, reference, symbol, alias, line_number))
    return output


def _resolve_module_path(
    source_path: str,
    module: str,
    paths: Sequence[str],
    *,
    level: int = 0,
) -> str | None:
    raw = str(module or "").strip().replace("\\", "/")
    if not raw or raw.startswith(("http://", "https://", "//")):
        return None
    source_dir = posixpath.dirname(source_path)
    if level:
        base = source_dir
        for _index in range(max(0, level - 1)):
            base = posixpath.dirname(base)
        raw = posixpath.join(base, raw.replace(".", "/"))
    candidates = [
        posixpath.join(source_dir, raw), raw,
        posixpath.join(source_dir, raw.replace(".", "/")),
        raw.replace(".", "/"),
    ]
    extensions = (
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".rb", ".sh",
        ".bash", ".ps1", ".psm1", ".html", ".htm",
    )
    expanded = []
    for candidate in candidates:
        normalized = _normalize_path(candidate)
        if not normalized:
            continue
        expanded.append(normalized)
        if not posixpath.splitext(normalized)[1]:
            expanded.extend(normalized + extension for extension in extensions)
            expanded.extend(posixpath.join(normalized, "__init__" + extension) for extension in (".py",))
            expanded.extend(posixpath.join(normalized, "index" + extension) for extension in (".js", ".ts", ".py"))
    lookup = {path.casefold(): path for path in paths}
    for candidate in dict.fromkeys(expanded):
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    suffix_matches = {
        path for candidate in expanded for path in paths
        if path.casefold().endswith("/" + candidate.casefold())
    }
    return next(iter(suffix_matches)) if len(suffix_matches) == 1 else None


def _classify_stage(callee: str, snippet: str) -> str | None:
    value = f"{callee} {snippet}".lower()
    if any(token in value for token in _SINK_TOKENS):
        if any(token in value for token in ("requests.get", "request.get", "downloadstring", "downloadfile")) and not any(
            token in value for token in ("post", "put", "upload", "send", "webhook", "data=", "json=")
        ):
            return "source"
        return "sink"
    if any(token in value for token in _TRANSFORM_TOKENS):
        return "transform"
    if any(token in value for token in _SOURCE_TOKENS):
        return "source"
    return None


def _trace_step(
    event: Event,
    stage: str,
    variable: str | None,
    *,
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "stage": stage,
        "stage_label": {
            "source": "来源",
            "transfer": "跨文件传递",
            "transform": "处理",
            "sink": "外传点",
        }.get(stage, stage),
        "file": event.path,
        "function": event.function_id.split("::", 1)[-1].rsplit(":", 1)[0],
        "line": event.line,
        "variable": variable,
        "callee": event.callee,
        "snippet": event.snippet,
        "detail": detail,
        "confidence": round(event.confidence or 0.6, 3),
    }


def _set_trace(mapping: dict, key: object, candidate: list[dict[str, object]]) -> bool:
    existing = mapping.get(key)
    if existing is None or _trace_quality(candidate) > _trace_quality(existing):
        mapping[key] = candidate[-32:]
        return True
    return False


def _trace_quality(trace: list[dict[str, object]]) -> tuple[int, int, float]:
    stages = {str(step.get("stage") or "") for step in trace}
    files = {str(step.get("file") or "") for step in trace}
    confidence = min((float(step.get("confidence") or 0.0) for step in trace), default=0.0)
    return len(stages) * 10 + len(files), min(len(trace), 32), confidence


def _best_named_trace(
    mapping: Mapping[str, list[dict[str, object]]],
    names: Sequence[str],
) -> list[dict[str, object]] | None:
    candidates = [mapping[name] for name in names if name in mapping]
    return max(candidates, key=_trace_quality) if candidates else None


def _first_tainted_name(
    mapping: Mapping[str, list[dict[str, object]]],
    names: Sequence[str],
) -> str | None:
    return next((name for name in names if name in mapping), None)


def _field_name(target: str | None) -> str | None:
    value = str(target or "")
    if value.startswith(("this.", "self.")):
        return value.split(".", 1)[1]
    return None


def _is_collection_mutation(callee: str | None) -> bool:
    return str(callee or "").lower().rsplit(".", 1)[-1] in {
        "add", "append", "extend", "insert", "put", "push", "enqueue",
    }


def _call_receiver(callee: str | None) -> str | None:
    value = str(callee or "")
    return value.rsplit(".", 1)[0] if "." in value else None


def _compact_trace(trace: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    seen = set()
    for step in trace:
        key = (step.get("stage"), step.get("file"), step.get("line"), step.get("variable"))
        if key in seen:
            continue
        seen.add(key)
        output.append(step)
    return output


def _event_order(kind: str) -> int:
    return {"source": 0, "assign": 1, "transform": 2, "call": 3, "sink": 4, "return": 5}.get(kind, 9)


def _first_name(names: Sequence[str]) -> str | None:
    return str(names[0]) if names else None


def _module_function(unit: SourceUnit) -> FunctionInfo:
    return FunctionInfo(
        identifier=_module_id(unit.path), path=unit.path, name="<module>",
        qualified_name="<module>", line=1, end_line=max(1, len(unit.rows)),
        parameters=(), language=unit.language,
    )


def _enclosing_function(
    path: str,
    line: int,
    functions: Mapping[str, FunctionInfo],
) -> FunctionInfo | None:
    candidates = [
        item for item in functions.values()
        if item.path == path and item.name != "<module>" and item.line <= line <= item.end_line
    ]
    return max(candidates, key=lambda item: item.line) if candidates else None


def _python_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        return _python_target_name(node.elts[0])
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _python_call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    return _python_expression_name(node.func)


def _python_expression_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _python_expression_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _python_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _source_segment(unit: SourceUnit, node: ast.AST) -> str:
    return ast.get_source_segment(unit.content, node) or _line(unit, int(getattr(node, "lineno", 1) or 1))


def _line(unit: SourceUnit, line: int) -> str:
    return unit.rows[line - 1].strip()[:240] if 1 <= line <= len(unit.rows) else ""


def _module_id(path: str) -> str:
    return f"function:{path}::<module>:1"


def _function_id(path: str, qualified_name: str, line: int) -> str:
    return f"function:{path}::{qualified_name}:{line}"


def _normalize_path(value: object) -> str:
    normalized = posixpath.normpath(str(value or "").replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return "" if normalized in {"", ".", ".."} else normalized.lstrip("/")


def _value(raw: object, name: str) -> object:
    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def _dedupe_dicts(
    values: Iterable[dict[str, object]],
    keys: Sequence[str],
) -> list[dict[str, object]]:
    output = []
    seen = set()
    for value in values:
        signature = tuple(str(value.get(key) or "") for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        output.append(value)
    return output
