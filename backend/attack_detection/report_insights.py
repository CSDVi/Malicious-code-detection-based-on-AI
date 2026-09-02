"""Build truthful, presentation-ready explainability data for scan reports."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .explainability import AI_ONLY_CATEGORY


RISK_LEVEL_LABELS = {
    "critical": "严重",
    "high": "高危",
    "medium": "中危",
    "low": "低危",
    "safe": "安全",
    "unknown": "需复核",
}
RISK_LEVEL_COLORS = {
    "critical": "#d35c5c",
    "high": "#e18a45",
    "medium": "#d5ae55",
    "low": "#7f9d74",
    "safe": "#4e9a7a",
    "unknown": "#80858f",
}

REPORT_CATEGORY_COLORS = (
    "#d6a84e",
    "#4f8fc0",
    "#b86b77",
    "#6e9b83",
    "#9a79b5",
    "#c27a45",
)
MODEL_FAMILY_BY_ENGINE = {
    "codet5p": "codet5p",
    "xgboost_malicious": "xgboost",
    "gatv2": "gatv2",
}
FILE_RADAR_ENGINE_ORDER = {
    "codet5p": 0,
    "xgboost_malicious": 1,
    "gatv2": 2,
}


def build_file_report_insights(
    report: dict[str, Any] | None,
    model_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a file result into decision and attribution views."""

    report = report if isinstance(report, dict) else {}
    ledger = []
    authority = str(report.get("decision_authority") or "")
    malicious_view = report.get("malicious_intent")
    malicious_view = malicious_view if isinstance(malicious_view, dict) else {}
    engine_votes = report.get("engine_votes")
    engine_votes = engine_votes if isinstance(engine_votes, dict) else {}
    malicious_vote = engine_votes.get("malicious_model")
    malicious_vote = malicious_vote if isinstance(malicious_vote, dict) else {}
    primary_engine = str(
        malicious_view.get("engine")
        or malicious_vote.get("engine")
        or ""
    )
    visible_engines = {"codet5p", "xgboost_malicious", "gatv2"}
    for raw_engine in report.get("engines") or []:
        if not isinstance(raw_engine, dict):
            continue
        name = str(raw_engine.get("name") or "")
        if (
            name not in visible_engines
            or raw_engine.get("status") != "completed"
            or raw_engine.get("probability") is None
        ):
            continue
        probability = _number(raw_engine.get("probability"))
        threshold = _number(raw_engine.get("threshold"))
        probability_percent = _percent(probability)
        threshold_percent = _percent(threshold)
        margin = (
            round(probability_percent - threshold_percent, 1)
            if probability_percent is not None and threshold_percent is not None
            else None
        )
        ledger.append({
            "name": name,
            "decision": str(raw_engine.get("decision") or "not_applicable"),
            "probability_percent": probability_percent,
            "threshold_percent": threshold_percent,
            "probability_width": _clamp(probability_percent),
            "threshold_position": _clamp(threshold_percent),
            "margin": margin,
            "margin_direction": "above" if margin is not None and margin >= 0 else "below",
            "model_version": str(raw_engine.get("model_version") or ""),
            "authority_label": (
                "本次最终主判"
                if authority == "ai" and name == primary_engine
                else "参与AI判断"
            ),
        })
    engine_order = {"codet5p": 0, "xgboost_malicious": 1, "gatv2": 2}
    ledger.sort(key=lambda item: engine_order.get(item["name"], 99))

    evidence_groups = build_evidence_groups(
        report.get("evidence_items")
        or report.get("rule_matches")
        or []
    )
    evidence_rows = [
        {"label": group["category"], "value": group["count"]}
        for group in evidence_groups
        if int(group.get("count") or 0) > 0
    ]

    return {
        "decision_ledger": ledger,
        "evidence_chart": _categorical_donut(evidence_rows),
        "model_radar": _file_model_performance_radar(
            report,
            model_catalog,
        ),
        "evidence_groups": evidence_groups,
    }


def build_evidence_groups(
    evidence_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Group repeated locations without losing their line-level evidence."""

    groups: list[dict[str, Any]] = []
    group_indexes: dict[tuple[str, str], int] = {}
    seen_occurrences: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for raw_item in evidence_items or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        category = str(item.get("category") or "未分类风险").strip()
        cwe = str(item.get("cwe") or "").strip().upper()
        key = (category.casefold(), cwe)
        group_index = group_indexes.get(key)
        if group_index is None:
            group_index = len(groups)
            group_indexes[key] = group_index
            groups.append({
                "category": category,
                "cwe": cwe or None,
                "owasp_category": item.get("owasp_category"),
                "api_security_category": item.get("api_security_category"),
                "items": [],
                "harms": [],
                "repair_suggestions": [],
                "remediation_references": [],
                "risk_domains": [],
                "cve_examples": [],
                "_seen_harms": set(),
                "_seen_suggestions": set(),
                "_seen_reference_urls": set(),
                "_seen_domains": set(),
                "_seen_examples": set(),
                "_total_occurrences": 0,
            })
            seen_occurrences[key] = set()
        group = groups[group_index]
        group["_total_occurrences"] += 1
        # Repeated copies of the same statement add noise rather than new
        # evidence. Keep one representative occurrence per category/CWE.
        snippet = " ".join(str(item.get("snippet") or "").split()).casefold()
        rule_id = str(item.get("rule_id") or "").strip().upper()
        occurrence_key = (rule_id, snippet)
        if category == "IOC 线索":
            # URL/domain extractors can report the same network literal using
            # different IOC rule IDs. Treat it as one clue and keep the report
            # compact while retaining the total occurrence count.
            canonical_ioc = snippet.removeprefix("http://").removeprefix("https://").rstrip("/")
            occurrence_key = ("ioc", canonical_ioc)
        elif category == "Command Execution":
            occurrence_key = ("command", snippet)
        if snippet and occurrence_key in seen_occurrences[key]:
            continue
        if snippet:
            seen_occurrences[key].add(occurrence_key)
        display_limit = 6 if category == "IOC 线索" else 8 if category == "Command Execution" else None
        if display_limit is not None and len(group["items"]) >= display_limit:
            continue
        group["items"].append(item)

        harm = _clean_text(item.get("harm") or item.get("description"))
        if harm and harm not in group["_seen_harms"]:
            group["harms"].append(harm)
            group["_seen_harms"].add(harm)

        suggestions = item.get("repair_suggestions") or [item.get("repair_advice")]
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        for raw_suggestion in suggestions or []:
            suggestion = _clean_text(raw_suggestion)
            if suggestion and suggestion not in group["_seen_suggestions"]:
                group["repair_suggestions"].append(suggestion)
                group["_seen_suggestions"].add(suggestion)

        for reference in item.get("remediation_references") or []:
            if not isinstance(reference, dict):
                continue
            url = str(reference.get("url") or "").strip()
            if not url or url in group["_seen_reference_urls"]:
                continue
            group["remediation_references"].append({
                "title": str(reference.get("title") or "修复依据"),
                "url": url,
            })
            group["_seen_reference_urls"].add(url)

        for raw_domain in item.get("risk_domains") or []:
            domain = _clean_text(raw_domain)
            if domain and domain not in group["_seen_domains"]:
                group["risk_domains"].append(domain)
                group["_seen_domains"].add(domain)

        for example in item.get("cve_examples") or []:
            if not isinstance(example, dict):
                continue
            example_key = str(example.get("id") or example.get("url") or "").strip()
            if not example_key or example_key in group["_seen_examples"]:
                continue
            group["cve_examples"].append(dict(example))
            group["_seen_examples"].add(example_key)

    for group in groups:
        group["count"] = len(group["items"])
        group["total_count"] = group.pop("_total_occurrences", group["count"])
        group["hidden_count"] = max(0, group["total_count"] - group["count"])
        group["context_only"] = group.get("category") == "IOC 线索"
        for private_key in (
            "_seen_harms",
            "_seen_suggestions",
            "_seen_reference_urls",
            "_seen_domains",
            "_seen_examples",
        ):
            group.pop(private_key, None)
    return groups


def build_project_report_insights(
    report: dict[str, Any] | None,
    model_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a project result into distribution and ranking views."""

    report = report if isinstance(report, dict) else {}
    level_counts = {
        str(key): int(value or 0)
        for key, value in (report.get("level_counts") or {}).items()
    }
    risk_legend = {
        key: (RISK_LEVEL_LABELS.get(key, key), RISK_LEVEL_COLORS.get(key, "#80858f"))
        for key in level_counts
    }

    file_results = [
        item for item in report.get("file_results") or report.get("high_risk_files") or []
        if isinstance(item, dict)
    ]
    category_counts: dict[str, int] = {}
    if file_results:
        for item in file_results:
            item_categories = [
                str(category)
                for category in item.get("categories", []) or []
                if str(category).strip()
            ]
            if not item_categories and str(item.get("final_decision") or "") == "malicious":
                item_categories = [AI_ONLY_CATEGORY]
            for category in item_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        project_ai_malicious = any(
            isinstance(engine, dict)
            and engine.get("status") == "completed"
            and engine.get("decision") == "malicious"
            for engine in report.get("project_engines") or []
        )
        if project_ai_malicious:
            category_counts[AI_ONLY_CATEGORY] = category_counts.get(AI_ONLY_CATEGORY, 0) + 1
    else:
        category_counts = {
            str(key): int(value or 0)
            for key, value in (report.get("category_counts") or {}).items()
            if int(value or 0) > 0
        }
        if not category_counts and str(report.get("final_decision") or "") == "malicious":
            category_counts[AI_ONLY_CATEGORY] = 1

    categories = [
        {"label": label, "value": value}
        for label, value in category_counts.items()
        if value > 0
    ]
    categories.sort(key=lambda item: (-item["value"], item["label"]))
    categories = _bar_rows(categories[:10])

    top_files = [{
        "label": str(item.get("filename") or "未命名文件"),
        "value": _score(item.get("risk_score")),
        "risk_level": str(item.get("risk_level") or "unknown"),
    } for item in file_results]
    top_files.sort(key=lambda item: (-item["value"], item["label"]))
    top_files = _bar_rows(top_files[:10], scale_max=100)

    return {
        "risk_chart": _donut_chart(level_counts, risk_legend),
        "language_chart": _categorical_columns(
            report.get("language_counts") or {},
            limit=7,
            other_label="其他语言",
        ),
        "category_rows": categories,
        "top_file_rows": top_files,
        "model_radar": _model_performance_radar(
            report,
            model_catalog,
            project=True,
        ),
        "relationship_graph": _project_relationship_layout(
            report.get("project_relationship_graph"),
        ),
        "graph_model_status": _project_graph_model_status(report),
        "cross_file_summary": _project_cross_file_summary(report),
    }


def _project_cross_file_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    analysis = report.get("project_cross_file_analysis")
    if not isinstance(analysis, dict):
        return None
    call_graph = analysis.get("call_graph")
    call_graph = call_graph if isinstance(call_graph, dict) else {}
    component = report.get("most_suspicious_component") or analysis.get(
        "most_suspicious_component"
    )
    component = component if isinstance(component, dict) else None
    chains = []
    for raw_chain in analysis.get("complete_chains") or []:
        if not isinstance(raw_chain, dict):
            continue
        steps = []
        for raw_step in raw_chain.get("trace_steps") or []:
            if not isinstance(raw_step, dict):
                continue
            steps.append({
                "stage": str(raw_step.get("stage") or ""),
                "stage_label": str(raw_step.get("stage_label") or raw_step.get("stage") or ""),
                "file": str(raw_step.get("file") or ""),
                "function": str(raw_step.get("function") or "<module>"),
                "line": _integer(raw_step.get("line")),
                "variable": str(raw_step.get("variable") or ""),
                "callee": str(raw_step.get("callee") or ""),
                "snippet": str(raw_step.get("snippet") or ""),
            })
        chains.append({
            "chain_id": str(raw_chain.get("chain_id") or ""),
            "confidence": _number(raw_chain.get("confidence")),
            "files": [str(value) for value in raw_chain.get("files") or []],
            "steps": steps,
        })
        if len(chains) >= 12:
            break
    resolved_call_count = _integer(call_graph.get("resolved_edge_count")) or 0
    file_relationship_count = _integer(call_graph.get("file_relationship_count")) or 0
    complete_chain_count = _integer(analysis.get("complete_chain_count")) or 0
    # The analyzer emits an empty result object for every standard/deep project.
    # Suppress the whole panel when there is no cross-file relationship to show.
    if not chains and not component and resolved_call_count == 0 and file_relationship_count == 0:
        return None
    return {
        "resolved_call_count": resolved_call_count,
        "file_relationship_count": file_relationship_count,
        "complete_chain_count": complete_chain_count,
        "most_suspicious_component": component,
        "most_suspicious_component_basis": str(
            analysis.get("most_suspicious_component_basis") or ""
        ),
        "chains": chains,
    }


def _project_graph_model_status(report: dict[str, Any]) -> dict[str, Any] | None:
    engine = next(
        (
            item for item in report.get("project_engines") or []
            if isinstance(item, dict) and item.get("name") == "gatv2"
        ),
        None,
    )
    if not isinstance(engine, dict):
        return None
    status = str(engine.get("status") or "unknown")
    metadata = engine.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    probability = _percent(_number(engine.get("probability")))
    threshold = _percent(_number(engine.get("threshold")))
    raw_reason = _clean_text(engine.get("error") or engine.get("reason"))
    if status == "completed":
        detail_parts = []
        if probability is not None:
            detail_parts.append(f"恶意概率 {probability}%")
        if threshold is not None:
            detail_parts.append(f"判定阈值 {threshold}%")
        decision_label = {
            "malicious": "恶意",
            "benign": "正常",
        }.get(str(engine.get("decision") or ""))
        if decision_label:
            detail_parts.append(f"GATv2判定 {decision_label}")
        detail = " · ".join(detail_parts) or "项目图推理已完成"
        label = "执行完成"
    elif "timed out after" in raw_reason.lower():
        match = re.search(r"timed out after\s+([0-9.]+)", raw_reason, re.IGNORECASE)
        seconds = match.group(1) if match else str(metadata.get("timeout_seconds") or "设定")
        detail = f"GATv2 推理超过 {seconds} 秒后被运行时终止；项目图已经构建，但没有产生模型概率。"
        label = "执行超时"
    elif status == "skipped":
        label = "未执行"
        if "no language with validated" in raw_reason.lower():
            detail = "项目语言不在当前 GATv2 已验证覆盖范围内。"
        elif "graph structure is insufficient" in raw_reason.lower():
            detail = "项目图节点或边不足，无法满足当前 GATv2 输入要求。"
        else:
            detail = raw_reason or "本次项目不满足 GATv2 执行条件。"
    elif status == "unavailable":
        label = "不可用"
        detail = "GATv2 运行环境或模型产物不可用。"
    else:
        label = "执行失败"
        detail = "GATv2 子进程异常退出，未产生可信概率；具体异常已保留在任务记录中。"
    return {
        "status": status,
        "label": label,
        "detail": detail,
        "model_version": str(engine.get("model_version") or ""),
        "node_count": _integer(metadata.get("node_count")),
        "edge_count": _integer(metadata.get("edge_count")),
        "duration_ms": _integer(engine.get("duration_ms")),
    }


def _project_relationship_layout(raw_graph: Any) -> dict[str, Any] | None:
    if not isinstance(raw_graph, dict):
        return None
    raw_nodes = [
        dict(item) for item in raw_graph.get("nodes") or []
        if isinstance(item, dict) and item.get("id")
    ]
    if not raw_nodes:
        return None
    by_id = {str(item["id"]): item for item in raw_nodes}
    edges = [
        dict(item) for item in raw_graph.get("edges") or []
        if isinstance(item, dict)
        and str(item.get("source") or "") in by_id
        and str(item.get("target") or "") in by_id
    ]
    indegree = Counter()
    adjacency: dict[str, list[str]] = {identifier: [] for identifier in by_id}
    connected: set[str] = set()
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        adjacency[source].append(target)
        indegree[target] += 1
        connected.update((source, target))

    levels: dict[str, int] = {}
    roots = sorted(
        (
            identifier for identifier in connected
            if indegree[identifier] == 0
        ),
        key=lambda identifier: (
            -len(adjacency[identifier]),
            str(by_id[identifier].get("path") or "").casefold(),
        ),
    )
    remaining = set(connected)
    seed_order = roots + sorted(
        connected - set(roots),
        key=lambda identifier: (
            -(len(adjacency[identifier]) + indegree[identifier]),
            str(by_id[identifier].get("path") or "").casefold(),
        ),
    )
    for seed in seed_order:
        if seed not in remaining:
            continue
        queue = [(seed, 0)]
        while queue:
            identifier, level = queue.pop(0)
            if identifier not in remaining:
                continue
            remaining.remove(identifier)
            levels[identifier] = min(4, level)
            for target in sorted(adjacency[identifier]):
                if target in remaining:
                    queue.append((target, level + 1))

    isolated = [identifier for identifier in by_id if identifier not in connected]
    for index, identifier in enumerate(sorted(isolated)):
        levels[identifier] = index % 4

    columns: dict[int, list[str]] = {}
    for identifier, level in levels.items():
        columns.setdefault(level, []).append(identifier)
    for identifiers in columns.values():
        identifiers.sort(key=lambda identifier: (
            -int(by_id[identifier].get("degree") or 0),
            -int(by_id[identifier].get("risk_score") or 0),
            str(by_id[identifier].get("path") or "").casefold(),
        ))

    width, height = 1000, 520
    level_values = sorted(columns)
    x_positions = {
        level: (
            width / 2
            if len(level_values) == 1
            else 80 + index * (width - 160) / (len(level_values) - 1)
        )
        for index, level in enumerate(level_values)
    }
    positioned = []
    positions = {}
    for level in level_values:
        identifiers = columns[level]
        for index, identifier in enumerate(identifiers):
            y = (
                height / 2
                if len(identifiers) == 1
                else 42 + index * (height - 84) / (len(identifiers) - 1)
            )
            x = x_positions[level]
            positions[identifier] = (x, y)
            item = by_id[identifier]
            positioned.append({
                **item,
                "x": round(x, 2),
                "y": round(y, 2),
                "label": _shorten(str(item.get("name") or item.get("path") or identifier), 26),
                "label_anchor": "end" if x > width - 170 else "start",
                "label_x": round(x - 12 if x > width - 170 else x + 12, 2),
            })

    relation_labels = {
        "import": "导入",
        "include": "包含",
        "require": "依赖",
        "source": "脚本引用",
        "load": "页面加载",
    }
    positioned_edges = []
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        if target_x > source_x:
            middle_x = (source_x + target_x) / 2
            path = (
                f"M {source_x + 7:.2f} {source_y:.2f} "
                f"C {middle_x:.2f} {source_y:.2f}, {middle_x:.2f} {target_y:.2f}, "
                f"{target_x - 8:.2f} {target_y:.2f}"
            )
        else:
            bend = max(28.0, abs(target_y - source_y) * 0.35)
            path = (
                f"M {source_x:.2f} {source_y - 7:.2f} "
                f"C {source_x:.2f} {source_y - bend:.2f}, {target_x:.2f} {target_y - bend:.2f}, "
                f"{target_x:.2f} {target_y - 8:.2f}"
            )
        relation = str(edge.get("relation") or "reference")
        positioned_edges.append({
            **edge,
            "path": path,
            "relation_label": relation_labels.get(relation, "本地引用"),
        })
    relations = [
        {"key": relation, "label": relation_labels.get(relation, "本地引用")}
        for relation in sorted({
            str(edge.get("relation") or "reference")
            for edge in positioned_edges
        })
    ]
    return {
        "width": width,
        "height": height,
        "node_count": int(raw_graph.get("node_count") or len(raw_nodes)),
        "edge_count": int(raw_graph.get("edge_count") or len(edges)),
        "displayed_node_count": len(positioned),
        "displayed_edge_count": len(positioned_edges),
        "truncated": bool(raw_graph.get("truncated")),
        "nodes": positioned,
        "edges": positioned_edges,
        "relations": relations,
    }


def _donut_chart(
    counts: dict[str, int],
    legend: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    items = []
    ordered_keys = (
        "critical", "high", "medium", "low", "safe", "unknown",
    )
    keys = [key for key in ordered_keys if key in legend]
    keys.extend(key for key in legend if key not in keys)
    for key in keys:
        label, color = legend[key]
        value = int(counts.get(key, 0) or 0)
        if value > 0:
            items.append({"key": key, "label": label, "color": color, "value": value})
    return _donut_from_items(items)


def _categorical_donut(
    rows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    ordered = sorted(
        (
            {"label": str(row.get("label") or "未分类风险"), "value": int(row.get("value") or 0)}
            for row in rows
            if int(row.get("value") or 0) > 0
        ),
        key=lambda item: (-item["value"], item["label"]),
    )
    if len(ordered) > limit:
        remainder = sum(item["value"] for item in ordered[limit:])
        ordered = ordered[:limit] + [{"label": "其他类别", "value": remainder}]
    items = [
        {
            "key": f"category-{index}",
            **item,
            "color": REPORT_CATEGORY_COLORS[index % len(REPORT_CATEGORY_COLORS)],
        }
        for index, item in enumerate(ordered)
    ]
    return _donut_from_items(items)


def _donut_from_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(item["value"] for item in items)
    cursor = 0.0
    segments = []
    for item in items:
        start = cursor
        cursor += item["value"] * 100 / total if total else 0
        end = cursor
        item["percent"] = round(item["value"] * 100 / total, 1) if total else 0.0
        start_angle = -90.0 + start * 3.6
        end_angle = -90.0 + end * 3.6
        mid_angle = start_angle + (end_angle - start_angle) / 2
        pop_distance = 8.0
        item["start_percent"] = round(start, 2)
        item["end_percent"] = round(end, 2)
        item["pop_x"] = round(math.cos(math.radians(mid_angle)) * pop_distance, 2)
        item["pop_y"] = round(math.sin(math.radians(mid_angle)) * pop_distance, 2)
        item["svg_path"] = _donut_segment_path(start_angle, end_angle)
        segments.append(f"{item['color']} {start:.2f}% {end:.2f}%")
    return {
        "entries": items,
        "total": total,
        "gradient": f"conic-gradient({', '.join(segments)})" if segments else "#2d3036",
    }


def _donut_segment_path(
    start_angle: float,
    end_angle: float,
    *,
    center: float = 82.0,
    outer_radius: float = 78.0,
    inner_radius: float = 48.0,
) -> str:
    sweep = max(0.0, end_angle - start_angle)
    if sweep >= 359.99:
        end_angle = start_angle + 359.99
        sweep = 359.99
    large_arc = 1 if sweep > 180 else 0

    def point(radius: float, angle: float) -> tuple[float, float]:
        radians = math.radians(angle)
        return (
            center + radius * math.cos(radians),
            center + radius * math.sin(radians),
        )

    outer_start = point(outer_radius, start_angle)
    outer_end = point(outer_radius, end_angle)
    inner_end = point(inner_radius, end_angle)
    inner_start = point(inner_radius, start_angle)
    return (
        f"M {outer_start[0]:.2f} {outer_start[1]:.2f} "
        f"A {outer_radius:.2f} {outer_radius:.2f} 0 {large_arc} 1 "
        f"{outer_end[0]:.2f} {outer_end[1]:.2f} "
        f"L {inner_end[0]:.2f} {inner_end[1]:.2f} "
        f"A {inner_radius:.2f} {inner_radius:.2f} 0 {large_arc} 0 "
        f"{inner_start[0]:.2f} {inner_start[1]:.2f} Z"
    )


def _categorical_columns(
    counts: dict[str, Any],
    *,
    limit: int,
    other_label: str,
) -> dict[str, Any]:
    rows = sorted(
        (
            {"label": str(label), "value": int(value or 0)}
            for label, value in counts.items()
            if int(value or 0) > 0
        ),
        key=lambda item: (-item["value"], item["label"]),
    )
    if len(rows) > limit:
        rows = rows[:limit] + [{
            "label": other_label,
            "value": sum(item["value"] for item in rows[limit:]),
        }]
    maximum = max((item["value"] for item in rows), default=0)
    total = sum(int(value or 0) for value in counts.values())
    for item in rows:
        item["height"] = round(item["value"] * 100 / maximum, 2) if maximum else 0.0
        item["percent"] = round(item["value"] * 100 / total, 1) if total else 0.0
    return {"entries": rows, "total": total}


def _file_model_performance_radar(
    report: dict[str, Any],
    model_catalog: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Overlay every executed file model with an exact evaluation match."""

    if not isinstance(model_catalog, dict):
        return None
    candidates = [
        item for item in report.get("engines") or []
        if isinstance(item, dict)
        and item.get("status") == "completed"
        and item.get("probability") is not None
        and item.get("name") in MODEL_FAMILY_BY_ENGINE
    ]
    candidates.sort(
        key=lambda item: FILE_RADAR_ENGINE_ORDER.get(
            str(item.get("name") or ""),
            99,
        )
    )
    groups = {
        str(group.get("key") or ""): group
        for group in model_catalog.get("version_groups") or []
        if isinstance(group, dict)
    }
    language = str(report.get("language") or "").strip().lower()
    series = []
    seen: set[tuple[str, str]] = set()
    for engine in candidates:
        engine_name = str(engine.get("name") or "")
        family = MODEL_FAMILY_BY_ENGINE.get(engine_name, "")
        version_name = str(engine.get("model_version") or "")
        series_key = (family, version_name)
        if not family or not version_name or series_key in seen:
            continue
        group = groups.get(family)
        if not isinstance(group, dict):
            continue
        version = next(
            (
                item for item in group.get("versions") or []
                if isinstance(item, dict)
                and str(item.get("version") or "") == version_name
            ),
            None,
        )
        if not isinstance(version, dict):
            continue
        task = next(
            (
                item for item in version.get("tasks") or []
                if isinstance(item, dict)
                and str(item.get("task") or "") == "malicious_intent"
            ),
            None,
        )
        if not isinstance(task, dict):
            continue
        metrics = next(
            (
                item for item in task.get("language_metrics") or []
                if isinstance(item, dict)
                and str(item.get("language") or "").strip().lower() == language
                and item.get("full_metrics") is True
            ),
            None,
        )
        if not isinstance(metrics, dict):
            continue
        values = (
            _number(metrics.get("accuracy")),
            _number(metrics.get("precision")),
            _number(metrics.get("f1")),
            _inverse_rate(metrics.get("false_negative_rate")),
            _inverse_rate(metrics.get("false_positive_rate")),
        )
        if any(value is None for value in values):
            continue
        series.append({
            "model_name": str(group.get("name") or engine_name),
            "version": version_name,
            "style_key": family,
            "values": tuple(float(value) for value in values),
        })
        seen.add(series_key)
    return _multi_radar_geometry(series) if series else None


def _multi_radar_geometry(
    raw_series: list[dict[str, Any]],
) -> dict[str, Any]:
    center_x, center_y, radius, label_radius = 180.0, 145.0, 84.0, 116.0
    metric_names = (
        ("准确率", "ACC"),
        ("精确率", "PRE"),
        ("F1 分数", "F1"),
        ("敏感度", "TPR"),
        ("特异度", "TNR"),
    )

    def point(index: int, scale: float, used_radius: float = radius) -> tuple[float, float]:
        angle = -math.pi / 2 + index * (2 * math.pi / len(metric_names))
        return (
            center_x + math.cos(angle) * used_radius * scale,
            center_y + math.sin(angle) * used_radius * scale,
        )

    def point_string(scales: tuple[float, ...] | list[float]) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}"
            for index, scale in enumerate(scales)
            for x, y in (point(index, scale),)
        )

    axes = []
    for index, (label, abbreviation) in enumerate(metric_names):
        outer_x, outer_y = point(index, 1.0)
        label_x, label_y = point(index, 1.0, label_radius)
        axes.append({
            "x": round(outer_x, 2),
            "y": round(outer_y, 2),
            "label_x": round(label_x, 2),
            "label_y": round(label_y, 2),
            "anchor": (
                "middle"
                if abs(label_x - center_x) < 12
                else "end" if label_x < center_x else "start"
            ),
            "label": label,
            "abbreviation": abbreviation,
        })
    series = []
    for item in raw_series:
        clamped = tuple(
            max(0.0, min(1.0, float(value)))
            for value in item["values"]
        )
        series.append({
            "model_name": item["model_name"],
            "version": item["version"],
            "style_key": item["style_key"],
            "points": point_string(list(clamped)),
            "dots": [
                {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "percent": round(value * 100, 2),
                }
                for index, value in enumerate(clamped)
                for x, y in (point(index, value),)
            ],
            "aria_metrics": "，".join(
                f"{label}{round(value * 100, 1)}%"
                for (label, _), value in zip(metric_names, clamped)
            ),
        })
    return {
        "center_x": center_x,
        "center_y": center_y,
        "grid_polygons": [
            {"level": level, "points": point_string([level] * len(metric_names))}
            for level in (0.2, 0.4, 0.6, 0.8, 1.0)
        ],
        "axes": axes,
        "series": series,
    }


def _model_performance_radar(
    report: dict[str, Any],
    model_catalog: dict[str, Any] | None,
    *,
    project: bool,
) -> dict[str, Any] | None:
    """Return a radar only when the executed and evaluated versions match.

    A report-time probability is deliberately not mixed with offline model
    quality metrics.  File reports additionally require a complete independent
    evaluation row for that exact language.
    """

    if not isinstance(model_catalog, dict):
        return None
    engine_key = "project_engines" if project else "engines"
    engines = [
        item for item in report.get(engine_key) or []
        if isinstance(item, dict)
        and item.get("status") == "completed"
        and item.get("probability") is not None
    ]
    if project:
        candidates = [item for item in engines if item.get("name") == "gatv2"]
        task_name = "project_malicious_intent"
    else:
        malicious_view = report.get("malicious_intent")
        malicious_view = malicious_view if isinstance(malicious_view, dict) else {}
        engine_votes = report.get("engine_votes")
        engine_votes = engine_votes if isinstance(engine_votes, dict) else {}
        malicious_vote = engine_votes.get("malicious_model")
        malicious_vote = malicious_vote if isinstance(malicious_vote, dict) else {}
        primary_name = str(
            malicious_view.get("engine")
            or malicious_vote.get("engine")
            or ""
        )
        order = {primary_name: -1, "codet5p": 0, "xgboost_malicious": 1}
        candidates = [item for item in engines if item.get("name") in MODEL_FAMILY_BY_ENGINE]
        candidates.sort(key=lambda item: order.get(str(item.get("name") or ""), 99))
        task_name = "malicious_intent"

    groups = {
        str(group.get("key") or ""): group
        for group in model_catalog.get("version_groups") or []
        if isinstance(group, dict)
    }
    for engine in candidates:
        engine_name = str(engine.get("name") or "")
        group = groups.get(MODEL_FAMILY_BY_ENGINE.get(engine_name, ""))
        if not isinstance(group, dict):
            continue
        version_name = str(engine.get("model_version") or "")
        version = next(
            (
                item for item in group.get("versions") or []
                if isinstance(item, dict)
                and str(item.get("version") or "") == version_name
            ),
            None,
        )
        if not isinstance(version, dict):
            continue
        task = next(
            (
                item for item in version.get("tasks") or []
                if isinstance(item, dict)
                and str(item.get("task") or "") == task_name
            ),
            None,
        )
        if not isinstance(task, dict):
            continue

        metrics = task
        scope = str(task.get("scope") or "独立评测")
        if not project:
            language = str(report.get("language") or "").strip().lower()
            metrics = next(
                (
                    item for item in task.get("language_metrics") or []
                    if isinstance(item, dict)
                    and str(item.get("language") or "").strip().lower() == language
                    and item.get("full_metrics") is True
                ),
                None,
            )
            if not isinstance(metrics, dict):
                continue
            scope = f"{metrics.get('language_label') or language.upper()} · 独立评测"

        values = (
            _number(metrics.get("accuracy")),
            _number(metrics.get("precision")),
            _number(metrics.get("f1")),
            _inverse_rate(metrics.get("false_negative_rate")),
            _inverse_rate(metrics.get("false_positive_rate")),
        )
        if any(value is None for value in values):
            continue
        return _radar_geometry(
            model_name=str(group.get("name") or engine_name),
            version=version_name,
            scope=scope,
            samples=_integer(metrics.get("samples")),
            values=tuple(float(value) for value in values),
        )
    return None


def _radar_geometry(
    *,
    model_name: str,
    version: str,
    scope: str,
    samples: int | None,
    values: tuple[float, float, float, float, float],
) -> dict[str, Any]:
    center_x, center_y, radius, label_radius = 180.0, 145.0, 84.0, 116.0
    metric_names = (
        ("准确率", "ACC"),
        ("精确率", "PRE"),
        ("F1 分数", "F1"),
        ("敏感度", "TPR"),
        ("特异度", "TNR"),
    )

    def point(index: int, scale: float, used_radius: float = radius) -> tuple[float, float]:
        angle = -math.pi / 2 + index * (2 * math.pi / len(metric_names))
        return (
            center_x + math.cos(angle) * used_radius * scale,
            center_y + math.sin(angle) * used_radius * scale,
        )

    def point_string(scales: tuple[float, ...] | list[float]) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}"
            for index, scale in enumerate(scales)
            for x, y in (point(index, scale),)
        )

    clamped = tuple(max(0.0, min(1.0, value)) for value in values)
    axes = []
    for index, ((label, abbreviation), value) in enumerate(zip(metric_names, clamped)):
        outer_x, outer_y = point(index, 1.0)
        label_x, label_y = point(index, 1.0, label_radius)
        axes.append({
            "x": round(outer_x, 2),
            "y": round(outer_y, 2),
            "label_x": round(label_x, 2),
            "label_y": round(label_y, 2),
            "anchor": (
                "middle"
                if abs(label_x - center_x) < 12
                else "end" if label_x < center_x else "start"
            ),
            "label": label,
            "abbreviation": abbreviation,
            "percent": round(value * 100, 1),
        })
    return {
        "model_name": model_name,
        "version": version,
        "version_short": _shorten(version, 44),
        "scope": scope,
        "samples": samples,
        "center_x": center_x,
        "center_y": center_y,
        "grid_polygons": [
            {"level": level, "points": point_string([level] * len(metric_names))}
            for level in (0.2, 0.4, 0.6, 0.8, 1.0)
        ],
        "axes": axes,
        "points": point_string(list(clamped)),
    }


def _bar_rows(rows: list[dict[str, Any]], scale_max: float | None = None) -> list[dict[str, Any]]:
    maximum = scale_max if scale_max is not None else max((float(row["value"]) for row in rows), default=0.0)
    for row in rows:
        row["width"] = round(float(row["value"]) * 100 / maximum, 1) if maximum else 0.0
    return rows


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _inverse_rate(value: Any) -> float | None:
    number = _number(value)
    return 1.0 - number if number is not None else None


def _percent(value: float | None) -> float | None:
    return round(value * 100, 1) if value is not None else None


def _score(value: Any) -> int:
    number = _number(value)
    return int(round(number or 0.0))


def _clamp(value: Any) -> float:
    number = _number(value)
    return round(max(0.0, min(100.0, number or 0.0)), 2)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _shorten(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    head = max(12, maximum - 14)
    return f"{value[:head]}…{value[-11:]}"
