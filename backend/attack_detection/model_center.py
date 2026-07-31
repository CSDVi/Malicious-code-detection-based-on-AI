"""Normalized, artifact-backed data for the model center."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .codet5p_registry import registry_view as codet5p_registry_view
from .task_policy import task_enabled


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_CENTER_ORDER = ("xgboost", "codet5p", "gatv2")
TASK_LABELS = {
    "malicious_intent": "恶意代码检测",
    "project_malicious_intent": "恶意代码检测",
}
LANGUAGE_LABELS = {
    "bash": "Bash / Shell",
    "batch": "Batch / CMD",
    "c": "C",
    "config": "config",
    "cpp": "C++",
    "csharp": "C#",
    "go": "Go",
    "html": "HTML / HTA",
    "java": "Java",
    "javascript": "JavaScript",
    "kotlin": "Kotlin",
    "lua": "Lua",
    "perl": "Perl",
    "php": "PHP",
    "powershell": "PowerShell",
    "python": "Python",
    "ruby": "Ruby",
    "rust": "Rust",
    "scala": "Scala",
    "sql": "SQL",
    "typescript": "TypeScript",
    "unknown": "其他文本",
}
LANGUAGE_DISPLAY_ORDER = {
    language: index
    for index, language in enumerate((
        "c", "cpp", "csharp", "go", "html", "java", "javascript",
        "typescript", "kotlin", "php", "python", "bash", "powershell",
        "batch", "ruby", "rust", "scala", "lua", "perl", "sql",
        "config", "unknown",
    ))
}
MODEL_COMPARISONS = {
    "xgboost": {
        "advantage": "语言覆盖广；CPU 推理快；适合快速模式批量初筛。",
        "limitation": "跨文件关系弱；深层语义有限；复杂混淆需深度模型复核。",
    },
    "legacy_svm": {
        "advantage": "模型体积小；CPU 推理最快；适合基线对照与轻量筛查。",
        "limitation": "只看词频特征；代码语义较弱；复杂混淆识别能力有限。",
    },
    "gatv2": {
        "advantage": "跨文件关系强；调用链建模强；适合项目级恶意行为检测。",
        "limitation": "依赖项目图质量；单文件收益有限；新增语言需独立训练验证。",
    },
    "codet5p": {
        "advantage": "代码语义理解强；复杂逻辑识别强；适合标准/深度模式复核。",
        "limitation": "CPU 推理较慢；内存占用较高；当前上线语言仍较少。",
    },
}


def model_center_view() -> dict[str, Any]:
    xgb_registry = _read_json(MODEL_DIR / "xgb_registry.json")
    gatv2 = _read_json(MODEL_DIR / "gatv2_manifest.json")
    codet5p = codet5p_registry_view()

    groups = [
        _registry_group(
            key="xgboost", name="XGBoost", registry=xgb_registry,
            registry_root=MODEL_DIR / "xgb_registry", metrics_name="xgb_metrics.json",
        ),
        _codet5p_group(codet5p),
        _manifest_group("gatv2", "GATv2", gatv2),
    ]
    groups = [group for group in groups if group["versions"]]
    performance = []
    for group in groups:
        active = next(
            (version for version in group["versions"] if version["version"] == group["active_version"]),
            group["versions"][0],
        )
        for task in active["tasks"]:
            comparison = MODEL_COMPARISONS.get(group["key"], {})
            performance.append({
                "model": group["name"],
                "version": active["version"],
                "published": active["published"],
                "advantage": comparison.get("advantage", "暂无"),
                "limitation": comparison.get("limitation", "暂无"),
                **task,
            })
    return {"performance_rows": performance, "version_groups": groups}


def single_file_supported_languages() -> list[str]:
    """Return the union of XGBoost/GATv2 languages with all five metrics.

    The single-file upload contract follows the currently active, strict
    evaluation rows rather than a hard-coded language list.  This keeps the
    detection center aligned with the model center when a routed version is
    promoted or its metric coverage changes.
    """
    center = model_center_view()
    languages: set[str] = set()
    for group in center.get("version_groups", []):
        if str(group.get("key") or "") not in {"xgboost", "gatv2"}:
            continue
        active_version = str(group.get("active_version") or "")
        active = next(
            (
                version
                for version in group.get("versions", [])
                if str(version.get("version") or "") == active_version
            ),
            None,
        )
        if not isinstance(active, dict):
            continue
        for task in active.get("tasks", []):
            if not isinstance(task, dict):
                continue
            for row in task.get("language_metrics", []):
                if not isinstance(row, dict) or row.get("full_metrics") is not True:
                    continue
                language = str(row.get("language") or "").strip().lower()
                if language:
                    languages.add(language)
    return sorted(
        languages,
        key=lambda language: (
            LANGUAGE_DISPLAY_ORDER.get(language, 999),
            LANGUAGE_LABELS.get(language, language).casefold(),
        ),
    )


def _codet5p_group(registry: dict[str, Any]) -> dict[str, Any]:
    active_routes = registry.get("active_routes") or {}
    entries_by_version = {
        str(entry.get("version") or ""): entry
        for entry in registry.get("versions", [])
        if entry.get("version")
    }
    active_versions = {
        str(version)
        for task, task_routes in active_routes.items()
        if task_enabled(task) and isinstance(task_routes, dict)
        for version in task_routes.values()
        if version
    }
    versions = []
    for entry in registry.get("versions", []):
        version = str(entry.get("version") or "")
        if not version:
            continue
        task = str(entry.get("task") or "")
        is_pretrained_base = entry.get("kind") == "pretrained_base"
        if not task_enabled(task) and not is_pretrained_base:
            continue
        metrics = entry.get("test_metrics") if isinstance(entry.get("test_metrics"), dict) else {}
        tasks = {task: metrics} if task and metrics else {}
        supported_languages = entry.get("supported_languages") or []
        metrics_by_language = (
            entry.get("test_metrics_by_language")
            if isinstance(entry.get("test_metrics_by_language"), dict)
            else {
                str(supported_languages[0]): metrics
            } if len(supported_languages) == 1 and metrics else {}
        )
        versions.append({
            "version": version,
            "created_at": str(entry.get("created_at") or ""),
            "published": version in active_versions,
            "quality_gate_passed": entry.get("quality_gate_passed"),
            "tasks": _task_rows(
                tasks,
                metrics_by_language=metrics_by_language,
                supported_languages_by_task={task: supported_languages},
            ),
            "checkpoint": str(entry.get("checkpoint") or ""),
            "kind": str(entry.get("kind") or ""),
            "trainable": entry.get("trainable") is True,
        })

    summary_tasks = []
    summary_languages = []
    for task_name, task_routes in active_routes.items():
        if not task_enabled(task_name) or not isinstance(task_routes, dict):
            continue
        route_metrics = {}
        for language, version in task_routes.items():
            normalized_language = str(language).strip().lower()
            entry = entries_by_version.get(str(version)) or {}
            metrics = entry.get("test_metrics") or {}
            by_language = entry.get("test_metrics_by_language") or {}
            if isinstance(by_language.get(normalized_language), dict):
                metrics = by_language[normalized_language]
            if normalized_language and isinstance(metrics, dict) and metrics:
                route_metrics[normalized_language] = metrics
        pooled = _pooled_language_metrics(route_metrics, list(route_metrics))
        if not pooled:
            continue
        labels = [
            LANGUAGE_LABELS.get(language, language.upper())
            for language in sorted(
                route_metrics,
                key=lambda value: LANGUAGE_DISPLAY_ORDER.get(value, 999),
            )
        ]
        summary_languages.extend(labels)
        summary_tasks.extend(_task_rows(
            {task_name: pooled},
            {
                task_name: (
                    f"已上线语言合并测试集 · {' / '.join(labels)}"
                    if labels else "已上线语言合并测试集"
                ),
            },
            route_metrics,
            {task_name: list(route_metrics)},
        ))

    route_version_parts = []
    for task_routes in active_routes.values():
        if not isinstance(task_routes, dict):
            continue
        for language, version in sorted(task_routes.items()):
            if not version:
                continue
            route_version_parts.extend((
                str(language).strip().lower(),
                str(version).rsplit("-", 1)[-1],
            ))
    summary_version = (
        "codet5p-active-" + "-".join(route_version_parts)
        if route_version_parts
        else "codet5p-active-summary"
    )
    if summary_tasks:
        active_entries = [
            entries_by_version[version]
            for version in active_versions
            if version in entries_by_version
        ]
        versions.insert(0, {
            "version": summary_version,
            "display_label": summary_version,
            "created_at": max(
                (str(entry.get("created_at") or "") for entry in active_entries),
                default="",
            ),
            "published": True,
            "quality_gate_passed": all(
                entry.get("quality_gate_passed") is True
                for entry in active_entries
            ) if active_entries else None,
            "tasks": summary_tasks,
            "checkpoint": "",
            "kind": "active_summary",
            "trainable": False,
        })
        selected_version = summary_version
    else:
        selected_version = next(
            (
                version["version"]
                for version in versions
                if version["version"] in active_versions
            ),
            versions[0]["version"] if versions else "",
        )
    return {
        "key": "codet5p",
        "name": str(registry.get("display_name") or "CodeT5+ 220M"),
        "active_version": selected_version,
        "active_version_label": selected_version,
        "active_versions": sorted(active_versions),
        "activation_supported": False,
        "versions": versions,
    }


def _registry_group(
    *, key: str, name: str, registry: dict[str, Any],
    registry_root: Path, metrics_name: str,
) -> dict[str, Any]:
    active_version = str(registry.get("active_version") or "")
    versions = []
    for entry in registry.get("versions", []):
        version = str(entry.get("version") or "")
        if "vulnerability" in version.lower():
            continue
        metrics = _read_json(registry_root / version / metrics_name)
        tasks = metrics.get("tasks") or entry.get("tasks") or {}
        task_rows = _task_rows(tasks)
        if not task_rows:
            continue
        versions.append({
            "version": version,
            "created_at": str(entry.get("created_at") or metrics.get("created_at") or ""),
            "published": version == active_version,
            "quality_gate_passed": (metrics.get("quality_gate") or {}).get("passed"),
            "tasks": task_rows,
        })
    return {
        "key": key,
        "name": name,
        "active_version": active_version,
        "activation_supported": key in {"xgboost", "legacy_svm"},
        "versions": versions,
    }


def _manifest_group(key: str, name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {
            "key": key, "name": name, "active_version": "",
            "activation_supported": False, "versions": [],
        }
    version = str(manifest.get("model_version") or "")
    ready = manifest.get("runtime_ready") is True
    if key == "gatv2":
        supported_languages = manifest.get("supported_languages") or []
        metrics_by_language = manifest.get("test_metrics_by_language") or {}
        pooled_metrics = _pooled_language_metrics(
            metrics_by_language,
            supported_languages,
            manifest.get("language_coverage") or {},
        )
        tasks = {
            "project_malicious_intent": pooled_metrics or manifest.get("test_metrics") or {},
        }
        languages = " / ".join(
            LANGUAGE_LABELS.get(str(language).lower(), str(language))
            for language in supported_languages
        )
        scopes = {
            "project_malicious_intent": (
                f"已验证语言合并测试集 · 项目级依赖图 · {languages}"
                if languages else "已验证语言合并测试集 · 项目级依赖图"
            )
        }
    else:
        tasks = manifest.get("test_metrics") or {}
        metrics_by_language = manifest.get("test_metrics_by_language") or {}
        scopes = {
            task: _clear_metric_scope(
                str((manifest.get("metric_scopes") or {}).get(task) or " / ".join(
                    (manifest.get("task_language_support") or {}).get(task, [])
                ) or "未标明")
            )
            for task in tasks
        }
    if key == "gatv2":
        supported_languages_by_task = {
            "project_malicious_intent": manifest.get("supported_languages") or [],
        }
    else:
        supported_languages_by_task = manifest.get("task_language_support") or {}
    task_rows = _task_rows(
        tasks,
        scopes,
        metrics_by_language,
        supported_languages_by_task,
    )
    if not task_rows:
        return {
            "key": key, "name": name, "active_version": "",
            "activation_supported": False, "versions": [],
        }
    return {
        "key": key,
        "name": name,
        "active_version": version,
        "activation_supported": False,
        "versions": [{
            "version": version,
            "created_at": str(manifest.get("created_at") or ""),
            "published": ready,
            "quality_gate_passed": None,
            "tasks": task_rows,
        }],
    }


def _task_rows(
    tasks: dict[str, Any],
    scopes: dict[str, str] | None = None,
    metrics_by_language: dict[str, Any] | None = None,
    supported_languages_by_task: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for task_name, metrics in tasks.items():
        if not isinstance(metrics, dict) or not task_enabled(task_name):
            continue
        has_deployment_metrics = isinstance(metrics.get("deployment"), dict)
        # Deployment metrics are the authoritative current-model values.  Keep
        # root-level fields as a backwards-compatible fallback for older
        # manifests whose deployment block does not repeat every metric.
        display_metrics = {
            **metrics,
            **(metrics["deployment"] if has_deployment_metrics else {}),
        }
        supported = (
            metrics.get("supported_languages")
            or (supported_languages_by_task or {}).get(task_name)
            or []
        )
        strict_supported = metrics.get("strict_supported_languages") or supported
        scope = (scopes or {}).get(task_name)
        if not scope:
            scope = (
                "全部语言合并测试集 · " + " / ".join(
                    LANGUAGE_LABELS.get(str(value).lower(), str(value))
                    for value in strict_supported
                )
                if has_deployment_metrics
                else "合并测试集"
            )
        if has_deployment_metrics and not (scopes or {}).get(task_name):
            scope = "已验证语言合并测试集 · " + " / ".join(
                LANGUAGE_LABELS.get(str(value).lower(), str(value))
                for value in strict_supported
            )
        route_metrics = {
            str(language): route.get("deployment")
            for language, route in (metrics.get("language_routes") or {}).items()
            if isinstance(route, dict)
            and isinstance(route.get("deployment"), dict)
        }
        language_metrics = (
            metrics.get("deployment_by_language")
            or route_metrics
            or metrics.get("by_language")
            or metrics_by_language
            or {}
        )
        rows.append({
            "task": task_name,
            "task_label": TASK_LABELS.get(task_name, task_name),
            "scope": scope.upper(),
            "accuracy": _number(display_metrics.get("accuracy")),
            "precision": _number(display_metrics.get("precision")),
            "false_positive_rate": _number(display_metrics.get("false_positive_rate")),
            "false_negative_rate": _number(display_metrics.get("false_negative_rate")),
            "f1": _number(display_metrics.get("f1")),
            "samples": _integer(display_metrics.get("samples")),
            "language_metrics": _language_metric_rows(
                language_metrics,
                task_name,
                supported,
            ),
        })
    return rows


def _language_metric_rows(
    metrics_by_language: dict[str, Any],
    task_name: str,
    supported_languages: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    normalized_metrics = {
        str(language).strip().lower(): raw_metrics
        for language, raw_metrics in metrics_by_language.items()
        if str(language).strip()
    }
    languages = {
        str(language).strip().lower()
        for language in (supported_languages or [])
        if str(language).strip()
    }
    languages.update(
        str(language).strip().lower()
        for language in normalized_metrics
        if str(language).strip()
    )
    for normalized in languages:
        raw_metrics = normalized_metrics.get(normalized) or {}
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}
        task_metrics = raw_metrics.get(task_name)
        if isinstance(task_metrics, dict):
            raw_metrics = task_metrics
        has_deployment_metrics = isinstance(raw_metrics.get("deployment"), dict)
        display_metrics = raw_metrics["deployment"] if has_deployment_metrics else raw_metrics
        accuracy = _number(display_metrics.get("accuracy"))
        precision = _number(display_metrics.get("precision"))
        false_positive_rate = _number(display_metrics.get("false_positive_rate"))
        false_negative_rate = _number(display_metrics.get("false_negative_rate"))
        f1 = _number(display_metrics.get("f1"))
        full_metrics = all(
            value is not None
            for value in (
                accuracy,
                precision,
                false_positive_rate,
                false_negative_rate,
                f1,
            )
        )
        samples = _integer(display_metrics.get("samples"))
        positive_samples = _integer(display_metrics.get("positive_samples"))
        negative_samples = _integer(display_metrics.get("negative_samples"))
        if full_metrics:
            metric_note = "独立评测"
        elif display_metrics.get("insufficient_for_full_metrics") is True:
            if positive_samples == 0:
                metric_note = "已支持检测 · 测试集缺少恶意样本"
            elif negative_samples == 0:
                metric_note = "已支持检测 · 测试集缺少正常样本"
            else:
                metric_note = "已支持检测 · 独立评测样本不足"
        else:
            metric_note = "已支持检测 · 暂无独立评测"
        rows.append({
            "language": normalized,
            "language_label": LANGUAGE_LABELS.get(normalized, normalized.upper()),
            "accuracy": accuracy,
            "precision": precision,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "f1": f1,
            "samples": samples,
            "full_metrics": full_metrics,
            "metric_note": metric_note,
        })
    return sorted(
        rows,
        key=lambda row: (
            LANGUAGE_DISPLAY_ORDER.get(row["language"], 999),
            row["language_label"].casefold(),
        ),
    )


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


def _metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    total = tn + fp + fn + tp
    positives = tp + fn
    negatives = tn + fp
    predicted_positives = tp + fp
    accuracy = (tn + tp) / total if total else None
    precision = tp / predicted_positives if predicted_positives else None
    recall = tp / positives if positives else None
    false_positive_rate = fp / negatives if negatives else None
    false_negative_rate = fn / positives if positives else None
    f1_denominator = 2 * tp + fp + fn
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "f1": (2 * tp / f1_denominator) if f1_denominator else None,
        "samples": total,
        "positive_samples": positives,
        "negative_samples": negatives,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _confusion_counts(
    metrics: dict[str, Any],
    *,
    positive_samples: int | None = None,
    negative_samples: int | None = None,
) -> tuple[int, int, int, int] | None:
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, dict):
        values = tuple(
            _integer(confusion.get(key))
            for key in ("tn", "fp", "fn", "tp")
        )
        if all(value is not None for value in values):
            return tuple(int(value) for value in values)
    if (
        isinstance(confusion, list)
        and len(confusion) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in confusion)
    ):
        values = (
            _integer(confusion[0][0]),
            _integer(confusion[0][1]),
            _integer(confusion[1][0]),
            _integer(confusion[1][1]),
        )
        if all(value is not None for value in values):
            return tuple(int(value) for value in values)

    positives = (
        positive_samples
        if positive_samples is not None
        else _integer(metrics.get("positive_samples"))
    )
    negatives = (
        negative_samples
        if negative_samples is not None
        else _integer(metrics.get("negative_samples"))
    )
    false_positive_rate = _number(metrics.get("false_positive_rate"))
    false_negative_rate = _number(metrics.get("false_negative_rate"))
    if (
        positives is None
        or negatives is None
        or false_positive_rate is None
        or false_negative_rate is None
    ):
        return None
    fp = round(false_positive_rate * negatives)
    fn = round(false_negative_rate * positives)
    return negatives - fp, fp, fn, positives - fn


def _pooled_language_metrics(
    metrics_by_language: dict[str, Any],
    supported_languages: list[str] | tuple[str, ...] | set[str],
    language_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_metrics = {
        str(language).strip().lower(): metrics
        for language, metrics in metrics_by_language.items()
        if isinstance(metrics, dict)
    }
    totals = [0, 0, 0, 0]
    normalized_languages = [
        str(language).strip().lower()
        for language in supported_languages
        if str(language).strip()
    ]
    for language in normalized_languages:
        metrics = normalized_metrics.get(language)
        if not metrics:
            return {}
        coverage = (language_coverage or {}).get(language) or {}
        test_coverage = (coverage.get("splits") or {}).get("test") or {}
        counts = _confusion_counts(
            metrics,
            positive_samples=_integer(test_coverage.get("positive")),
            negative_samples=_integer(test_coverage.get("negative")),
        )
        if counts is None:
            return {}
        totals = [total + value for total, value in zip(totals, counts)]
    if not normalized_languages:
        return {}
    pooled = _metrics_from_confusion(*totals)
    pooled["aggregation"] = "pooled_language_test_samples"
    pooled["supported_languages"] = normalized_languages
    return pooled


def _clear_metric_scope(value: str) -> str:
    prefix = "最差已验证语言："
    if value.startswith(prefix):
        return "已验证语言合并测试集：" + value[len(prefix):]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
