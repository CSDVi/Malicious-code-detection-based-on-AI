"""Detection center, history, reporting, and model-management routes."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import date
from functools import partial
from pathlib import Path

from flask import Blueprint, Response, abort, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from attack_detection.database import (
    cancel_persisted_scan_job,
    get_scan_job,
    get_record,
    get_statistics,
    list_scan_jobs,
    list_record_summaries,
    list_records,
    list_training_jobs,
    save_detection,
    save_project_results,
    save_scan_job,
)
from attack_detection.cancellation import raise_if_cancelled
from attack_detection.jobs import scan_jobs
from attack_detection.fusion import risk_level as score_risk_level
from attack_detection.explainability import (
    build_ai_explainability,
    merge_model_line_attributions,
    order_evidence_items,
)
from attack_detection.ml import classifier
from attack_detection.languages import (
    BINARY_EXTENSIONS,
    EXTENSION_LANGUAGE,
    display_language,
    is_generic_text_path,
    language_from_path,
)
from attack_detection.model_center import (
    LANGUAGE_DISPLAY_ORDER,
    LANGUAGE_LABELS,
    MODEL_CENTER_ORDER,
    model_center_view,
    single_file_supported_languages,
)
from attack_detection.model_immunity import run_training_poisoning_gate
from attack_detection.model_registry import activate_version, runtime_status
from attack_detection.project_scanner import ArchiveTooLargeError, scan_zip_project, stage_project_archive
from attack_detection.report import render_record_markdown
from attack_detection.remediation import remediation_for_finding
from attack_detection.report_insights import (
    RISK_LEVEL_LABELS,
    build_file_report_insights,
    build_project_report_insights,
)
from attack_detection.risk_taxonomy import taxonomy_for_category
from attack_detection.scanner import is_allowed_file, scan_file
from attack_detection.task_policy import is_active_finding
from attack_detection.training_jobs import training_jobs
from attack_detection.trainer import train_model
from attack_detection.training.deep_web_trainer import train_codet5p, train_gatv2
from attack_detection.training.xgb_trainer import train_xgboost
from attack_detection.xgb_registry import activate_version as activate_xgb_version

attack_bp = Blueprint("attack", __name__, url_prefix="/attack")
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODE_LABELS = {"auto": "自动", "quick": "快速", "standard": "标准", "deep": "深度"}
MAX_CODE_UPLOAD_SIZE = 20 * 1024 * 1024
TRAINING_UPLOAD_ROOT = BACKEND_DIR / "data" / "training_uploads"
MAX_TRAINING_DATASET_SIZE = 512 * 1024 * 1024
TRAINING_DATASET_SUFFIXES = {".jsonl", ".csv"}
TRAINING_REQUIRED_FIELDS = {
    "code", "label", "split", "language", "review_status", "label_confidence",
}
GAT_GRAPH_REQUIRED_FIELDS = {"nodes", "edges", "label", "split"}
TRAINING_MODEL_FAMILIES = {
    "codet5p": {"label": "CodeT5+ 220M", "trainer": None, "reload": None},
    "xgboost": {"label": "XGBoost", "trainer": None, "reload": None},
    "legacy_svm": {"label": "TF-IDF / SVM", "trainer": train_model, "reload": classifier.reload},
    "gatv2": {"label": "GATv2", "trainer": train_gatv2, "reload": None},
}


def _train_xgboost(dataset_path: str | Path, progress_callback=None) -> dict[str, object]:
    if progress_callback:
        progress_callback(0.08, "正在准备 XGBoost 训练数据")
    metrics = train_xgboost(dataset_path)
    if progress_callback:
        progress_callback(0.96, "正在登记 XGBoost 模型版本")
    return metrics


TRAINING_MODEL_FAMILIES["xgboost"]["trainer"] = _train_xgboost


def _single_file_upload_contract() -> dict[str, object]:
    """Build the single-file UI and validation contract from gated coverage."""
    languages = single_file_supported_languages()
    # The routed ``config`` model is trained/evaluated as YAML in this
    # project.  Keep other generic config syntaxes out of the single-file
    # contract so the displayed language and accepted suffixes agree.
    source_extensions = {
        extension
        for extension, language in EXTENSION_LANGUAGE.items()
        if language in languages
        and (language != "config" or extension in {".yml", ".yaml"})
    }
    source_extensions.add(".txt")
    extensions = source_extensions | set(BINARY_EXTENSIONS)
    labels = [
        LANGUAGE_LABELS.get(language, language.upper())
        for language in languages
    ]
    return {
        "languages": languages,
        "labels": labels,
        "extensions": extensions,
        # An accept allowlist prevents browsers from selecting extensionless
        # files. Backend validation remains authoritative for every upload.
        "accept": "",
    }


def _validate_training_dataset(path: Path, model_family: str = "legacy_svm") -> None:
    """Perform a bounded schema check before a dataset enters the training queue."""
    first_record: dict[str, object] | None = None
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            fields = set(csv.DictReader(stream).fieldnames or [])
    else:
        fields: set[str] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_number} 行不是有效 JSON。") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL 第 {line_number} 行必须是一个对象。")
                first_record = record
                fields = set(record)
                break
    if not fields:
        raise ValueError("训练集为空，或没有可读取的字段。")
    if model_family == "gatv2" and path.suffix.lower() == ".jsonl" and GAT_GRAPH_REQUIRED_FIELDS.issubset(fields):
        nodes = first_record.get("nodes") if isinstance(first_record, dict) else None
        has_file_language = isinstance(nodes, list) and any(
            isinstance(node, dict)
            and node.get("type") == "file"
            and str(node.get("language") or "").strip()
            for node in nodes
        )
        if not has_file_language:
            raise ValueError("GATv2 项目图必须至少包含一个带 language 字段的文件节点。")
        return
    missing = sorted(TRAINING_REQUIRED_FIELDS - fields)
    if missing:
        suffix = "；GATv2 也可以使用包含 nodes、edges、label、split 的图 JSONL" if model_family == "gatv2" else ""
        raise ValueError(f"训练集缺少必要字段：{', '.join(missing)}{suffix}。")


def _stage_training_dataset(upload, model_family: str = "legacy_svm") -> tuple[Path, str]:
    """Stream a user-selected dataset into an isolated project directory."""
    original_name = Path(str(upload.filename or "").replace("\\", "/")).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in TRAINING_DATASET_SUFFIXES:
        raise ValueError("训练集仅支持 .jsonl 或 .csv 文件。")
    safe_name = secure_filename(original_name) or f"training-dataset{suffix}"
    target_dir = TRAINING_UPLOAD_ROOT / uuid.uuid4().hex
    target_dir.mkdir(parents=True, exist_ok=False)
    target_path = target_dir / safe_name
    total = 0
    try:
        with target_path.open("wb") as output:
            while True:
                block = upload.stream.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_TRAINING_DATASET_SIZE:
                    raise ValueError("训练集不能超过 512 MB。")
                output.write(block)
        if total == 0:
            raise ValueError("训练集文件为空。")
        _validate_training_dataset(target_path, model_family)
        return target_path, original_name
    except Exception:
        target_path.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass
        raise


def _training_model_options(center: dict[str, object]) -> list[dict[str, str]]:
    """Expose versions from the three model families visible in the product UI."""
    groups = {str(group["key"]): group for group in center.get("version_groups", [])}
    options = []
    for family_key in MODEL_CENTER_ORDER:
        family = TRAINING_MODEL_FAMILIES[family_key]
        group = groups.get(family_key)
        if not group:
            continue
        active_version = str(group.get("active_version") or "")
        versions = sorted(
            group.get("versions", []),
            key=lambda item: str(item.get("version") or "") == active_version,
            reverse=True,
        )
        for version_data in versions:
            version = str(version_data.get("version") or "")
            if not version:
                continue
            if not version_data.get("tasks") and not (
                family_key == "codet5p" and version_data.get("kind") == "pretrained_base"
            ):
                continue
            if family_key == "codet5p" and version_data.get("trainable") is not True:
                continue
            state = "当前版本" if version == active_version else "历史版本"
            if family_key == "codet5p" and version_data.get("kind") == "pretrained_base":
                state = "可训练基础版本"
            options.append({
                "key": f"{family_key}|{version}",
                "family": family_key,
                "family_label": str(family["label"]),
                "version": version,
                "label": f"{family['label']} {state}：{version}",
            })
    return options


def _visible_runtime_models(models: list[dict[str, object]]) -> list[dict[str, object]]:
    """Hide report-only control/fallback models and preserve quick-to-deep order."""
    by_engine = {
        str(model.get("engine") or ""): model
        for model in models
    }
    return [
        by_engine[engine]
        for engine in MODEL_CENTER_ORDER
        if engine in by_engine
    ]


def _auxiliary_analysis_view(result: dict[str, object] | None) -> dict[str, list[dict[str, str]]]:
    """Separate completed auxiliary analyzers from unavailable/inapplicable ones."""

    if not isinstance(result, dict):
        return {"executed": [], "inactive": []}
    language = str(result.get("language") or "unknown").lower()
    engines = {
        str(engine.get("name") or ""): engine
        for engine in (result.get("engines") or [])
        if isinstance(engine, dict) and engine.get("name")
    }
    executed: list[dict[str, str]] = []
    inactive: list[dict[str, str]] = []
    static = engines.get("static_evidence")
    pe_static = engines.get("pe_static")

    if static and static.get("status") == "completed":
        metadata = static.get("metadata") if isinstance(static.get("metadata"), dict) else {}
        findings = static.get("findings") if isinstance(static.get("findings"), list) else []
        chain_count = sum(
            isinstance(item, dict) and item.get("source") == "behavior_chain"
            for item in findings
        )
        executed.extend([
            {
                "name": "字符串与 IOC",
                "detail": f"本地只读分析，发现 {int(metadata.get('ioc_count') or 0)} 条 IOC 线索",
            },
            {
                "name": "静态去混淆",
                "detail": f"已检查，静态解码 {int(metadata.get('decoded_count') or 0)} 个字面量",
            },
            {
                "name": "行为链",
                "detail": f"已检查，发现 {chain_count} 条组合行为链",
            },
        ])
    elif language == "binary" and pe_static and pe_static.get("status") == "completed":
        metadata = pe_static.get("metadata") if isinstance(pe_static.get("metadata"), dict) else {}
        executed.append({
            "name": "字符串与 IOC",
            "detail": f"从二进制中提取，发现 {int(metadata.get('ioc_count') or 0)} 条 IOC 线索",
        })
        inactive.extend([
            {"name": "静态去混淆", "detail": "当前二进制文件不适用"},
            {"name": "行为链", "detail": "当前二进制文件不适用"},
        ])
    else:
        reason = _auxiliary_inactive_reason(static, "本次记录未执行本地源码分析")
        inactive.extend([
            {"name": "字符串与 IOC", "detail": reason},
            {"name": "静态去混淆", "detail": reason},
            {"name": "行为链", "detail": reason},
        ])

    if pe_static and pe_static.get("status") == "completed":
        metadata = pe_static.get("metadata") if isinstance(pe_static.get("metadata"), dict) else {}
        executed.append({
            "name": "PE/DLL 只读解析",
            "detail": f"已读取文件结构，共 {int(metadata.get('section_count') or 0)} 个节区",
        })
    elif language != "binary":
        inactive.append({"name": "PE/DLL 只读解析", "detail": "当前不是 EXE/DLL/SYS/OCX 文件"})
    else:
        inactive.append({
            "name": "PE/DLL 只读解析",
            "detail": _auxiliary_inactive_reason(pe_static, "本次没有完成二进制解析"),
        })

    reputation = engines.get("hash_reputation")
    if reputation and reputation.get("status") == "completed":
        metadata = reputation.get("metadata") if isinstance(reputation.get("metadata"), dict) else {}
        executed.append({
            "name": "SHA256 外部信誉",
            "detail": (
                f"{metadata.get('provider') or '外部服务'}已返回："
                f"恶意 {int(metadata.get('malicious') or 0)}，"
                f"可疑 {int(metadata.get('suspicious') or 0)}；"
                "仅作外部复核线索，不参与AI结论或风险分"
            ),
        })
    elif _external_reputation_was_configured(reputation):
        inactive.append({
            "name": "SHA256 外部信誉",
            "detail": _external_reputation_reason(reputation),
        })

    sandbox = engines.get("isolated_sandbox")
    if sandbox and sandbox.get("status") == "completed":
        metadata = sandbox.get("metadata") if isinstance(sandbox.get("metadata"), dict) else {}
        state = {
            "submitted": "已提交",
            "queued": "排队中",
            "running": "运行中",
            "completed": "已完成",
            "finished": "已完成",
            "failed": "失败",
            "error": "失败",
            "terminated": "已终止",
        }.get(str(metadata.get("status") or "").lower(), "已接收")
        executed.append({
            "name": "隔离动态沙箱",
            "detail": f"外部隔离服务已接收，任务状态：{state}",
        })
    else:
        inactive.append({
            "name": "隔离动态沙箱",
            "detail": _sandbox_inactive_reason(sandbox),
        })
    return {"executed": executed, "inactive": inactive}


def _auxiliary_inactive_reason(engine: dict[str, object] | None, default: str) -> str:
    if not engine:
        return default
    status = str(engine.get("status") or "")
    if status == "failed":
        return "执行失败"
    if status == "skipped":
        return "本次已跳过"
    if status == "unavailable":
        return "当前不可用"
    return default


def _external_reputation_was_configured(
    engine: dict[str, object] | None,
) -> bool:
    if not engine:
        return False
    metadata = engine.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(engine.get("reason") or engine.get("error") or "").lower()
    return bool(metadata.get("provider")) or engine.get("status") == "failed" or any(
        token in reason
        for token in (
            "hash not found",
            "unsupported reputation provider",
            "api key is not configured",
            "api_key is not configured",
        )
    )


def _external_reputation_reason(engine: dict[str, object] | None) -> str:
    if not engine:
        return "未配置/未查询"
    reason = str(engine.get("reason") or engine.get("error") or "").lower()
    if "hash not found" in reason:
        return "外部服务未查询到该哈希记录"
    if engine.get("status") == "failed":
        return "外部信誉查询失败"
    if "unsupported reputation provider" in reason:
        return "配置的外部信誉服务不受支持"
    return "未配置/未查询"


def _sandbox_inactive_reason(engine: dict[str, object] | None) -> str:
    if not engine:
        return "未配置/未提交"
    reason = str(engine.get("reason") or engine.get("error") or "").lower()
    if "exceeds sandbox limit" in reason:
        return "文件超过沙箱提交大小限制"
    if "requires xiezhi_sandbox_auto_scan=1" in reason:
        return "已配置服务，但未开启自动提交"
    if engine.get("status") == "failed":
        return "沙箱提交或结果查询失败"
    return "未配置/未提交"


@attack_bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None
    record_id = None
    job = None
    upload_contract = _single_file_upload_contract()
    if request.method == "POST":
        mode = _selected_mode()
        upload = request.files.get("code_file")
        if upload and upload.filename:
            filename = secure_filename(upload.filename)
            payload = upload.stream.read(MAX_CODE_UPLOAD_SIZE + 1)
            if len(payload) > MAX_CODE_UPLOAD_SIZE:
                flash("单个代码文件不能超过 20 MB。")
                return redirect(url_for("attack.index", tab="file"))
        else:
            flash("请选择需要检测的代码文件。")
            return redirect(url_for("attack.index", tab="file"))

        suffix = Path(filename).suffix.lower()
        if is_generic_text_path(filename) and not is_allowed_file(filename, payload):
            flash("TXT 或无后缀文件必须是可读取的文本，不能包含二进制内容。")
            return redirect(url_for("attack.index", tab="file"))
        if not is_generic_text_path(filename) and suffix not in upload_contract["extensions"]:
            flash("不支持该文件类型，请使用检测中心列出的源码、YAML 或 EXE/DLL/SYS/OCX 格式。")
            return redirect(url_for("attack.index", tab="file"))
        job = scan_jobs.submit(
            mode,
            current_user.username,
            filename,
            lambda cancel_event, progress: _scan_single_file(
                filename,
                payload,
                mode,
                cancel_event,
                progress,
            ),
            on_update=_persist_file_job,
            target_type="file",
        )
        return redirect(url_for("attack.index", job_id=job.id))
    else:
        job_id = request.args.get("job_id", "").strip()
        if job_id:
            job = scan_jobs.get(job_id)
            if job is None:
                job = get_scan_job(job_id, current_user.username)
            job_username = job.get("username") if isinstance(job, dict) else getattr(job, "username", None)
            job_target_type = job.get("target_type") if isinstance(job, dict) else getattr(job, "target_type", None)
            if not job or job_username != current_user.username or job_target_type != "file":
                abort(404)
            job_status_value = job.get("status") if isinstance(job, dict) else job.status
            if job_status_value == "completed":
                result = job.get("result") if isinstance(job, dict) else job.result
                record_id = (result or {}).get("_record_id")
        else:
            job = _latest_active_scan_job(current_user.username, "file")

    return render_template(
        "attack/index.html",
        active_tab="file",
        result=result,
        project_result=None,
        record_id=record_id,
        job=job,
        mode_availability=_mode_availability(),
        single_file_language_labels=upload_contract["labels"],
        single_file_accept=upload_contract["accept"],
        auxiliary_analysis=_auxiliary_analysis_view(result),
        file_report_insights=build_file_report_insights(
            result,
            model_center_view() if result else None,
        ),
    )


@attack_bp.route("/project", methods=["GET", "POST"])
@login_required
def project_scan():
    result = None
    job = None
    if request.method == "POST":
        upload = request.files.get("project_zip")
        if not upload or not upload.filename:
            flash("请选择需要检测的 ZIP 项目包。")
            return redirect(url_for("attack.project_scan"))
        filename = secure_filename(upload.filename)
        if not filename.lower().endswith(".zip"):
            flash("项目检测仅支持 ZIP 压缩包。")
            return redirect(url_for("attack.project_scan"))
        mode = _selected_mode()
        # Standard/deep candidate scans always generate their line-level AI
        # explanation. Quick mode still suppresses it inside project_scanner.
        line_explanations = True
        try:
            staged_path = stage_project_archive(upload.stream)
        except ArchiveTooLargeError:
            flash("ZIP 项目包不能超过 1 GB。")
            return redirect(url_for("attack.project_scan"))
        except OSError as exc:
            flash(f"ZIP 项目包暂存失败：{exc}")
            return redirect(url_for("attack.project_scan"))
        job = scan_jobs.submit(
            mode,
            current_user.username,
            filename,
            lambda cancel_event, progress, path=staged_path: _scan_staged_project(
                path,
                filename,
                mode,
                line_explanations,
                cancel_event,
                progress,
            ),
            on_update=_persist_project_job,
            target_type="project",
        )
        return redirect(url_for("attack.project_scan", job_id=job.id))
    else:
        job_id = request.args.get("job_id", "").strip()
        if job_id:
            job = scan_jobs.get(job_id)
            if job is None:
                job = get_scan_job(job_id, current_user.username)
            job_username = job.get("username") if isinstance(job, dict) else getattr(job, "username", None)
            job_target_type = job.get("target_type") if isinstance(job, dict) else getattr(job, "target_type", None)
            if not job or job_username != current_user.username or job_target_type != "project":
                abort(404)
            job_status_value = job.get("status") if isinstance(job, dict) else job.status
            if job_status_value == "completed":
                result = job.get("result") if isinstance(job, dict) else job.result
                result = _project_result_view(result)
        else:
            job = _latest_active_scan_job(current_user.username, "project")

    return render_template(
        "attack/index.html",
        active_tab="project",
        result=None,
        project_result=result,
        record_id=None,
        job=job,
        mode_availability=_mode_availability(),
        project_report_insights=build_project_report_insights(
            result,
            model_center_view() if result else None,
        ),
    )


@attack_bp.route("/project/<job_id>/files/<int:file_index>")
@login_required
def project_file_detail(job_id: str, file_index: int):
    job = scan_jobs.get(job_id)
    if job is None:
        job = get_scan_job(job_id, current_user.username)
    job_username = job.get("username") if isinstance(job, dict) else getattr(job, "username", None)
    job_status_value = job.get("status") if isinstance(job, dict) else getattr(job, "status", None)
    job_target_type = job.get("target_type") if isinstance(job, dict) else getattr(job, "target_type", None)
    if (
        not job
        or job_username != current_user.username
        or job_status_value != "completed"
        or job_target_type != "project"
    ):
        abort(404)

    project_result = _project_result_view(job.get("result") if isinstance(job, dict) else job.result)
    file_result = _project_file_detail_result(project_result, file_index)
    if file_result is None:
        abort(404)

    evidence_items = [
        dict(item)
        for item in file_result.get("evidence_items") or []
        if isinstance(item, dict)
    ]
    evidence_items.sort(
        key=_project_evidence_sort_key
    )
    return render_template(
        "attack/project_file_detail.html",
        job_id=job_id,
        project_name=str((project_result or {}).get("project_name") or "项目检测"),
        file_result=file_result,
        evidence_items=evidence_items,
        file_report_insights=build_file_report_insights(
            file_result,
            model_center_view(),
        ),
    )


def _project_file_detail_result(
    project_result: dict[str, object] | None,
    file_index: int,
) -> dict[str, object] | None:
    """Return an explainable standard/deep file result by its risk-list index."""

    if not isinstance(project_result, dict) or file_index < 0:
        return None
    files = project_result.get("high_risk_files")
    if not isinstance(files, list) or file_index >= len(files):
        return None
    file_result = files[file_index]
    if not isinstance(file_result, dict):
        return None
    if str(file_result.get("effective_mode") or "") not in {"standard", "deep"}:
        return None
    return dict(file_result)


_SKIPPED_WARNING_PREFIXES = {
    "已跳过超出大小限制的文件：": "超过单文件大小限制",
    "已跳过压缩比异常的文件：": "压缩比异常",
    "已跳过可疑路径：": "路径不安全",
    "已跳过超出项目根目录的路径：": "路径超出项目根目录",
}


def _project_result_view(project_result: dict[str, object] | None) -> dict[str, object] | None:
    """Add stable display rows, file types, and readable scan-scope notes."""

    if not isinstance(project_result, dict):
        return None
    result = dict(project_result)
    average_score = float(result.get("average_score") or 0)
    max_score = float(result.get("max_score") or 0)
    result["risk_score"] = round(max(average_score, max_score * 0.8), 1)
    high_risk_files = [
        dict(item)
        for item in result.get("high_risk_files") or []
        if isinstance(item, dict)
    ]
    skipped_files = _project_skipped_files(result)
    display_files: list[dict[str, object]] = []
    extension_counts: dict[str, int] = {}
    normalized_skipped = []

    # Surface restricted files first so their audit number is easy to find and
    # remains the same across searching, filtering, and pagination.
    for item in skipped_files:
        filename = str(item.get("filename") or "").strip()
        if not filename:
            continue
        extension = _project_file_extension(filename)
        normalized = {
            "project_serial": len(display_files) + 1,
            "filename": filename,
            "reason": str(item.get("reason") or "受扫描安全限制"),
            "file_extension": extension,
        }
        normalized_skipped.append(normalized)
        display_files.append({
            **normalized,
            "language": "unknown",
            "display_language": "未检测",
            "filter_language": "__skipped__",
            "final_decision": "unknown",
            "risk_level": "unknown",
            "risk_score": None,
            "effective_mode": "skipped",
            "detail_index": None,
            "detail_available": False,
            "scan_status": "skipped",
        })
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

    for detail_index, item in enumerate(high_risk_files):
        filename = str(item.get("filename") or "")
        extension = _project_file_extension(filename)
        raw_language = str(item.get("language") or "unknown")
        if raw_language == "unknown":
            raw_language = language_from_path(filename)
        display_lang = str(
            item.get("display_language")
            or display_language(raw_language, filename)
        )
        row = dict(item)
        row.update({
            "project_serial": len(display_files) + 1,
            "detail_index": detail_index,
            "detail_available": str(item.get("effective_mode") or "") in {"standard", "deep"},
            "file_extension": extension,
            "scan_status": "completed",
            "display_language": display_lang,
            "filter_language": display_lang,
        })
        display_files.append(row)
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

    warnings = [str(item) for item in result.get("warnings") or []]
    other_warnings = [
        warning
        for warning in warnings
        if not warning.startswith("为控制扫描时长，CodeT5+ 220M 实际复核")
        and not any(warning.startswith(prefix) for prefix in _SKIPPED_WARNING_PREFIXES)
    ]
    result["high_risk_files"] = high_risk_files
    if high_risk_files:
        language_counts: dict[str, int] = {}
        for item in high_risk_files:
            filename = str(item.get("filename") or "")
            raw_language = str(item.get("language") or "unknown")
            if raw_language == "unknown":
                raw_language = language_from_path(filename)
            concrete_language = str(
                item.get("display_language")
                or display_language(raw_language, filename)
            )
            item["display_language"] = concrete_language
            language_counts[concrete_language] = (
                language_counts.get(concrete_language, 0) + 1
            )
        result["language_counts"] = language_counts
    result["display_files"] = display_files
    result["skipped_files"] = normalized_skipped
    result["other_warnings"] = other_warnings
    result["warning_count"] = len(normalized_skipped) + len(other_warnings)
    skipped_reason_counts: dict[str, int] = {}
    for item in normalized_skipped:
        reason = str(item.get("reason") or "受扫描安全限制")
        skipped_reason_counts[reason] = skipped_reason_counts.get(reason, 0) + 1
    result["skipped_reason_counts"] = [
        {"reason": reason, "count": count}
        for reason, count in sorted(skipped_reason_counts.items())
    ]
    result["warning_breakdown"] = {
        "skipped_file_count": len(normalized_skipped),
        "other_limit_count": len(other_warnings),
    }
    result["quick_only_file_count"] = int(
        result.get("quick_only_file_count")
        or max(0, len(high_risk_files) - int(result.get("deep_scanned_file_count") or 0))
    )
    result["file_extensions"] = [
        {
            "value": extension,
            "label": "无后缀" if extension == "__none__" else extension,
            "count": count,
        }
        for extension, count in sorted(extension_counts.items(), key=lambda item: item[0])
    ]
    language_filters: dict[str, dict[str, object]] = {}
    risk_filters: dict[str, dict[str, object]] = {}
    for row in display_files:
        language_value = str(
            row.get("filter_language")
            or row.get("display_language")
            or row.get("language")
            or "unknown"
        )
        language_label = (
            "未检测"
            if language_value == "__skipped__"
            else str(row.get("display_language") or language_value)
        )
        language_option = language_filters.setdefault(
            language_value,
            {"value": language_value, "label": language_label, "count": 0},
        )
        language_option["count"] = int(language_option["count"]) + 1

        risk_value = str(row.get("risk_level") or "unknown")
        risk_label = RISK_LEVEL_LABELS.get(risk_value, risk_value)
        risk_option = risk_filters.setdefault(
            risk_value,
            {"value": risk_value, "label": risk_label, "count": 0},
        )
        risk_option["count"] = int(risk_option["count"]) + 1

    risk_order = {
        key: index
        for index, key in enumerate(("critical", "high", "medium", "low", "safe", "unknown"))
    }
    result["file_language_filters"] = sorted(
        language_filters.values(),
        key=lambda item: (str(item["value"]) == "__skipped__", str(item["label"]).casefold()),
    )
    result["file_risk_filters"] = sorted(
        risk_filters.values(),
        key=lambda item: (risk_order.get(str(item["value"]), 99), str(item["label"])),
    )
    return result


def _project_skipped_files(project_result: dict[str, object]) -> list[dict[str, str]]:
    existing = project_result.get("skipped_files")
    if isinstance(existing, list):
        structured = [
            {
                "filename": str(item.get("filename") or ""),
                "reason": str(item.get("reason") or "受扫描安全限制"),
            }
            for item in existing
            if isinstance(item, dict) and item.get("filename")
        ]
        if structured:
            return structured

    skipped = []
    for warning_value in project_result.get("warnings") or []:
        warning = str(warning_value)
        for prefix, reason in _SKIPPED_WARNING_PREFIXES.items():
            if warning.startswith(prefix):
                filename = warning[len(prefix):].strip()
                if filename:
                    skipped.append({"filename": filename, "reason": reason})
                break
    return skipped


def _project_file_extension(filename: str) -> str:
    suffix = Path(filename.replace("\\", "/")).suffix.lower()
    return suffix or "__none__"


def _project_evidence_sort_key(
    item: dict[str, object],
) -> tuple[int, float, int]:
    basis_rank = {
        "ai_decision": 0,
        "ai_and_rule": 1,
        "ai_only": 2,
        "rule_only": 3,
    }.get(str(item.get("evidence_basis") or "rule_only"), 3)
    try:
        score = float(item.get("suspicion_score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        line = int(item.get("line") or 0)
    except (TypeError, ValueError):
        line = 0
    return basis_rank, -score, line


@attack_bp.route("/api/jobs/<job_id>")
@login_required
def job_status(job_id: str):
    job = scan_jobs.get(job_id)
    if job is None:
        job = get_scan_job(job_id, current_user.username)
    job_username = job.get("username") if isinstance(job, dict) else getattr(job, "username", None)
    if not job or job_username != current_user.username:
        abort(404)
    return jsonify(_job_payload(job))


@attack_bp.route("/api/jobs")
@login_required
def job_list():
    jobs = [
        _job_payload(job, include_result=False)
        for job in scan_jobs.list(current_user.username)
    ]
    return jsonify({"jobs": jobs})


@attack_bp.route("/api/jobs/<job_id>/cancel", methods=["POST"])
@login_required
def cancel_job(job_id: str):
    job = scan_jobs.cancel(job_id, current_user.username)
    if job is None:
        job = cancel_persisted_scan_job(job_id, current_user.username)
        if job is None:
            abort(404)
    return jsonify(_job_payload(job))


def _job_payload(job, include_result: bool = True) -> dict[str, object]:
    if isinstance(job, dict):
        payload = {
            "id": job.get("id"), "username": job.get("username"), "mode": job.get("mode"),
            "target_type": job.get("target_type", "project"),
            "target_name": job.get("target_name"), "status": job.get("status"), "stage": job.get("stage"),
            "processed_files": job.get("processed_files", 0), "total_files": job.get("total_files", 0),
            "created_at": job.get("created_at"), "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"), "error": job.get("error"),
            "risk_score": job.get("risk_score"), "final_decision": job.get("final_decision"),
        }
        if include_result:
            payload["result"] = job.get("result")
        return payload
    payload = {
        "id": job.id,
        "username": job.username,
        "mode": job.mode,
        "target_type": job.target_type,
        "target_name": job.target_name,
        "status": job.status,
        "stage": job.stage,
        "processed_files": job.processed_files,
        "total_files": job.total_files,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "risk_score": (
            (job.result or {}).get("risk_score", (job.result or {}).get("max_score"))
            if job.result else None
        ),
        "final_decision": (job.result or {}).get("final_decision") if job.result else None,
    }
    if include_result:
        payload["result"] = job.result
    return payload


def _latest_active_scan_job(username: str, target_type: str):
    return next(
        (
            job for job in scan_jobs.list(username)
            if job.target_type == target_type
            and job.status in {"queued", "running", "cancelling"}
        ),
        None,
    )


def _persist_project_job(job) -> None:
    payload = _job_payload(job)
    save_scan_job(payload)
    if job.status == "completed" and job.result:
        save_project_results(job.id, job.username, job.result)


def _persist_file_job(job) -> None:
    if job.status == "completed" and job.result and not job.result.get("_record_id"):
        job.result["_record_id"] = save_detection(job.username, job.result)
    save_scan_job(_job_payload(job))


@attack_bp.route("/history")
@login_required
def history():
    all_records = list_record_summaries(current_user.username, 500)
    for item in all_records:
        item["display_language"] = display_language(
            str(item.get("language") or "unknown"),
            str(item.get("filename") or ""),
        )
        extension = Path(str(item.get("filename") or "")).suffix.lower()
        item["file_type"] = extension or "__none__"
        item["file_type_label"] = extension.upper() if extension else "无后缀"

    all_project_jobs = list_scan_jobs(
        current_user.username,
        100,
        include_result=False,
        target_type="project",
    )
    for project_job in all_project_jobs:
        project_risk_score = project_job.get("risk_score")
        project_job["risk_level"] = (
            score_risk_level(int(project_risk_score))
            if project_risk_score is not None
            else "unknown"
        )

    allowed_risks = {"critical", "high", "medium", "low", "safe"}
    query = request.args.get("q", "").strip()[:160]
    query_folded = query.casefold()
    project_risk = request.args.get("project_risk", "").strip().lower()
    file_risk = request.args.get("file_risk", "").strip().lower()
    file_type = request.args.get("file_type", "").strip().lower()
    if project_risk not in allowed_risks:
        project_risk = ""
    if file_risk not in allowed_risks:
        file_risk = ""

    file_type_options = sorted(
        {(item["file_type"], item["file_type_label"]) for item in all_records},
        key=lambda option: (option[0] == "__none__", option[1]),
    )
    allowed_file_types = {value for value, _label in file_type_options}
    if file_type not in allowed_file_types:
        file_type = ""

    filtered_projects = all_project_jobs
    if query_folded:
        filtered_projects = [
            item for item in filtered_projects
            if query_folded in str(item.get("target_name") or "").casefold()
            or query_folded in str(item.get("id") or "").casefold()
        ]
    if project_risk:
        filtered_projects = [
            item for item in filtered_projects
            if item.get("risk_level") == project_risk
        ]

    filtered_records = all_records
    if query_folded:
        filtered_records = [
            item for item in filtered_records
            if query_folded in str(item.get("filename") or "").casefold()
            or query_folded in str(item.get("file_hash") or "").casefold()
            or query_folded in str(item.get("id") or "").casefold()
        ]
    if file_risk:
        filtered_records = [
            item for item in filtered_records
            if item.get("risk_level") == file_risk
        ]
    if file_type:
        filtered_records = [
            item for item in filtered_records
            if item.get("file_type") == file_type
        ]

    project_jobs, project_pagination = _paginate_history_rows(
        filtered_projects,
        request.args.get("project_page", "1"),
    )
    records, file_pagination = _paginate_history_rows(
        filtered_records,
        request.args.get("file_page", "1"),
    )
    return render_template(
        "attack/history.html",
        records=records,
        project_jobs=project_jobs,
        project_pagination=project_pagination,
        file_pagination=file_pagination,
        file_type_options=file_type_options,
        filters={
            "q": query,
            "project_risk": project_risk,
            "file_risk": file_risk,
            "file_type": file_type,
        },
    )


def _paginate_history_rows(rows: list[dict], requested_page: str, page_size: int = 10):
    try:
        page = max(1, int(requested_page))
    except (TypeError, ValueError):
        page = 1
    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    return rows[start:start + page_size], {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "previous_page": max(1, page - 1),
        "next_page": min(total_pages, page + 1),
    }


@attack_bp.route("/history/compare")
@login_required
def compare_records():
    try:
        left_id = int(request.args.get("left", ""))
        right_id = int(request.args.get("right", ""))
    except ValueError:
        flash("请选择两条检测记录后再进行对比。")
        return redirect(url_for("attack.history"))
    if left_id == right_id:
        flash("请选择两条不同的检测记录。")
        return redirect(url_for("attack.history"))
    left = get_record(left_id, current_user.username)
    right = get_record(right_id, current_user.username)
    if not left or not right:
        abort(404)
    left = _public_record_view(left)
    right = _public_record_view(right)
    return render_template("attack/compare.html", left=left, right=right)


@attack_bp.route("/record/<int:record_id>")
@login_required
def record(record_id: int):
    record_data = get_record(record_id, current_user.username)
    if not record_data:
        abort(404)
    public_record = _public_record_view(record_data)
    return render_template(
        "attack/record.html",
        record=public_record,
        auxiliary_analysis=_auxiliary_analysis_view(public_record),
        file_report_insights=build_file_report_insights(
            public_record,
            model_center_view(),
        ),
    )


@attack_bp.route("/record/<int:record_id>/report.md")
@login_required
def record_report(record_id: int):
    record_data = get_record(record_id, current_user.username)
    if not record_data:
        abort(404)
    filename = f"xiezhi-report-{record_id}.md"
    return Response(
        render_record_markdown(_public_record_view(record_data)),
        mimetype="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@attack_bp.route("/models")
@login_required
def models():
    center = model_center_view()
    training_models = _training_model_options(center)
    jobs = list_training_jobs(None if current_user.role == "admin" else current_user.username, 50)
    return render_template(
        "attack/models.html",
        version_groups=center["version_groups"],
        runtime_models=_visible_runtime_models(runtime_status()),
        training_jobs=jobs,
        training_models=training_models,
        training_active=training_jobs.has_active_jobs(),
    )


@attack_bp.route("/models/train", methods=["POST"])
@login_required
def train_models():
    if current_user.role != "admin":
        abort(403)
    model_key = request.form.get("model_version", "")
    model_options = {item["key"]: item for item in _training_model_options(model_center_view())}
    selected_model = model_options.get(model_key)
    if selected_model is None:
        abort(400)
    training_task = {
        "gatv2": "project_malicious_intent",
    }.get(selected_model["family"], "malicious_intent")
    target_language = "all"
    upload = request.files.get("training_file")
    if not upload or not upload.filename:
        flash("请选择本地训练集文件。")
        return redirect(url_for("attack.models", _anchor="training-jobs"))
    try:
        dataset_path, dataset_name = _stage_training_dataset(upload, selected_model["family"])
    except (OSError, UnicodeError, ValueError) as exc:
        flash(f"训练集上传失败：{exc}")
        return redirect(url_for("attack.models", _anchor="training-jobs"))
    try:
        family = TRAINING_MODEL_FAMILIES[selected_model["family"]]
        trainer = family["trainer"]
        if selected_model["family"] == "codet5p":
            trainer = partial(
                train_codet5p,
                base_version=selected_model["version"],
                task=training_task,
                target_language=target_language,
            )
        job = training_jobs.submit(
            username=current_user.username,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            engine_name=selected_model["family_label"],
            base_version=selected_model["version"],
            model_family=selected_model["family"],
            training_task=training_task,
            target_language=target_language,
            trainer=trainer,
            preflight=partial(
                run_training_poisoning_gate,
                model_family=selected_model["family"],
            ),
            on_complete=family["reload"],
        )
        flash(
            f"训练任务 {job.id[:12]} 已提交：{selected_model['family_label']} / "
            f"{selected_model['version']}，训练集为 {dataset_name}；"
            "任务将先执行投毒检测，通过后才开始模型训练。"
        )
    except RuntimeError as exc:
        dataset_path.unlink(missing_ok=True)
        try:
            dataset_path.parent.rmdir()
        except OSError:
            pass
        flash(str(exc))
    return redirect(url_for("attack.models", _anchor="training-jobs"))


@attack_bp.route("/models/activate/<version>", methods=["POST"])
@login_required
def activate_model(version: str):
    if current_user.role != "admin":
        abort(403)
    try:
        activate_version(version)
        classifier.reload()
        flash(f"模型版本 {version} 已发布并重新加载。")
    except (OSError, ValueError) as exc:
        flash(f"模型版本切换失败：{exc}")
    return redirect(url_for("attack.models", _anchor="versions"))


@attack_bp.route("/models/xgboost/activate/<version>", methods=["POST"])
@login_required
def activate_xgb_model(version: str):
    if current_user.role != "admin":
        abort(403)
    try:
        activate_xgb_version(version)
        flash(f"XGBoost 版本 {version} 已发布并将在下次推理时重新加载。")
    except (OSError, ValueError) as exc:
        flash(f"XGBoost 版本切换失败：{exc}")
    return redirect(url_for("attack.models", _anchor="versions"))


@attack_bp.route("/evaluation")
@login_required
def evaluation():
    return redirect(url_for("attack.models", _anchor="versions"))


@attack_bp.route("/stats")
@login_required
def stats():
    return redirect(url_for("main.dashboard", _anchor="risk-statistics"))


@attack_bp.route("/api/stats")
@login_required
def stats_api():
    return jsonify(get_statistics(current_user.username))


def _selected_mode() -> str:
    mode = request.form.get("mode", "auto")
    return mode if mode in MODE_LABELS else "auto"


def _public_record_view(record_data: dict[str, object]) -> dict[str, object]:
    """Hide archived vulnerability-task fields without deleting stored history."""

    data = dict(record_data)
    data["display_language"] = display_language(
        str(data.get("language") or "unknown"),
        str(data.get("filename") or ""),
    )
    data["rule_matches"] = [
        match for match in (data.get("rule_matches") or [])
        if isinstance(match, dict) and is_active_finding(match)
    ]
    if not (
        data.get("final_decision") == "malicious"
        and data.get("decision_authority") == "ai"
    ):
        data["rule_matches"] = []
    data["engines"] = [
        engine for engine in (data.get("engines") or [])
        if not (
            isinstance(engine, dict)
            and (
                str(engine.get("name") or "") == "xgboost_vulnerability"
                or str((engine.get("metadata") or {}).get("task") or "") == "vulnerability_risk"
            )
        )
    ]
    for match in data["rule_matches"]:
        match.setdefault(
            "harm",
            match.get("description") or "该位置可能引入代码或配置安全风险。",
        )
        remediation = remediation_for_finding(
            match,
            str(data.get("language") or "unknown"),
        )
        if not match.get("cwe"):
            match["cwe"] = remediation.get("cwe")
        if not match.get("cve_examples"):
            match["cve_examples"] = remediation.get("cve_examples") or []
        if not match.get("repair_suggestions"):
            match["repair_suggestions"] = remediation.get("suggestions") or []
        if not match.get("remediation_references"):
            match["remediation_references"] = remediation.get("references") or []
        taxonomy = taxonomy_for_category(str(match.get("category") or ""))
        match.setdefault("risk_domains", taxonomy["risk_domains"])
        match.setdefault(
            "api_security_category",
            taxonomy["api_security_category"],
        )
    merged_evidence, ai_only_evidence = merge_model_line_attributions(
        data["rule_matches"],
        data["engines"],
    )
    data["rule_matches"] = merged_evidence
    data["evidence_items"] = order_evidence_items(
        merged_evidence,
        ai_only_evidence,
    )
    data["ai_explainability"] = build_ai_explainability(
        data["engines"],
        data["evidence_items"],
        len(ai_only_evidence),
        data,
    )
    votes = dict(data.get("engine_votes") or {})
    votes.pop("vulnerability_model", None)
    xgb_votes = votes.get("xgboost")
    if isinstance(xgb_votes, dict):
        votes["xgboost"] = {
            key: value for key, value in xgb_votes.items()
            if str(key) != "vulnerability_risk"
        }
    data["engine_votes"] = votes
    data["vulnerability_label"] = "disabled"
    data["vulnerability_probability"] = None
    suggestions: list[str] = []
    seen_suggestions: set[str] = set()
    remediation_references: list[dict[str, str]] = []
    seen_reference_urls: set[str] = set()
    for match in data["rule_matches"]:
        advice_values = match.get("repair_suggestions") or [match.get("repair_advice")]
        if isinstance(advice_values, str):
            advice_values = [advice_values]
        for advice_value in advice_values:
            advice = " ".join(str(advice_value or "").split())
            if advice and advice not in seen_suggestions:
                suggestions.append(advice)
                seen_suggestions.add(advice)
        for reference in match.get("remediation_references") or []:
            if not isinstance(reference, dict):
                continue
            url = str(reference.get("url") or "")
            if not url or url in seen_reference_urls:
                continue
            remediation_references.append({
                "title": str(reference.get("title") or "修复依据"),
                "url": url,
            })
            seen_reference_urls.add(url)
    data["repair_suggestions"] = suggestions
    data["remediation_references"] = remediation_references
    return data


def _scan_staged_project(
    path: Path,
    filename: str,
    mode: str,
    line_explanations: bool,
    cancel_event,
    progress,
) -> dict[str, object]:
    try:
        return scan_zip_project(
            path,
            filename,
            mode=mode,
            cancel_event=cancel_event,
            progress_callback=progress,
            generate_line_attributions=line_explanations,
        )
    finally:
        path.unlink(missing_ok=True)


def _scan_single_file(
    filename: str,
    payload: bytes,
    mode: str,
    cancel_event,
    progress,
) -> dict[str, object]:
    progress(10, 100, "正在准备文件")
    raise_if_cancelled(cancel_event)
    progress(30, 100, "AI 模型检测中")
    result = scan_file(
        filename,
        payload,
        mode=mode,
        cancel_event=cancel_event,
    )
    raise_if_cancelled(cancel_event)
    progress(90, 100, "正在生成检测报告")
    progress(100, 100, "检测完成")
    return result


def _mode_availability() -> dict[str, dict[str, object]]:
    by_engine = {item["engine"]: item for item in runtime_status()}
    requirements = {
        "quick": ("xgboost",),
        "standard": ("xgboost", "codet5p"),
        "deep": ("xgboost", "codet5p", "gatv2"),
    }
    output = {
        "auto": {"available": True, "reason": "按实际证据自动升级；不可用引擎会明确标记"},
    }
    for mode, engines in requirements.items():
        missing = [by_engine[name]["name"] for name in engines if by_engine[name]["status"] != "completed"]
        output[mode] = {
            "available": not missing,
            "reason": "全部所需引擎已加载" if not missing else "未加载：" + "、".join(missing),
        }
    return output
