"""Project archive scanner."""

from __future__ import annotations

import hashlib
import os
import json
import multiprocessing
import re
import shutil
import threading
import zipfile
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Callable
from uuid import uuid4

from .data_pipeline import make_sample
from .cross_file_analysis import analyze_cross_file_project
from .engines.codet5p_engine import CodeT5PEngine
from .engines.gat_engine import GATEngine
from .engines.rule_engine import RuleEngine
from .explainability import AI_ONLY_CATEGORY
from .features.graph_builder import (
    build_project_graph,
    build_project_relationship_graph,
)
from .fusion import fuse_engine_results
from .languages import (
    decode_source_payload,
    display_language,
    is_generic_text_path,
    is_probably_text_payload,
)
from .scanner import (
    detect_language,
    is_allowed_file,
    prepare_xgb_batch,
    scan_file,
    scan_xgb_attribution_batch,
    scan_xgb_prepared,
)
from .static_analysis.engine import StaticAnalysisEngine

MAX_FILES = 500
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_ARCHIVE_SIZE = 1024 * 1024 * 1024
MAX_ZIP_MEMBERS = 50_000
# Keep decompressed project data bounded independently from the per-file
# allowance. This prevents a ZIP bomb from multiplying the 100 MiB limit by
# every permitted member.
MAX_TOTAL_EXTRACTED_SIZE = MAX_ARCHIVE_SIZE
MAX_COMPRESSION_RATIO = 100
COPY_CHUNK_SIZE = 1024 * 1024
MAX_WARNINGS = 100
DEEP_FILE_LIMIT = 12
CODET5_PROJECT_FILE_LIMIT = max(
    1,
    min(
        DEEP_FILE_LIMIT,
        int(os.environ.get(
            "XIEZHI_CODET5_PROJECT_MAX_FILES",
            "1" if os.name == "nt" else str(DEEP_FILE_LIMIT),
        )),
    ),
)
QUICK_SCAN_WORKERS = max(
    1,
    min(
        8,
        int(os.environ.get(
            "XIEZHI_PROJECT_SCAN_WORKERS",
            "4" if os.name == "nt" else "1",
        )),
    ),
)
PROJECT_ANALYSIS_MAX_BYTES = max(
    32 * 1024,
    min(1024 * 1024, int(os.environ.get("XIEZHI_PROJECT_ANALYSIS_MAX_BYTES", str(64 * 1024)))),
)
XGB_BATCH_MIN_FILES = max(
    2,
    int(os.environ.get("XIEZHI_XGB_BATCH_MIN_FILES", "8")),
)
QUICK_EVIDENCE_PROCESS_WORKERS = max(
    1,
    min(
        12,
        int(os.environ.get(
            "XIEZHI_QUICK_EVIDENCE_PROCESS_WORKERS",
            str(os.cpu_count() or 1),
        )),
    ),
)
QUICK_EVIDENCE_PROCESS_MIN_FILES = max(
    2,
    int(os.environ.get("XIEZHI_QUICK_EVIDENCE_PROCESS_MIN_FILES", "8")),
)
QUICK_EVIDENCE_THREAD_WORKERS = max(
    1,
    min(
        8,
        int(os.environ.get(
            "XIEZHI_QUICK_EVIDENCE_THREAD_WORKERS",
            str(max(2, min(8, (os.cpu_count() or 2) // 2))),
        )),
    ),
)
SCAN_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "tmp"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
WINDOWS_INVALID_COMPONENT_PATTERN = re.compile(
    r'[<>:"/\\|?*\x00-\x1f]',
)
WINDOWS_RESERVED_COMPONENT_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
MAX_SAFE_PATH_COMPONENT_LENGTH = 120


class ArchiveTooLargeError(ValueError):
    pass


class _ProjectProgress:
    """Translate per-stage counters into one monotonic project percentage."""

    _RANGES = (
        ("准备项目文件", 1, 2),
        ("解压项目文件", 2, 10),
        ("读取项目文件", 10, 15),
        ("加载 XGBoost 模型", 15, 18),
        ("提取 XGBoost 特征", 18, 42),
        ("XGBoost 批量推理", 42, 50),
        ("XGBoost AI快速初筛", 50, 65),
        ("规则与静态解释筛选", 65, 72),
        ("快速分析", 72, 78),
        ("整理检测报告", 96, 99),
    )
    _DEEP_LABELS = {
        "codet5p": "CodeT5+",
        "attribution": "AI行级归因",
        "graph": "项目图",
    }

    def __init__(
        self,
        callback: Callable[[int, int, str], None] | None,
    ) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self._highest = 0
        self._deep: dict[str, float] = {}

    def set_deep_branches(self, branches: set[str]) -> None:
        with self._lock:
            self._deep = {name: 0.0 for name in branches}
        if branches:
            self._emit(
                78,
                "候选复核（"
                + "、".join(
                    self._DEEP_LABELS[name]
                    for name in sorted(branches)
                )
                + "）",
            )

    def __call__(self, done: int, total: int, stage: str) -> None:
        branch = self._deep_branch(stage)
        if branch and branch in self._deep:
            fraction = _progress_fraction(done, total)
            with self._lock:
                self._deep[branch] = max(
                    self._deep[branch],
                    fraction,
                )
                combined = sum(self._deep.values()) / len(self._deep)
                waiting = [
                    self._DEEP_LABELS[name]
                    for name, value in self._deep.items()
                    if value < 1.0
                ]
            label = (
                "候选复核：等待" + "、".join(waiting)
                if waiting
                else "候选复核完成"
            )
            self._emit(78 + round(18 * combined), label)
            return
        for prefix, start, end in self._RANGES:
            if stage.startswith(prefix):
                self._emit(
                    start + round(
                        (end - start)
                        * _progress_fraction(done, total)
                    ),
                    stage,
                )
                return
        self._emit(self._highest, stage)

    def finish(self) -> None:
        self._emit(100, "检测结果已整理")

    def _emit(self, percent: int, stage: str) -> None:
        if self._callback is None:
            return
        with self._lock:
            self._highest = max(
                self._highest,
                min(100, max(0, int(percent))),
            )
            current = self._highest
        self._callback(current, 100, stage)

    @staticmethod
    def _deep_branch(stage: str) -> str | None:
        if stage.startswith("CodeT5+"):
            return "codet5p"
        if stage.startswith("生成候选文件行级解释"):
            return "attribution"
        if stage.startswith("构建项目图并执行 GATv2"):
            return "graph"
        return None


def _progress_fraction(done: int, total: int) -> float:
    if int(total) <= 0:
        return 0.0
    return min(1.0, max(0.0, int(done) / int(total)))


def _scan_quick_evidence_worker(
    request: tuple[str, str, bytes],
) -> list[dict[str, object]]:
    """Run the two CPU-heavy, model-independent quick engines."""

    content, language, payload = request
    return [
        RuleEngine().scan(content, language),
        StaticAnalysisEngine().scan(
            content,
            language,
            raw_bytes=payload,
        ),
    ]


def _start_quick_evidence_pool(
    record_count: int,
) -> object | None:
    """Start clean POSIX workers before the parent initializes model runtimes.

    Jobs are submitted only after the XGBoost prefilter has selected the files
    that actually need rule/static evidence.
    """

    if (
        os.name != "posix"
        or QUICK_EVIDENCE_PROCESS_WORKERS < 2
        or record_count < QUICK_EVIDENCE_PROCESS_MIN_FILES
    ):
        return None
    try:
        context = multiprocessing.get_context("fork")
        return context.Pool(
            processes=min(
                QUICK_EVIDENCE_PROCESS_WORKERS,
                record_count,
            ),
        )
    except (OSError, RuntimeError, ValueError):
        return None


def _finish_quick_evidence_pool(
    pool: object | None,
    records: list[dict[str, object]],
    cancel_event: object | None,
    progress_callback: Callable[[int, int, str], None] | None,
) -> list[list[dict[str, object]]] | None:
    if pool is None:
        return None
    total = len(records)
    if not records:
        pool.close()
        pool.join()
        return []
    iterator = pool.imap(
        _scan_quick_evidence_worker,
        [
            (
                str(record.get("content") or ""),
                str(record.get("language") or ""),
                bytes(record.get("payload") or b""),
            )
            for record in records
        ],
        chunksize=1,
    )
    results: list[list[dict[str, object]]] = []
    try:
        while len(results) < total:
            if _cancelled(cancel_event):
                pool.terminate()
                pool.join()
                return None
            try:
                results.append(iterator.next(timeout=0.05))
                _progress(
                    progress_callback,
                    len(results),
                    total,
                    "规则与静态解释筛选",
                )
            except multiprocessing.TimeoutError:
                continue
        pool.close()
        pool.join()
        return results
    except Exception:
        pool.terminate()
        pool.join()
        return None


def _discard_quick_evidence_pool(pool: object | None) -> None:
    if pool is None:
        return
    try:
        pool.terminate()
        pool.join()
    except Exception:
        return


def _run_quick_evidence_threads(
    records: list[dict[str, object]],
    cancel_event: object | None,
    progress_callback: Callable[[int, int, str], None] | None,
) -> list[list[dict[str, object]]] | None:
    """Parallelize rule/static explanation work on Windows.

    Windows lacks the low-overhead ``fork`` pool used on the GPU/Linux host.
    Threads avoid repeatedly importing the application and still improve the
    regex-heavy evidence pass without changing its detection coverage.
    """

    if (
        os.name != "nt"
        or QUICK_EVIDENCE_THREAD_WORKERS < 2
        or len(records) < QUICK_EVIDENCE_PROCESS_MIN_FILES
    ):
        return None
    output: list[list[dict[str, object]] | None] = [
        None
    ] * len(records)
    executor = ThreadPoolExecutor(
        max_workers=min(
            QUICK_EVIDENCE_THREAD_WORKERS,
            len(records),
        ),
        thread_name_prefix="project-evidence",
    )
    futures = {
        executor.submit(
            _scan_quick_evidence_worker,
            (
                str(record.get("content") or ""),
                str(record.get("language") or ""),
                bytes(record.get("payload") or b""),
            ),
        ): index
        for index, record in enumerate(records)
    }
    completed = 0
    cancelled = False
    try:
        for future in as_completed(futures):
            if _cancelled(cancel_event):
                cancelled = True
                for pending in futures:
                    pending.cancel()
                return None
            output[futures[future]] = future.result()
            completed += 1
            _progress(
                progress_callback,
                completed,
                len(records),
                "规则与静态解释筛选",
            )
    except Exception:
        for pending in futures:
            pending.cancel()
        return None
    finally:
        executor.shutdown(
            wait=not cancelled,
            cancel_futures=True,
        )
    return [
        item for item in output
        if item is not None
    ]


def _run_quick_evidence(
    records: list[dict[str, object]],
    cancel_event: object | None,
    progress_callback: Callable[[int, int, str], None] | None,
    *,
    process_pool: object | None = None,
) -> list[list[dict[str, object]]] | None:
    """Run full evidence extraction only for AI-selected files."""

    if not records:
        if process_pool is not None:
            process_pool.close()
            process_pool.join()
        return []
    output = _finish_quick_evidence_pool(
        process_pool,
        records,
        cancel_event,
        progress_callback,
    )
    if output is None and not _cancelled(cancel_event):
        output = _run_quick_evidence_threads(
            records,
            cancel_event,
            progress_callback,
        )
    if output is not None or _cancelled(cancel_event):
        return output
    output = []
    for index, record in enumerate(records, start=1):
        if _cancelled(cancel_event):
            return None
        output.append(_scan_quick_evidence_worker((
            str(record.get("content") or ""),
            str(record.get("language") or ""),
            bytes(record.get("payload") or b""),
        )))
        _progress(
            progress_callback,
            index,
            len(records),
            "规则与静态解释筛选",
        )
    return output


def _xgb_requires_full_evidence(
    engines: list[dict[str, object]],
) -> bool:
    """Use rules/static only when AI cannot make a reliable decision."""

    return _xgb_evidence_state(engines) == "fallback"


def _xgb_evidence_state(
    engines: list[dict[str, object]],
) -> str:
    malicious_engines = [
        engine
        for engine in engines
        if (
            str(engine.get("name") or "").startswith("xgboost_")
            and (engine.get("metadata") or {}).get("task")
            == "malicious_intent"
        )
    ]
    if not malicious_engines:
        return "fallback"
    labels: set[str] = set()
    for engine in malicious_engines:
        if (
            engine.get("status") != "completed"
            or engine.get("probability") is None
        ):
            return "fallback"
        metadata = engine.get("metadata") or {}
        if metadata.get("advisory_only"):
            return "fallback"
        probability = float(engine.get("probability") or 0.0)
        uncertain_low = metadata.get("uncertain_low")
        uncertain_high = metadata.get("uncertain_high")
        if (
            uncertain_low is not None
            and uncertain_high is not None
            and float(uncertain_low) <= probability <= float(uncertain_high)
        ):
            return "fallback"
        decision = str(
            metadata.get("raw_model_decision")
            or engine.get("decision")
            or ""
        )
        if decision not in {"malicious", "benign"}:
            return "fallback"
        labels.add(decision)
    if len(labels) != 1:
        return "fallback"
    return next(iter(labels))


def _attach_quick_evidence(
    record: dict[str, object],
    evidence: list[dict[str, object]],
) -> None:
    if len(evidence) != 2:
        return
    xgb_engines = [
        dict(engine)
        for engine in (
            record.get("precomputed_xgb_engines") or []
        )
        if isinstance(engine, dict)
    ]
    record["precomputed_quick_result"] = {
        "engines": [
            evidence[0],
            *xgb_engines,
            evidence[1],
        ],
    }
    record["quick_evidence_generated"] = True


def _budget_skipped_codet5() -> dict[str, object]:
    return {
        "name": "codet5p",
        "status": "skipped",
        "reason": (
            "CodeT5+ 220M was not run for this candidate because the "
            "local CPU project time budget prioritizes the highest-ranked "
            "semantic-review file"
        ),
        "probability": None,
        "metadata": {
            "primary_task": "malicious_intent",
            "time_budget_skip": True,
        },
    }


def stage_project_archive(fileobj: BinaryIO) -> Path:
    """Copy an upload to a bounded on-disk file without retaining it in memory."""
    SCAN_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = (SCAN_TEMP_ROOT / f"xiezhi_upload_{uuid4().hex}.zip").resolve()
    if SCAN_TEMP_ROOT.resolve() not in path.parents:
        raise ValueError("project upload path escaped its root")
    total = 0
    try:
        with path.open("xb") as output:
            while True:
                block = fileobj.read(COPY_CHUNK_SIZE)
                if not block:
                    break
                total += len(block)
                if total > MAX_ARCHIVE_SIZE:
                    raise ArchiveTooLargeError("ZIP 压缩包超过 1 GB 上传限制")
                output.write(block)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def scan_zip_project(
    source: BinaryIO | str | Path,
    original_filename: str = "project.zip",
    mode: str = "auto",
    cancel_event: object | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    generate_line_attributions: bool = True,
) -> dict[str, object]:
    project_progress = _ProjectProgress(progress_callback)
    progress_callback = project_progress
    _progress(progress_callback, 1, 1, "准备项目文件")
    SCAN_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_root = (SCAN_TEMP_ROOT / f"xiezhi_scan_{uuid4().hex}").resolve()
    if SCAN_TEMP_ROOT.resolve() not in temporary_root.parents:
        raise ValueError("project scan temporary path escaped its root")
    temporary_root.mkdir()
    staged_here = not isinstance(source, (str, Path))
    archive_path: Path | None = None
    results: list[dict[str, object]] = []
    warnings: list[str] = []
    deep_indices: list[int] = []
    semantic_indices: list[int] = []
    project_engines: list[dict[str, object]] = []
    project_relationship_graph: dict[str, object] | None = None
    project_cross_file_analysis: dict[str, object] | None = None
    try:
        try:
            archive_path = stage_project_archive(source) if staged_here else Path(source)
        except ArchiveTooLargeError:
            return summarize_project(original_filename, [], ["ZIP 压缩包超过 1 GB 上传限制，本次未执行检测。"])
        if not archive_path.is_file() or archive_path.stat().st_size > MAX_ARCHIVE_SIZE:
            return summarize_project(original_filename, [], ["ZIP 压缩包超过 1 GB 上传限制，本次未执行检测。"])
        extract_dir = temporary_root / "src"
        extract_dir.mkdir()
        extracted_path_names: dict[str, str] = {}
        try:
            with zipfile.ZipFile(archive_path) as archive:
                warnings.extend(_safe_extract(
                    archive,
                    extract_dir,
                    cancel_event,
                    progress_callback,
                    extracted_path_names,
                ))
        except (zipfile.BadZipFile, zipfile.LargeZipFile):
            return summarize_project(original_filename, [], ["上传的文件不是有效的 ZIP 压缩包。"])
        if _cancelled(cancel_event):
            return {"cancelled": True}
        records = []
        graph_samples = []
        source_paths = [
            path
            for path in extract_dir.rglob("*")
            if (
                path.is_file()
                and is_allowed_file(path.name)
                and path.stat().st_size <= MAX_FILE_SIZE
            )
        ][:MAX_FILES]
        _progress(
            progress_callback,
            0,
            len(source_paths),
            "读取项目文件",
        )
        for completed_count, path in enumerate(source_paths, start=1):
            if _cancelled(cancel_event):
                return {"cancelled": True}
            safe_rel = str(
                path.relative_to(extract_dir),
            ).replace("\\", "/")
            rel = extracted_path_names.get(safe_rel, safe_rel)
            payload = path.read_bytes()
            extension_language = detect_language(rel)
            content = (
                _bounded_source_content(payload, PROJECT_ANALYSIS_MAX_BYTES)
                if extension_language != "binary"
                else ""
            )
            language = detect_language(rel, content)
            records.append({
                "filename": rel,
                "content": content,
                "payload": payload,
                "language": language,
                "analysis_truncated": (
                    language != "binary"
                    and len(payload) > PROJECT_ANALYSIS_MAX_BYTES
                ),
            })
            # GATv2 is a deep-mode-only project model.  Building these samples
            # normalizes and hashes every source file, so doing it for auto,
            # standard, or quick scans only delays XGBoost without changing
            # their results.
            if mode == "deep" and language != "binary":
                graph_samples.append(make_sample(
                    content, label="benign", category="runtime_unlabeled", language=language,
                    source="runtime_project_scan", package_name=original_filename, version="uploaded",
                    family=f"runtime:{original_filename}", split="runtime", file_path=rel,
                    label_basis="unlabeled_runtime_graph; label field is excluded from GATv2 features",
                ))
            _progress(
                progress_callback,
                completed_count,
                len(source_paths),
                "读取项目文件",
            )
        if len(source_paths) >= MAX_FILES:
            warnings.append(f"文件数量达到 {MAX_FILES} 个，剩余文件已跳过。")
        total = len(records)
        source_positions = [
            index
            for index, record in enumerate(records)
            if record.get("language") != "binary"
        ]
        quick_evidence_pool = (
            _start_quick_evidence_pool(len(source_positions))
            if mode != "quick"
            else None
        )
        if (
            len(source_positions) >= XGB_BATCH_MIN_FILES
            and not _cancelled(cancel_event)
        ):
            try:
                prepared_xgb = prepare_xgb_batch(
                    [
                        {
                            "content": str(records[index]["content"]),
                            "language": str(records[index]["language"]),
                        }
                        for index in source_positions
                    ],
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
            except Exception:
                prepared_xgb = []
            if len(prepared_xgb) == len(source_positions):
                for record_index, prepared in zip(
                    source_positions,
                    prepared_xgb,
                ):
                    records[record_index]["precomputed_xgb"] = prepared
        if _cancelled(cancel_event):
            _discard_quick_evidence_pool(
                quick_evidence_pool,
            )
            return {"cancelled": True}
        evidence_positions: list[int] = []
        _progress(
            progress_callback,
            0,
            len(source_positions),
            "XGBoost AI快速初筛",
        )
        for completed_count, record_index in enumerate(
            source_positions,
            start=1,
        ):
            record = records[record_index]
            prepared = record.get("precomputed_xgb")
            try:
                xgb_engines = scan_xgb_prepared(
                    str(record.get("content") or ""),
                    str(record.get("language") or ""),
                    cancel_event=cancel_event,
                    precomputed_batch=(
                        prepared
                        if isinstance(prepared, dict)
                        else None
                    ),
                )
            except Exception as exc:
                xgb_engines = [{
                    "name": "xgboost_malicious",
                    "status": "failed",
                    "reason": "XGBoost AI prefilter failed",
                    "error": str(exc),
                    "probability": None,
                    "metadata": {"task": "malicious_intent"},
                }]
            if _cancelled(cancel_event):
                _discard_quick_evidence_pool(
                    quick_evidence_pool,
                )
                return {"cancelled": True}
            record["precomputed_xgb_engines"] = xgb_engines
            record["precomputed_quick_result"] = {
                "engines": [
                    dict(engine)
                    for engine in xgb_engines
                    if isinstance(engine, dict)
                ],
            }
            record["quick_evidence_generated"] = False
            record["xgb_evidence_state"] = (
                _xgb_evidence_state(xgb_engines)
            )
            if (
                mode != "quick"
                and record["xgb_evidence_state"] == "fallback"
            ):
                evidence_positions.append(record_index)
            _progress(
                progress_callback,
                completed_count,
                len(source_positions),
                "XGBoost AI快速初筛",
            )

        deep_languages: set[str] = set()
        planned_deep_indices: list[int] = []
        if mode in {"standard", "deep", "auto"} and records:
            deep_languages = _deep_languages()
            prefilter_results = [
                {
                    "engines": (
                        record.get(
                            "precomputed_xgb_engines",
                        )
                        or []
                    ),
                }
                for record in records
            ]
            if mode == "auto":
                auto_candidates = [
                    index
                    for index in evidence_positions
                    if records[index].get("language")
                    in deep_languages
                ]
                planned_deep_indices = sorted(
                    auto_candidates,
                    key=lambda index: (
                        _ai_candidate_priority(
                            prefilter_results[index],
                        ),
                        str(records[index]["filename"]),
                    ),
                    reverse=True,
                )[:DEEP_FILE_LIMIT]
            else:
                planned_deep_indices = _select_deep_candidates(
                    records,
                    prefilter_results,
                    DEEP_FILE_LIMIT,
                    deep_languages,
                )

        quick_evidence = _run_quick_evidence(
            [
                records[index]
                for index in evidence_positions
            ],
            cancel_event,
            progress_callback,
            process_pool=quick_evidence_pool,
        )
        if _cancelled(cancel_event):
            return {"cancelled": True}
        if (
            quick_evidence is not None
            and len(quick_evidence) == len(evidence_positions)
        ):
            for record_index, evidence in zip(
                evidence_positions,
                quick_evidence,
            ):
                _attach_quick_evidence(
                    records[record_index],
                    evidence,
                )
        _progress(progress_callback, 0, total, "快速分析")
        def quick_scan(record: dict[str, object]) -> dict[str, object]:
            scan_kwargs: dict[str, object] = {
                "mode": "quick",
                "cancel_event": cancel_event,
                "generate_line_attributions": False,
                "analysis_max_bytes": PROJECT_ANALYSIS_MAX_BYTES,
                "run_legacy_baseline": False,
            }
            if isinstance(record.get("precomputed_xgb"), dict):
                scan_kwargs["precomputed_xgb"] = record["precomputed_xgb"]
            if isinstance(
                record.get("precomputed_quick_result"),
                dict,
            ):
                scan_kwargs["precomputed_quick_result"] = record[
                    "precomputed_quick_result"
                ]
            return scan_file(
                str(record["filename"]),
                bytes(record.get("payload") or b""),
                **scan_kwargs,
            )

        ordered_results: list[dict[str, object] | None] = [None] * total
        with ThreadPoolExecutor(
            max_workers=min(QUICK_SCAN_WORKERS, max(1, total)),
            thread_name_prefix="project-quick",
        ) as executor:
            futures = {
                executor.submit(quick_scan, record): index
                for index, record in enumerate(records)
            }
            completed_count = 0
            for future in as_completed(futures):
                if _cancelled(cancel_event):
                    for pending in futures:
                        pending.cancel()
                    return {"cancelled": True}
                record_index = futures[future]
                result = future.result()
                if mode != "quick":
                    result["selected_mode"] = mode
                    result["project_scan_note"] = "项目快速初筛；仅候选文件进入深度模型"
                ordered_results[record_index] = result
                completed_count += 1
                _progress(progress_callback, completed_count, total, "快速分析")
        results = [result for result in ordered_results if result is not None]

        graph_result: dict[str, object] | None = None
        if mode in {"standard", "deep", "auto"} and records:
            deep_indices = planned_deep_indices
            requests = [
                {"content": str(records[index]["content"]), "language": str(records[index]["language"])}
                for index in deep_indices
            ]
            semantic_indices = (
                list(deep_indices)
                if mode == "auto"
                else list(
                    deep_indices[
                        :CODET5_PROJECT_FILE_LIMIT
                    ]
                )
            )
            semantic_requests = [
                {
                    "content": str(records[index]["content"]),
                    "language": str(records[index]["language"]),
                }
                for index in semantic_indices
            ]
            if requests and not _cancelled(cancel_event):
                active_deep_branches = set()
                if semantic_requests:
                    active_deep_branches.add("codet5p")
                if generate_line_attributions:
                    active_deep_branches.add("attribution")
                if mode == "deep" and graph_samples:
                    active_deep_branches.add("graph")
                project_progress.set_deep_branches(
                    active_deep_branches,
                )

                def semantic_branch() -> list[dict[str, object]]:
                    _progress(
                        progress_callback,
                        0,
                        len(semantic_requests),
                        "CodeT5+ 220M 批量复核",
                    )
                    output = CodeT5PEngine().scan_batch(
                        semantic_requests,
                        cancel_event=cancel_event,
                    )
                    _progress(
                        progress_callback, len(output), len(output),
                        "CodeT5+ 220M 批量复核完成",
                    )
                    return output

                def explanation_branch() -> list[dict[str, object]]:
                    _progress(
                        progress_callback,
                        0,
                        len(requests),
                        "生成候选文件行级解释",
                    )
                    with _windows_high_core_affinity(4):
                        explained_batches = scan_xgb_attribution_batch(
                            requests,
                            prepared_batch=[
                                (
                                    records[index].get("precomputed_xgb")
                                    if isinstance(
                                        records[index].get("precomputed_xgb"),
                                        dict,
                                    )
                                    else {}
                                )
                                for index in deep_indices
                            ],
                            cancel_event=cancel_event,
                        )
                    explained = []
                    for record_index, explained_xgb in zip(
                        deep_indices,
                        explained_batches,
                    ):
                        quick_result = dict(results[record_index])
                        quick_result["engines"] = _replace_xgb_engines(
                            list(quick_result.get("engines") or []),
                            explained_xgb,
                        )
                        explained.append(quick_result)
                    _progress(
                        progress_callback,
                        len(explained),
                        len(requests),
                        "生成候选文件行级解释",
                    )
                    return explained

                def graph_branch(
                    graph: dict[str, object] | None,
                ) -> dict[str, object] | None:
                    if graph is None:
                        return None
                    output = GATEngine().scan_project(
                        graph, cancel_event=cancel_event,
                    )
                    _progress(progress_callback, 1, 1, "构建项目图并执行 GATv2")
                    return output

                with ThreadPoolExecutor(
                    max_workers=3,
                    thread_name_prefix="project-deep",
                ) as deep_executor:
                    semantic_future = (
                        deep_executor.submit(semantic_branch)
                        if semantic_requests
                        else None
                    )
                    graph = None
                    if graph_samples and mode == "deep":
                        _progress(
                            progress_callback,
                            0,
                            1,
                            "构建项目图并执行 GATv2",
                        )
                        # Graph construction is Python/regex heavy. Building it
                        # while the GPU semantic worker runs, then starting
                        # XGBoost explanations and GAT inference together,
                        # avoids making graph extraction and line explanation
                        # fight for the same interpreter lock.
                        graph = build_project_graph(graph_samples)
                        project_relationship_graph = _build_project_relationship_view(
                            graph_samples,
                            results,
                            warnings,
                        )
                    explanation_future = (
                        deep_executor.submit(explanation_branch)
                        if generate_line_attributions
                        else None
                    )
                    graph_future = deep_executor.submit(
                        graph_branch,
                        graph,
                    )
                    semantic_results = (
                        semantic_future.result()
                        if semantic_future is not None
                        else []
                    )
                    explained_results = (
                        explanation_future.result()
                        if explanation_future is not None
                        else [
                            dict(results[index])
                            for index in deep_indices
                        ]
                    )
                    graph_result = graph_future.result()
                if _cancelled(cancel_event):
                    return {"cancelled": True}
                semantic_by_index = dict(zip(
                    semantic_indices,
                    semantic_results,
                ))
                explained_by_index = dict(zip(
                    deep_indices,
                    explained_results,
                ))
                for record_index in deep_indices:
                    record = records[record_index]
                    results[record_index] = scan_file(
                        str(record["filename"]), bytes(record.get("payload") or b""), mode=mode,
                        precomputed_semantic=semantic_by_index.get(
                            record_index,
                            _budget_skipped_codet5(),
                        ),
                        precomputed_quick_result=explained_by_index[
                            record_index
                        ],
                        cancel_event=cancel_event,
                        generate_line_attributions=generate_line_attributions,
                        analysis_max_bytes=PROJECT_ANALYSIS_MAX_BYTES,
                        run_legacy_baseline=False,
                    )
            skipped = sum(
                record["language"] in deep_languages
                for record in records
            ) - len(deep_indices)
            if skipped > 0:
                if mode == "auto":
                    warnings.append(
                        f"自动模式仅将 {len(deep_indices)} 个AI不确定文件交给"
                        f"CodeT5+ 220M复核；其余 {skipped} 个文件保留"
                        "XGBoost明确结论。"
                    )
                else:
                    warnings.append(
                        f"为控制扫描时长，CodeT5+ 220M 实际复核"
                        f" {len(semantic_indices)} 个候选文件，XGBoost为"
                        f" {len(deep_indices)} 个候选文件生成行级归因；"
                        f"其余 {skipped} 个受支持语言文件保留快速模式结果。"
                    )
        truncated_count = sum(
            bool(record.get("analysis_truncated")) for record in records
        )
        if truncated_count:
            warnings.append(
                f"{truncated_count} 个大文件采用头尾分块快速分析；"
                f"每个文件最多分析 {PROJECT_ANALYSIS_MAX_BYTES // 1024} KiB，"
                "完整原始字节仍用于文件哈希。"
            )
        if mode in {"standard", "deep", "auto"} and records:
            _progress(progress_callback, 0, 1, "解析跨文件调用与数据流")
            try:
                project_cross_file_analysis = analyze_cross_file_project(records)
                _attach_cross_file_findings(
                    results,
                    project_cross_file_analysis,
                )
            except Exception as exc:
                _warn(
                    warnings,
                    f"跨文件调用与数据流分析失败，本次AI模型结果不受影响：{exc}",
                )
            _progress(progress_callback, 1, 1, "解析跨文件调用与数据流")
        project_engines = aggregate_project_xgboost(results)
        if graph_result is not None:
            project_engines.append(graph_result)
        elif (
            graph_samples
            and mode == "deep"
            and not _cancelled(cancel_event)
        ):
            _progress(progress_callback, 0, 1, "构建项目图并执行 GATv2")
            graph = build_project_graph(graph_samples)
            project_relationship_graph = _build_project_relationship_view(
                graph_samples,
                results,
                warnings,
            )
            project_engines.append(
                GATEngine().scan_project(graph, cancel_event=cancel_event)
            )
            _progress(progress_callback, 1, 1, "构建项目图并执行 GATv2")
        if project_cross_file_analysis is not None:
            _merge_gat_component_attribution(
                project_cross_file_analysis,
                project_engines,
            )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if staged_here and archive_path is not None:
            archive_path.unlink(missing_ok=True)
        summary = summarize_project(
            original_filename,
            results,
            warnings,
            project_engines,
            project_relationship_graph,
            project_cross_file_analysis,
        )
    summary["scan_strategy"] = "all_files_quick_then_batched_candidate_deep"
    summary["deep_scanned_file_count"] = len(deep_indices)
    summary["semantic_scanned_file_count"] = len(
        semantic_indices
    )
    summary["quick_only_file_count"] = len(results) - len(deep_indices)
    summary["line_explanations_enabled"] = bool(
        generate_line_attributions
        and mode != "quick"
        and deep_indices
    )
    evidence_analyzed_count = sum(
        bool(record.get("quick_evidence_generated"))
        for record in records
        if record.get("language") != "binary"
    )
    summary["evidence_strategy"] = (
        "xgboost_only"
        if mode == "quick"
        else "xgboost_ai_line_attribution_then_rule_static_fallback"
    )
    summary["rule_static_analyzed_file_count"] = (
        evidence_analyzed_count
    )
    summary["rule_static_skipped_file_count"] = max(
        0, len(source_positions) - evidence_analyzed_count,
    )
    summary["ai_confident_benign_evidence_skipped_count"] = sum(
        record.get("xgb_evidence_state") == "benign"
        and not bool(record.get("quick_evidence_generated"))
        for record in records
        if record.get("language") != "binary"
    )
    summary[
        "ai_decisive_malicious_rule_static_skipped_count"
    ] = sum(
        record.get("xgb_evidence_state") == "malicious"
        and not bool(record.get("quick_evidence_generated"))
        for record in records
        if record.get("language") != "binary"
    )
    summary["automatic_effective_mode"] = (
        "standard"
        if mode == "auto" and deep_indices
        else "quick"
        if mode == "auto"
        else None
    )
    _progress(progress_callback, 1, 1, "整理检测报告")
    project_progress.finish()
    return summary


def _replace_xgb_engines(
    engines: list[dict[str, object]],
    replacements: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Replace cached XGBoost rows without rerunning unrelated quick engines."""

    output: list[dict[str, object]] = []
    inserted = False
    for engine in engines:
        if str(engine.get("name") or "").startswith("xgboost_"):
            if not inserted:
                output.extend(dict(item) for item in replacements)
                inserted = True
            continue
        output.append(dict(engine))
    if not inserted:
        output.extend(dict(item) for item in replacements)
    return output


def summarize_project(
    project_name: str,
    results: list[dict[str, object]],
    warnings: list[str] | None = None,
    project_engines: list[dict[str, object]] | None = None,
    project_relationship_graph: dict[str, object] | None = None,
    project_cross_file_analysis: dict[str, object] | None = None,
) -> dict[str, object]:
    category_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for result in results:
        categories = [
            str(category)
            for category in result.get("categories", []) or []
            if str(category).strip()
        ]
        if not categories and str(result.get("final_decision") or "") == "malicious":
            categories = [AI_ONLY_CATEGORY]
        category_counts.update(categories)
        level_counts.update([str(result.get("risk_level", "unknown"))])
        language_counts.update([
            str(
                result.get("display_language")
                or display_language(
                    str(result.get("language") or "unknown"),
                    str(result.get("filename") or ""),
                )
            )
        ])
        decision_counts.update([str(result.get("final_decision", "unknown"))])
    high_risk = sorted(
        results,
        key=lambda item: int(item.get("risk_score", 0)),
        reverse=True,
    )
    avg = round(sum(int(item.get("risk_score", 0)) for item in results) / len(results), 1) if results else 0
    max_score = int(high_risk[0].get("risk_score", 0)) if high_risk else 0
    project_engines = list(project_engines or [])
    project_fusion = fuse_engine_results([
        engine for engine in project_engines
        if isinstance(engine, dict)
    ])
    project_xgb = [
        engine for engine in project_engines
        if str(engine.get("name", "")).startswith("xgboost_project_")
        and engine.get("status") == "completed"
    ]
    has_project_ai_result = bool(project_fusion.get("ai_participated"))
    max_score = max(max_score, int(project_fusion.get("risk_score") or 0))
    final_decision = "malicious" if (
        project_fusion.get("final_decision") == "malicious"
        or decision_counts["malicious"]
    ) else (
        "unknown"
        if (
            decision_counts["unknown"]
            or decision_counts["vulnerable"]
            or (
                has_project_ai_result
                and project_fusion.get("final_decision") == "unknown"
            )
        )
        else "benign"
    )
    if project_fusion.get("final_decision") == "malicious":
        category_counts.update([AI_ONLY_CATEGORY])
    project_decision_counts = Counter(str(engine.get("decision") or "unknown") for engine in project_xgb)
    ai_models_executed = sorted({
        str(name)
        for result in results
        for name in result.get("ai_model_names", []) or []
        if name
    } | {
        str(engine.get("name"))
        for engine in project_engines
        if (
            engine.get("status") == "completed"
            and engine.get("probability") is not None
            and engine.get("name") in {
                "gatv2",
                "xgboost_project_malicious",
            }
        )
    })
    return {
        "project_name": project_name,
        "file_count": len(results),
        "average_score": avg,
        "max_score": max_score,
        "risk_level": (
            "unknown"
            if final_decision == "unknown"
            else _project_level(avg, max_score)
        ),
        "final_decision": final_decision,
        "category_counts": dict(category_counts),
        "level_counts": dict(level_counts),
        "language_counts": dict(language_counts),
        "decision_counts": dict(decision_counts),
        "project_decision_counts": dict(project_decision_counts),
        "decision_policy": "ai_authoritative_rule_explanation_only",
        "ai_models_executed": ai_models_executed,
        # Retain file-level results so a completed task can be reopened and audited.
        "file_results": results,
        "high_risk_files": high_risk,
        "warnings": list(warnings or []),
        "project_engines": project_engines,
        "project_relationship_graph": project_relationship_graph,
        "project_cross_file_analysis": project_cross_file_analysis,
        "project_findings": list(
            (project_cross_file_analysis or {}).get("findings") or []
        ),
        "project_call_graph": (
            (project_cross_file_analysis or {}).get("call_graph")
        ),
        "most_suspicious_component": (
            (project_cross_file_analysis or {}).get(
                "most_suspicious_component"
            )
        ),
    }


def aggregate_project_xgboost(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate validated project/package malicious XGBoost routes over file scores.

    File-level fallback and file-scoped routes are intentionally excluded.
    The release metrics for Python malicious-intent were measured with max
    aggregation, so the runtime uses the same rule.
    """

    grouped: dict[tuple[str, str], list[tuple[dict[str, object], dict[str, object]]]] = {}
    for result in results:
        language = str(result.get("language") or "unknown")
        filename = str(result.get("filename") or "")
        for engine in result.get("engines", []) or []:
            if not isinstance(engine, dict) or not str(engine.get("name", "")).startswith("xgboost_"):
                continue
            if engine.get("status") != "completed" or engine.get("probability") is None:
                continue
            metadata = engine.get("metadata") or {}
            scope = str(metadata.get("evaluation_scope") or "")
            if scope not in {"project", "package", "project_or_package"}:
                continue
            if (
                metadata.get("route_quality_gate_passed") is not True
                or metadata.get("source_heldout_verified") is False
            ):
                continue
            task = str(metadata.get("task") or "")
            if task != "malicious_intent":
                continue
            grouped.setdefault((task, language), []).append((result, engine))

    output: list[dict[str, object]] = []
    positive_labels = {"malicious_intent": "malicious"}
    names = {"malicious_intent": "xgboost_project_malicious"}
    for (task, language), rows in sorted(grouped.items()):
        selected_result, selected_engine = max(
            rows,
            key=lambda item: float(item[1].get("probability") or 0.0),
        )
        probability = max(float(engine.get("probability") or 0.0) for _, engine in rows)
        threshold = float(selected_engine.get("threshold") or 0.5)
        positive = positive_labels[task]
        output.append({
            "name": names[task],
            "status": "completed",
            "decision": positive if probability >= threshold else "benign",
            "probability": round(probability, 4),
            "threshold": threshold,
            "model_version": selected_engine.get("model_version"),
            "duration_ms": 0,
            "metadata": {
                "task": task,
                "language": language,
                "aggregation": "max_file_probability",
                "evaluation_scope": "project_or_package",
                "route_quality_gate_passed": True,
                "source_heldout_verified": (selected_engine.get("metadata") or {}).get(
                    "source_heldout_verified"
                ),
                "source_files": [str(result.get("filename") or "") for result, _ in rows],
                "top_file": str(selected_result.get("filename") or ""),
                "file_count": len(rows),
                "advisory_only": False,
            },
        })
    return output


def _project_level(avg: float, max_score: int) -> str:
    score = max(avg, max_score * 0.8)
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    if score > 0:
        return "low"
    return "safe"


def _safe_extract(
    archive: zipfile.ZipFile,
    target: Path,
    cancel_event: object | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    extracted_path_names: dict[str, str] | None = None,
) -> list[str]:
    warnings = []
    root = target.resolve()
    members = [member for member in archive.infolist() if not member.is_dir()]
    if len(members) > MAX_ZIP_MEMBERS:
        return [f"压缩包包含 {len(members)} 个文件，超过 {MAX_ZIP_MEMBERS} 个成员的限制。"]
    extracted_files = 0
    extracted_size = 0
    component_cache: dict[tuple[tuple[str, ...], str], str] = {}
    used_components: dict[tuple[tuple[str, ...], str], str] = {}
    renamed_components: dict[str, str] = {}
    _progress(progress_callback, 0, len(members), "解压项目文件")
    for completed_count, member in enumerate(members, start=1):
        if _cancelled(cancel_event):
            break
        # Report before processing the current member so even skipped archive
        # entries make the visible ZIP traversal move continuously.
        _progress(
            progress_callback,
            completed_count - 1,
            len(members),
            "解压项目文件",
        )
        name = member.filename.replace("\\", "/")
        original_parts = [
            part
            for part in name.split("/")
            if part not in {"", "."}
        ]
        if (
            name.startswith("/")
            or not original_parts
            or ".." in original_parts
        ):
            _warn(warnings, f"已跳过可疑路径：{member.filename}")
            continue
        if not is_allowed_file(Path(name).name):
            continue
        if member.flag_bits & 0x1:
            _warn(warnings, f"已跳过需要密码的加密文件：{member.filename}")
            continue
        if extracted_files >= MAX_FILES:
            _warn(warnings, f"文件数量超过 {MAX_FILES} 个，剩余源代码文件已跳过。")
            break
        if member.file_size > MAX_FILE_SIZE:
            _warn(warnings, f"已跳过超出大小限制的文件：{member.filename}")
            continue
        if member.compress_size and member.file_size / member.compress_size > MAX_COMPRESSION_RATIO:
            _warn(warnings, f"已跳过压缩比异常的文件：{member.filename}")
            continue
        if extracted_size + member.file_size > MAX_TOTAL_EXTRACTED_SIZE:
            _warn(warnings, f"已解压源代码总大小超过 {MAX_TOTAL_EXTRACTED_SIZE} 字节，剩余文件已跳过。")
            break
        safe_parts = _windows_safe_archive_parts(
            original_parts,
            component_cache,
            used_components,
            renamed_components,
        )
        safe_name = "/".join(safe_parts)
        dest = target.joinpath(*safe_parts).resolve()
        if os.path.commonpath([str(root), str(dest)]) != str(root):
            _warn(warnings, f"已跳过超出项目根目录的路径：{member.filename}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, dest.open("wb") as output:
            remaining = MAX_FILE_SIZE + 1
            while remaining:
                block = source.read(min(64 * 1024, remaining))
                if not block:
                    break
                output.write(block)
                remaining -= len(block)
        if dest.stat().st_size > MAX_FILE_SIZE:
            dest.unlink(missing_ok=True)
            _warn(warnings, f"已跳过超出大小限制的文件：{member.filename}")
            continue
        if is_generic_text_path(name):
            with dest.open("rb") as stream:
                text_probe = stream.read(64 * 1024)
            if not is_probably_text_payload(text_probe):
                dest.unlink(missing_ok=True)
                _warn(warnings, f"已跳过非文本的 TXT/无后缀文件：{member.filename}")
                continue
        extracted_files += 1
        extracted_size += dest.stat().st_size
        if extracted_path_names is not None:
            extracted_path_names[safe_name] = name
    if not _cancelled(cancel_event):
        _progress(
            progress_callback,
            len(members),
            len(members),
            "解压项目文件",
        )
    for original, safe in renamed_components.items():
        _warn(
            warnings,
            "为兼容 Windows，已安全转换压缩包路径片段："
            f"{original} → {safe}",
        )
    return warnings


def _windows_safe_archive_parts(
    original_parts: list[str],
    component_cache: dict[tuple[tuple[str, ...], str], str],
    used_components: dict[tuple[tuple[str, ...], str], str],
    renamed_components: dict[str, str],
) -> list[str]:
    """Map ZIP path components to collision-safe Windows filesystem names."""

    safe_parts: list[str] = []
    parent_key: tuple[str, ...] = ()
    for original in original_parts:
        cache_key = (parent_key, original)
        cached = component_cache.get(cache_key)
        if cached is not None:
            safe_parts.append(cached)
            parent_key = (*parent_key, cached.casefold())
            continue

        candidate = _windows_safe_component(original)
        owner_key = (parent_key, candidate.casefold())
        owner = used_components.get(owner_key)
        collision_index = 0
        while owner is not None and owner != original:
            collision_index += 1
            candidate = _append_component_hash(
                _windows_safe_component(original),
                f"{original}\0{collision_index}",
            )
            owner_key = (parent_key, candidate.casefold())
            owner = used_components.get(owner_key)

        component_cache[cache_key] = candidate
        used_components[owner_key] = original
        if candidate != original:
            renamed_components.setdefault(original, candidate)
        safe_parts.append(candidate)
        parent_key = (*parent_key, candidate.casefold())
    return safe_parts


def _windows_safe_component(original: str) -> str:
    candidate = WINDOWS_INVALID_COMPONENT_PATTERN.sub("_", original)
    candidate = candidate.rstrip(" .")
    if not candidate:
        candidate = "_"
    base_name = candidate.split(".", 1)[0].upper()
    if base_name in WINDOWS_RESERVED_COMPONENT_NAMES:
        candidate = f"_{candidate}"
    if (
        candidate != original
        or len(candidate) > MAX_SAFE_PATH_COMPONENT_LENGTH
    ):
        candidate = _append_component_hash(candidate, original)
    return candidate


def _append_component_hash(candidate: str, identity: str) -> str:
    digest = hashlib.sha256(
        identity.encode("utf-8", errors="surrogatepass"),
    ).hexdigest()[:10]
    marker = f"__{digest}"
    suffix = Path(candidate).suffix
    if (
        suffix
        and len(suffix) <= 20
        and len(marker) + len(suffix) < MAX_SAFE_PATH_COMPONENT_LENGTH
    ):
        stem_budget = (
            MAX_SAFE_PATH_COMPONENT_LENGTH
            - len(marker)
            - len(suffix)
        )
        return f"{candidate[:-len(suffix)][:stem_budget]}{marker}{suffix}"
    return (
        candidate[
            :MAX_SAFE_PATH_COMPONENT_LENGTH - len(marker)
        ]
        + marker
    )


def _warn(warnings: list[str], message: str) -> None:
    if len(warnings) < MAX_WARNINGS:
        warnings.append(message)


def _attach_cross_file_findings(
    results: list[dict[str, object]],
    analysis: dict[str, object],
) -> None:
    """Attach project evidence to its sink file without changing AI authority."""

    by_path = {
        str(item.get("filename") or "").replace("\\", "/").casefold(): item
        for item in results
        if isinstance(item, dict) and item.get("filename")
    }
    for raw in analysis.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        finding = dict(raw)
        path = str(finding.get("file") or "").replace("\\", "/")
        result = by_path.get(path.casefold())
        if result is None:
            continue
        result.setdefault("cross_file_findings", []).append(finding)
        result.setdefault("findings", []).append(finding)
        result.setdefault("evidence_items", []).append(finding)
        category = str(finding.get("category") or "").strip()
        if category:
            categories = result.setdefault("categories", [])
            if category not in categories:
                categories.append(category)
            counts = result.setdefault("category_counts", {})
            counts[category] = int(counts.get(category) or 0) + 1
        type_counts = result.setdefault("risk_type_counts", {})
        risk_type = str(finding.get("risk_type") or "context")
        type_counts[risk_type] = int(type_counts.get(risk_type) or 0) + 1
        domains = result.setdefault("risk_domains", [])
        if "跨文件数据流" not in domains:
            domains.append("跨文件数据流")
        result["project_evidence_note"] = (
            "跨文件证据用于解释项目级结论；不会单独覆盖AI最终判定或风险分。"
        )


def _merge_gat_component_attribution(
    analysis: dict[str, object],
    project_engines: list[dict[str, object]],
) -> None:
    gat = next(
        (
            item for item in project_engines
            if isinstance(item, dict)
            and item.get("name") == "gatv2"
            and item.get("status") == "completed"
        ),
        None,
    )
    static_component = analysis.get("most_suspicious_component")
    if gat is None:
        analysis["component_attribution"] = {
            "gatv2": None,
            "cross_file_static": static_component,
        }
        analysis["most_suspicious_component_basis"] = (
            "cross_file_chain_stage_weighting"
            if static_component else None
        )
        return
    metadata = gat.get("metadata") or {}
    gat_component = metadata.get("most_suspicious_component")
    analysis["component_attribution"] = {
        "gatv2": {
            "method": metadata.get("attribution_method"),
            "most_suspicious_component": gat_component,
            "node_attributions": metadata.get("node_attributions") or [],
            "attributed_file_count": metadata.get("attributed_file_count", 0),
            "total_file_count": metadata.get("total_file_count", 0),
            "coverage_complete": bool(
                metadata.get("attribution_coverage_complete")
            ),
            "model_probability": gat.get("probability"),
            "model_threshold": gat.get("threshold"),
            "model_decision": gat.get("decision"),
        },
        "cross_file_static": static_component,
    }
    if isinstance(gat_component, dict):
        analysis["most_suspicious_component"] = dict(gat_component)
        analysis["most_suspicious_component_basis"] = (
            "gatv2_leave_one_file_component_out"
        )
    else:
        analysis["most_suspicious_component_basis"] = (
            "cross_file_chain_stage_weighting"
            if static_component else None
        )


def _build_project_relationship_view(
    graph_samples: list[object],
    results: list[dict[str, object]],
    warnings: list[str],
) -> dict[str, object] | None:
    try:
        return build_project_relationship_graph(
            graph_samples,
            {
                str(item.get("filename") or ""): item
                for item in results
                if isinstance(item, dict) and item.get("filename")
            },
        )
    except Exception as exc:
        _warn(
            warnings,
            f"项目文件调用关系图生成失败，本次模型检测结果不受影响：{exc}",
        )
        return None


def _bounded_source_content(payload: bytes, maximum_bytes: int) -> str:
    if len(payload) <= maximum_bytes:
        return decode_source_payload(payload)
    marker = b"\n/* ... project quick-scan middle omitted ... */\n"
    remaining = max(2, maximum_bytes - len(marker))
    head_size = remaining // 2
    tail_size = remaining - head_size
    sampled = payload[:head_size] + marker + payload[-tail_size:]
    return decode_source_payload(sampled)


def _select_deep_candidates(
    records: list[dict[str, object]], results: list[dict[str, object]], limit: int,
    supported_languages: set[str] | None = None,
) -> list[int]:
    supported_languages = supported_languages or _deep_languages()
    supported = [index for index, record in enumerate(records) if record.get("language") in supported_languages]
    if len(supported) <= limit:
        return supported
    entry_tokens = ("main", "app", "server", "controller", "service", "route", "view", "auth", "security")
    ranked = sorted(
        supported,
        key=lambda index: (
            _ai_candidate_priority(results[index]),
            int(any(token in str(records[index]["filename"]).lower() for token in entry_tokens)),
            min(len(str(records[index]["content"])), MAX_FILE_SIZE),
            str(records[index]["filename"]),
        ),
        reverse=True,
    )
    high_risk_count = min(limit * 2 // 3, len(ranked))
    selected = ranked[:high_risk_count]
    selected_set = set(selected)
    remaining = [index for index in sorted(supported, key=lambda value: str(records[value]["filename"])) if index not in selected_set]
    slots = limit - len(selected)
    if slots and remaining:
        for slot in range(slots):
            position = min(len(remaining) - 1, int(slot * len(remaining) / slots))
            candidate = remaining[position]
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
    return selected[:limit]


def _ai_candidate_priority(result: dict[str, object]) -> float:
    """Rank semantic-review candidates by AI output, not rule severity."""

    priorities = []
    for engine in result.get("engines", []) or []:
        if (
            not isinstance(engine, dict)
            or engine.get("name") != "xgboost_malicious"
            or engine.get("status") != "completed"
            or engine.get("probability") is None
        ):
            continue
        probability = float(engine.get("probability") or 0.0)
        threshold = max(
            0.0001,
            min(0.9999, float(engine.get("threshold") or 0.5)),
        )
        metadata = engine.get("metadata") or {}
        raw_decision = str(
            metadata.get("raw_model_decision")
            or engine.get("decision")
            or ""
        )
        if raw_decision == "malicious":
            priorities.append(
                100.0
                + ((probability - threshold) / (1.0 - threshold)) * 100.0
            )
        else:
            priorities.append(
                (probability / threshold) * 100.0
            )
    return max(priorities, default=0.0)


def _deep_languages() -> set[str]:
    """Read the independently validated semantic-model language set."""

    try:
        registry = json.loads((MODEL_DIR / "codet5p_registry.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {}
    active_routes = (registry.get("active_routes") or {}).get("malicious_intent") or {}
    return (
        {str(language) for language in active_routes}
        if isinstance(active_routes, dict)
        else set()
    )


def _cancelled(cancel_event: object | None) -> bool:
    return bool(cancel_event and getattr(cancel_event, "is_set", lambda: False)())


def _progress(callback: Callable[[int, int, str], None] | None, done: int, total: int, stage: str) -> None:
    if callback:
        callback(done, total, stage)


@contextmanager
def _windows_high_core_affinity(core_count: int):
    """Keep the CPU-heavy explanation thread off resident model cores."""

    if os.name != "nt":
        yield
        return
    cpu_count = os.cpu_count() or 1
    selected = max(1, min(core_count, cpu_count))
    mask = ((1 << selected) - 1) << (cpu_count - selected)
    previous = 0
    try:
        import ctypes

        thread_handle = ctypes.windll.kernel32.GetCurrentThread()
        previous = ctypes.windll.kernel32.SetThreadAffinityMask(
            thread_handle,
            mask,
        )
        yield
    finally:
        if previous:
            ctypes.windll.kernel32.SetThreadAffinityMask(
                thread_handle,
                previous,
            )
