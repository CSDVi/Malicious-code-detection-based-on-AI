from io import BytesIO
from pathlib import Path
import shutil
import threading
from uuid import uuid4
from zipfile import ZipFile

import pytest

import attack_detection.project_scanner as project_scanner
from attack_detection.languages import display_language
from app import create_app, display_zh


@pytest.fixture
def local_tmp_path():
    path = Path(__file__).resolve().parents[1] / ".test-work" / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_default_request_limit_accepts_one_gib_archive_with_multipart_overhead(monkeypatch):
    monkeypatch.delenv("XIEZHI_MAX_UPLOAD_BYTES", raising=False)
    app = create_app()
    assert project_scanner.MAX_FILE_SIZE == 100 * 1024 * 1024
    assert project_scanner.MAX_ARCHIVE_SIZE == 1024 * 1024 * 1024
    assert project_scanner.MAX_TOTAL_EXTRACTED_SIZE == project_scanner.MAX_ARCHIVE_SIZE
    assert app.config["MAX_CONTENT_LENGTH"] > project_scanner.MAX_ARCHIVE_SIZE


def test_login_redirect_does_not_render_the_default_english_flash_box():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/attack/", follow_redirects=True)

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Please log in to access this page." not in page
    assert "flash-error" not in page


def test_project_cancel_during_preparation_returns_cancelled_instead_of_failing(
    local_tmp_path, monkeypatch,
):
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("src/app.py", "print('ok')")
    archive_data.seek(0)
    cancel_event = threading.Event()
    cancel_event.set()
    monkeypatch.setattr(project_scanner, "SCAN_TEMP_ROOT", local_tmp_path)

    result = project_scanner.scan_zip_project(
        archive_data,
        original_filename="project.zip",
        mode="deep",
        cancel_event=cancel_event,
    )

    assert result == {"cancelled": True}


def test_project_large_source_sampling_keeps_head_and_tail():
    payload = b"HEAD-" + (b"x" * 200_000) + b"-TAIL"

    content = project_scanner._bounded_source_content(payload, 64 * 1024)

    assert content.startswith("HEAD-")
    assert content.endswith("-TAIL")
    assert len(content.encode("utf-8")) <= 64 * 1024


def test_user_visible_internal_values_have_chinese_labels():
    assert display_zh("quick") == "快速"
    assert display_zh("batch") == "批处理/CMD"
    assert display_zh("xgboost_project_malicious") == "XGBoost 项目恶意代码模型"
    assert display_zh("json") == "JSON"
    assert display_zh("yaml") == "YAML"
    assert display_zh("text") == "TXT"
    assert display_language("config", "package.json") == "json"
    assert display_language("config", "compose.yml") == "yaml"
    assert display_language("config", "settings.yaml") == "yaml"
    assert display_language("unknown", "notes.txt") == "text"
    assert display_zh("no language route passed Precision/FPR/FNR release gate") == (
        "没有语言路由通过精确率、误报率和漏报率发布门禁"
    )


def test_staging_rejects_oversized_archive_and_removes_partial_file(local_tmp_path, monkeypatch):
    monkeypatch.setattr(project_scanner, "SCAN_TEMP_ROOT", local_tmp_path)
    monkeypatch.setattr(project_scanner, "MAX_ARCHIVE_SIZE", 10)
    with pytest.raises(project_scanner.ArchiveTooLargeError):
        project_scanner.stage_project_archive(BytesIO(b"x" * 11))
    assert list(local_tmp_path.iterdir()) == []


def test_safe_extract_only_writes_bounded_source_files(local_tmp_path):
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("src/main.py", "print('ok')")
        archive.writestr("assets/logo.bin", b"binary")
        archive.writestr("../escape.py", "print('escape')")
    archive_data.seek(0)
    target = local_tmp_path / "source"
    target.mkdir()
    with ZipFile(archive_data) as archive:
        warnings = project_scanner._safe_extract(archive, target)
    assert (target / "src" / "main.py").read_text(encoding="utf-8") == "print('ok')"
    assert not (target / "assets" / "logo.bin").exists()
    assert not (local_tmp_path / "escape.py").exists()
    assert any("可疑路径" in warning for warning in warnings)


def test_safe_extract_converts_invalid_windows_components_without_losing_paths(
    local_tmp_path,
):
    archive_data = BytesIO()
    original_names = {
        "Dropper:AndroidOS.AndroidTrojanStarter/src/CON.java",
        "Dropper?AndroidOS.AndroidTrojanStarter/src/payload.java",
    }
    with ZipFile(archive_data, "w") as archive:
        for original_name in original_names:
            archive.writestr(original_name, "class Payload {}")
    archive_data.seek(0)
    target = local_tmp_path / "windows-compatible"
    target.mkdir()
    extracted_path_names = {}

    with ZipFile(archive_data) as archive:
        warnings = project_scanner._safe_extract(
            archive,
            target,
            extracted_path_names=extracted_path_names,
        )

    assert set(extracted_path_names.values()) == original_names
    assert len(extracted_path_names) == 2
    for safe_name in extracted_path_names:
        assert (target / safe_name).is_file()
        assert not any(
            character in safe_name
            for character in '<>:"\\|?*'
        )
    assert any("兼容 Windows" in warning for warning in warnings)
    assert any("Dropper:AndroidOS" in warning for warning in warnings)
    assert any("CON.java" in warning for warning in warnings)


def test_safe_extract_accepts_file_at_limit_and_skips_only_above_it(local_tmp_path, monkeypatch):
    monkeypatch.setattr(project_scanner, "MAX_FILE_SIZE", 10)
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("src/at-limit.py", b"x" * 10)
        archive.writestr("src/over-limit.py", b"x" * 11)
    archive_data.seek(0)
    target = local_tmp_path / "boundary"
    target.mkdir()

    with ZipFile(archive_data) as archive:
        warnings = project_scanner._safe_extract(archive, target)

    assert (target / "src" / "at-limit.py").read_bytes() == b"x" * 10
    assert not (target / "src" / "over-limit.py").exists()
    assert warnings == ["已跳过超出大小限制的文件：src/over-limit.py"]


def test_project_upload_template_reports_one_gib_limit():
    template = Path(__file__).resolve().parents[2] / "frontend" / "templates" / "attack" / "index.html"
    content = template.read_text(encoding="utf-8")
    assert "当前仅支持 ZIP（.zip）压缩包，最大 1 GB" in content
    assert 'data-max-bytes="1073741824"' in content
    assert 'name="project_zip" accept=".zip"' in content
    assert 'name="line_explanations"' not in content
    assert ".rar" not in content.lower()
    assert ".7z" not in content.lower()


def test_xgboost_prefilter_requests_fallback_only_when_ai_is_unreliable():
    def engine(
        decision="benign",
        probability=0.1,
        status="completed",
        **metadata,
    ):
        return {
            "name": "xgboost_malicious",
            "status": status,
            "decision": decision,
            "probability": (
                probability
                if status == "completed"
                else None
            ),
            "threshold": 0.8,
            "metadata": {
                "task": "malicious_intent",
                "raw_model_decision": decision,
                "advisory_only": False,
                **metadata,
            },
        }

    assert project_scanner._xgb_requires_full_evidence([
        engine(),
    ]) is False
    assert project_scanner._xgb_requires_full_evidence([
        engine("malicious", 0.9),
    ]) is False
    assert project_scanner._xgb_evidence_state([
        engine(),
    ]) == "benign"
    assert project_scanner._xgb_evidence_state([
        engine("malicious", 0.9),
    ]) == "malicious"
    assert project_scanner._xgb_requires_full_evidence([
        engine(
            probability=0.5,
            uncertain_low=0.45,
            uncertain_high=0.55,
        ),
    ]) is True
    assert project_scanner._xgb_requires_full_evidence([
        engine(status="unavailable"),
    ]) is True
    assert project_scanner._xgb_requires_full_evidence([
        engine(advisory_only=True),
    ]) is True
    assert project_scanner._xgb_requires_full_evidence([
        engine(),
        engine("malicious", 0.9),
    ]) is True


def test_quick_project_uses_xgboost_only_even_when_ai_is_uncertain(
    local_tmp_path,
    monkeypatch,
):
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("safe.py", "SAFE")
        archive.writestr("bad.py", "BAD")
        archive.writestr("uncertain.py", "UNCERTAIN")
        archive.writestr("unavailable.py", "UNAVAILABLE")
    archive_data.seek(0)
    evidence_contents = []
    quick_engine_names = {}
    progress_updates = []

    def xgb_result(content, _language, **_kwargs):
        metadata = {
            "task": "malicious_intent",
            "advisory_only": False,
        }
        if content == "UNAVAILABLE":
            return [{
                "name": "xgboost_malicious",
                "status": "unavailable",
                "probability": None,
                "metadata": metadata,
            }]
        if content == "UNCERTAIN":
            metadata.update({
                "raw_model_decision": "benign",
                "uncertain_low": 0.45,
                "uncertain_high": 0.55,
            })
            probability = 0.5
            decision = "benign"
        elif content == "BAD":
            metadata["raw_model_decision"] = "malicious"
            probability = 0.9
            decision = "malicious"
        else:
            metadata["raw_model_decision"] = "benign"
            probability = 0.1
            decision = "benign"
        return [{
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": decision,
            "probability": probability,
            "threshold": 0.8,
            "metadata": metadata,
        }]

    def evidence_worker(request):
        content, _language, _payload = request
        evidence_contents.append(content)
        return [
            {
                "name": "rule_engine",
                "status": "completed",
                "findings": [],
            },
            {
                "name": "static_evidence",
                "status": "completed",
                "findings": [],
            },
        ]

    def fake_scan_file(
        filename,
        _payload,
        precomputed_quick_result=None,
        **_kwargs,
    ):
        engines = list(
            (precomputed_quick_result or {}).get(
                "engines",
                [],
            )
        )
        quick_engine_names[filename] = [
            engine["name"] for engine in engines
        ]
        return {
            "filename": filename,
            "language": "python",
            "engines": engines,
            "risk_score": 0,
            "risk_level": "safe",
            "final_decision": "benign",
            "categories": [],
            "ai_participated": True,
            "decision_authority": "ai",
        }

    monkeypatch.setattr(
        project_scanner,
        "SCAN_TEMP_ROOT",
        local_tmp_path,
    )
    monkeypatch.setattr(
        project_scanner,
        "prepare_xgb_batch",
        lambda requests, cancel_event=None: [
            {} for _request in requests
        ],
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_xgb_prepared",
        xgb_result,
    )
    monkeypatch.setattr(
        project_scanner,
        "_scan_quick_evidence_worker",
        evidence_worker,
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_file",
        fake_scan_file,
    )

    result = project_scanner.scan_zip_project(
        archive_data,
        original_filename="project.zip",
        mode="quick",
        progress_callback=lambda done, total, stage: (
            progress_updates.append((done, total, stage))
        ),
    )

    assert evidence_contents == []
    for filename in quick_engine_names:
        assert quick_engine_names[filename] == [
            "xgboost_malicious",
        ]
    assert result["rule_static_analyzed_file_count"] == 0
    assert result["rule_static_skipped_file_count"] == 4
    assert result["evidence_strategy"] == "xgboost_only"
    assert (
        result[
            "ai_confident_benign_evidence_skipped_count"
        ]
        == 1
    )
    assert all(total == 100 for _done, total, _stage in progress_updates)
    assert [
        done for done, _total, _stage in progress_updates
    ] == sorted(
        done for done, _total, _stage in progress_updates
    )
    assert progress_updates[-1] == (
        100,
        100,
        "检测结果已整理",
    )
    assert (
        result[
            "ai_decisive_malicious_rule_static_skipped_count"
        ]
        == 1
    )


def test_auto_project_escalates_only_uncertain_files_and_skips_project_graph(
    local_tmp_path,
    monkeypatch,
):
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("safe.py", "SAFE")
        archive.writestr("bad.py", "BAD")
        archive.writestr("uncertain.py", "UNCERTAIN")
    archive_data.seek(0)
    semantic_contents = []
    attribution_contents = []

    def xgb_result(content, _language, **_kwargs):
        uncertain = content == "UNCERTAIN"
        malicious = content == "BAD"
        decision = "malicious" if malicious else "benign"
        probability = (
            0.9 if malicious else 0.5 if uncertain else 0.1
        )
        return [{
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": decision,
            "probability": probability,
            "threshold": 0.8,
            "metadata": {
                "task": "malicious_intent",
                "raw_model_decision": decision,
                "advisory_only": False,
                "uncertain_low": 0.45 if uncertain else None,
                "uncertain_high": 0.55 if uncertain else None,
            },
        }]

    def fake_scan_file(
        filename,
        payload,
        precomputed_quick_result=None,
        precomputed_semantic=None,
        **_kwargs,
    ):
        engines = list(
            (precomputed_quick_result or {}).get("engines", [])
        )
        if precomputed_semantic is not None:
            engines.append(precomputed_semantic)
        return {
            "filename": filename,
            "language": "python",
            "engines": engines,
            "risk_score": 90 if b"BAD" in payload else 10,
            "risk_level": "high" if b"BAD" in payload else "safe",
            "final_decision": (
                "malicious" if b"BAD" in payload else "benign"
            ),
            "categories": [],
            "ai_participated": True,
            "decision_authority": "ai",
        }

    def fake_semantic_batch(_self, requests, cancel_event=None):
        semantic_contents.extend(
            request["content"] for request in requests
        )
        return [{
            "name": "codet5p",
            "status": "completed",
            "decision": "benign",
            "probability": 0.1,
            "threshold": 0.8,
        } for _request in requests]

    def fake_attribution_batch(requests, **_kwargs):
        attribution_contents.extend(
            request["content"] for request in requests
        )
        return [
            xgb_result(request["content"], request["language"])
            for request in requests
        ]

    monkeypatch.setattr(
        project_scanner,
        "SCAN_TEMP_ROOT",
        local_tmp_path,
    )
    monkeypatch.setattr(
        project_scanner,
        "prepare_xgb_batch",
        lambda requests, cancel_event=None: [
            {} for _request in requests
        ],
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_xgb_prepared",
        xgb_result,
    )
    monkeypatch.setattr(
        project_scanner,
        "_deep_languages",
        lambda: {"python"},
    )
    monkeypatch.setattr(
        project_scanner.CodeT5PEngine,
        "scan_batch",
        fake_semantic_batch,
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_xgb_attribution_batch",
        fake_attribution_batch,
    )
    monkeypatch.setattr(
        project_scanner,
        "build_project_graph",
        lambda _samples: pytest.fail(
            "auto mode must not build the project graph",
        ),
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_file",
        fake_scan_file,
    )

    result = project_scanner.scan_zip_project(
        archive_data,
        original_filename="project.zip",
        mode="auto",
    )

    assert semantic_contents == ["UNCERTAIN"]
    assert attribution_contents == ["UNCERTAIN"]
    assert result["deep_scanned_file_count"] == 1
    assert result["automatic_effective_mode"] == "standard"
    assert not any(
        engine.get("name") == "gatv2"
        for engine in result["project_engines"]
    )


def test_standard_project_uses_codet5p_without_hidden_model_fallback(local_tmp_path, monkeypatch):
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("src/app.js", "eval(payload)")
        archive.writestr("src/main.py", "print('ok')")
    archive_data.seek(0)

    codet5p_requests = []
    scan_requests = []

    def semantic_result(name, probability):
        return {
            "name": name,
            "status": "completed",
            "decision": "malicious" if probability >= 0.8 else "benign",
            "probability": probability,
            "threshold": 0.8,
            "model_version": f"{name}-test",
            "metadata": {
                "primary_task": "malicious_intent",
                "task_probabilities": {"malicious_intent": probability},
                "task_thresholds": {"malicious_intent": 0.8},
                "task_versions": {"malicious_intent": f"{name}-test"},
            },
        }

    def fake_codet5p_batch(requests):
        codet5p_requests.extend(requests)
        return [
            semantic_result("codet5p", 0.91),
            {"name": "codet5p", "status": "unavailable", "reason": "no python route"},
        ]

    def fake_scan_file(
        filename, payload, mode="auto", precomputed_semantic=None,
        precomputed_quick_result=None, cancel_event=None,
        generate_line_attributions=True, analysis_max_bytes=None,
        run_legacy_baseline=True,
    ):
        scan_requests.append({
            "filename": filename,
            "mode": mode,
            "generate_line_attributions": generate_line_attributions,
            "has_semantic": precomputed_semantic is not None,
            "run_legacy_baseline": run_legacy_baseline,
        })
        language = "javascript" if filename.endswith(".js") else "python"
        return {
            "filename": filename,
            "language": language,
            "engines": [precomputed_semantic] if precomputed_semantic else [],
            "risk_score": 0,
            "risk_level": "safe",
            "final_decision": "benign",
            "categories": [],
        }

    monkeypatch.setattr(project_scanner, "SCAN_TEMP_ROOT", local_tmp_path)
    monkeypatch.setattr(
        project_scanner,
        "CODET5_PROJECT_FILE_LIMIT",
        12,
    )
    monkeypatch.setattr(project_scanner, "_deep_languages", lambda: {"javascript", "python"})
    monkeypatch.setattr(
        project_scanner.CodeT5PEngine,
        "scan_batch",
        lambda _self, requests, cancel_event=None: fake_codet5p_batch(requests),
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_xgb_attribution_batch",
        lambda requests, **_kwargs: [[] for _request in requests],
    )
    monkeypatch.setattr(project_scanner, "scan_file", fake_scan_file)

    result = project_scanner.scan_zip_project(
        archive_data,
        original_filename="project.zip",
        mode="standard",
    )

    assert [request["language"] for request in codet5p_requests] == ["javascript", "python"]
    initial_scans = [request for request in scan_requests if request["mode"] == "quick"]
    assert [request["generate_line_attributions"] for request in initial_scans] == [
        False, False,
    ]
    assert all(
        request["run_legacy_baseline"] is False
        for request in initial_scans
    )
    semantic_engines = {
        item["filename"]: (
            item["engines"][0]["name"],
            item["engines"][0]["status"],
        )
        for item in result["file_results"]
    }
    assert semantic_engines == {
        "src/app.js": ("codet5p", "completed"),
        "src/main.py": ("codet5p", "unavailable"),
    }


def test_standard_project_can_skip_xgboost_line_attributions(
    local_tmp_path,
    monkeypatch,
):
    archive_data = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("src/app.js", "eval(payload)")
    archive_data.seek(0)
    scan_requests = []

    def fake_scan_file(
        filename,
        _payload,
        mode="auto",
        precomputed_semantic=None,
        **kwargs,
    ):
        scan_requests.append({
            "mode": mode,
            "generate_line_attributions": kwargs.get(
                "generate_line_attributions",
            ),
        })
        return {
            "filename": filename,
            "language": "javascript",
            "engines": (
                [precomputed_semantic]
                if precomputed_semantic is not None
                else []
            ),
            "risk_score": 80,
            "risk_level": "high",
            "final_decision": "malicious",
            "categories": [],
        }

    monkeypatch.setattr(project_scanner, "SCAN_TEMP_ROOT", local_tmp_path)
    monkeypatch.setattr(
        project_scanner,
        "_deep_languages",
        lambda: {"javascript"},
    )
    monkeypatch.setattr(
        project_scanner.CodeT5PEngine,
        "scan_batch",
        lambda _self, requests, cancel_event=None: [
            {
                "name": "codet5p",
                "status": "completed",
                "decision": "malicious",
                "probability": 0.9,
                "threshold": 0.8,
            }
            for _request in requests
        ],
    )
    monkeypatch.setattr(
        project_scanner,
        "scan_xgb_attribution_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "line attribution must remain disabled",
        ),
    )
    monkeypatch.setattr(project_scanner, "scan_file", fake_scan_file)

    result = project_scanner.scan_zip_project(
        archive_data,
        original_filename="project.zip",
        mode="standard",
        generate_line_attributions=False,
    )

    assert result["line_explanations_enabled"] is False
    final_scan = next(
        request
        for request in scan_requests
        if request["mode"] == "standard"
    )
    assert final_scan["generate_line_attributions"] is False
