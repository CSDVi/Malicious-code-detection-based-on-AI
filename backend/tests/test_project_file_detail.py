from pathlib import Path

from attack_detection.remediation import remediation_for_finding
from web.routes.attack_routes import (
    _project_evidence_sort_key,
    _project_file_detail_result,
    _project_result_view,
)


def test_project_file_detail_template_stays_compact():
    template = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "templates"
        / "attack"
        / "project_file_detail.html"
    ).read_text(encoding="utf-8")

    assert "文件风险点分析" not in template
    assert "所属项目：" not in template
    assert "模型概率针对整个文件" not in template
    assert "具体依据以每条卡片中的说明为准" not in template
    assert "<strong>风险说明</strong>" not in template
    assert "指数依据：" not in template
    assert 'class="metric-sub"' not in template


def test_base_template_uses_only_local_frontend_dependencies():
    template = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "templates"
        / "base.html"
    ).read_text(encoding="utf-8")
    app_js = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "static"
        / "js"
        / "app.js"
    ).read_text(encoding="utf-8")
    dashboard_template = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "templates"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "cdn.jsdelivr.net" not in template
    assert "cdn.jsdelivr.net" not in dashboard_template
    assert "chart.umd" not in template
    assert "gsap.min.js" not in template
    assert "vendor/bootstrap-icons/bootstrap-icons.min.css" in template
    assert "vendor/three/three.module.min.js" in dashboard_template
    assert "vendor/three/addons/" in dashboard_template
    assert "models/xiezhi-particles-v1.bin" in dashboard_template
    assert "xiezhi_active_scan_jobs" in app_js
    assert "正在上传项目并创建检测任务" in app_js


def test_remediation_is_short_and_specific_to_language_and_category():
    command = remediation_for_finding(
        {"category": "Command Execution"},
        "python",
    )["suggestions"]
    sql = remediation_for_finding(
        {"category": "SQL Injection"},
        "python",
    )["suggestions"]

    assert 1 <= len(command) <= 2
    assert 1 <= len(sql) <= 2
    assert command != sql
    assert "subprocess" in command[0]
    assert "DB-API" in sql[0]


def test_project_file_detail_only_allows_standard_or_deep_results():
    project_result = {
        "high_risk_files": [
            {"filename": "quick.py", "effective_mode": "quick"},
            {"filename": "standard.py", "effective_mode": "standard"},
            {"filename": "deep.py", "effective_mode": "deep"},
        ]
    }

    assert _project_file_detail_result(project_result, 0) is None
    assert _project_file_detail_result(project_result, 1)["filename"] == "standard.py"
    assert _project_file_detail_result(project_result, 2)["filename"] == "deep.py"
    assert _project_file_detail_result(project_result, -1) is None
    assert _project_file_detail_result(project_result, 3) is None


def test_project_evidence_is_sorted_by_score_then_line():
    evidence = [
        {"line": 20, "suspicion_score": 40},
        {"line": 12, "suspicion_score": 80},
        {"line": 5, "suspicion_score": 80},
        {"line": "未知", "suspicion_score": "未评分"},
    ]

    ordered = sorted(evidence, key=_project_evidence_sort_key)

    assert [item["line"] for item in ordered] == [5, 12, 20, "未知"]


def test_project_result_view_numbers_files_and_builds_project_only_extensions():
    view = _project_result_view({
        "high_risk_files": [
            {"filename": "demo/src/app.js", "effective_mode": "deep"},
            {"filename": "demo/run.py", "effective_mode": "quick"},
        ],
        "deep_scanned_file_count": 1,
        "quick_only_file_count": 1,
        "warnings": [
            "已跳过超出大小限制的文件：demo/package-lock.json",
            "为控制扫描时长，CodeT5+ 220M 实际复核 1 个候选文件；其余 1 个受支持语言文件保留快速模式结果。",
        ],
    })

    assert view is not None
    assert [
        (item["project_serial"], item["filename"])
        for item in view["display_files"]
    ] == [
        (1, "demo/package-lock.json"),
        (2, "demo/src/app.js"),
        (3, "demo/run.py"),
    ]
    assert view["skipped_files"] == [{
        "project_serial": 1,
        "filename": "demo/package-lock.json",
        "reason": "超过单文件大小限制",
        "file_extension": ".json",
    }]
    assert view["other_warnings"] == []
    assert view["warning_count"] == 1
    assert view["quick_only_file_count"] == 1
    assert {
        item["label"]: item["count"]
        for item in view["file_extensions"]
    } == {".js": 1, ".json": 1, ".py": 1}
