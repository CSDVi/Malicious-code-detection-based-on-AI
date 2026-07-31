import json
import hashlib
from pathlib import Path

from attack_detection.fusion import fuse_engine_results
from attack_detection.explainability import merge_model_line_attributions
from attack_detection.engines.xgb_engine import (
    XGBoostEngine,
    _line_attributions,
)
from attack_detection.features.graph_builder import build_lightweight_graph
from attack_detection.project_scanner import aggregate_project_xgboost, summarize_project
from attack_detection.model_center import TASK_LABELS, _task_rows, model_center_view
from attack_detection.model_registry import runtime_status
from attack_detection.languages import (
    BINARY_EXTENSIONS,
    SOURCE_EXTENSIONS,
    display_language,
)
from attack_detection.orchestrator import DetectionOrchestrator
from attack_detection.owasp_coverage import OWASP_TOP10_2025, coverage_summary
from attack_detection.remediation import catalog_statistics, load_remediation_catalog
from attack_detection.risk_taxonomy import taxonomy_for_category
from attack_detection.scanner import detect_language, scan_code, scan_file
from attack_detection.trainer import meets_quality_gate
from attack_detection.training.byte_tcn_trainer import _eligible_languages
from attack_detection.training.byte_tcn_infer import _task_settings
from attack_detection.training.codet5p_infer import _group_requests_by_route
from attack_detection.static_analysis.strings_ioc import EMAIL_RE, _email_matches
from attack_detection.training import xgb_trainer
from web.routes.attack_routes import (
    _auxiliary_analysis_view,
    _public_record_view,
    _single_file_upload_contract,
    _training_model_options,
    _visible_runtime_models,
)
from web.routes.main_routes import _merge_recent_detection_records


BACKEND_DIR = Path(__file__).resolve().parents[1]


def _scan_with_rule_engine(
    filename: str,
    content: str,
) -> dict[str, object]:
    return scan_file(
        filename,
        content.encode("utf-8"),
        mode="standard",
        precomputed_semantic={
            "name": "codet5p",
            "status": "unavailable",
            "reason": "rule-engine test",
        },
        generate_line_attributions=False,
        run_legacy_baseline=False,
    )


def test_email_ioc_fast_path_preserves_whole_text_regex_matches():
    content = (
        "<?php @include($path); $owner = 'admin@example.com'; ?>\r\n"
        "@unlink($temp); support+security@sub.example.org\n"
        "not-an-email @ suppressed_call();"
    )
    expected = [
        (match.start(), match.group(0))
        for match in EMAIL_RE.finditer(content)
    ]

    assert list(_email_matches(content)) == expected


def test_project_xgboost_batch_matches_individual_route_predictions():
    requests = [
        {
            "language": "php",
            "content": "<?php eval($_POST['cmd']); ?>",
        },
        {
            "language": "php",
            "content": "<?php echo htmlspecialchars($name); ?>",
        },
    ]
    engine = XGBoostEngine()
    prepared = engine.prepare_batch(requests)
    batch_outputs = [
        engine.scan(
            request["content"],
            request["language"],
            generate_line_attributions=False,
            precomputed_batch=batch,
        )
        for request, batch in zip(requests, prepared)
    ]
    individual_outputs = [
        engine.scan(
            request["content"],
            request["language"],
            generate_line_attributions=False,
        )
        for request in requests
    ]

    assert [
        output[0]["probability"]
        for output in batch_outputs
    ] == [
        output[0]["probability"]
        for output in individual_outputs
    ]
    assert [
        output[0]["decision"]
        for output in batch_outputs
    ] == [
        output[0]["decision"]
        for output in individual_outputs
    ]


def test_dashboard_merges_single_file_and_completed_project_records_by_time():
    recent = _merge_recent_detection_records(
        [{
            "id": 7,
            "filename": "older.py",
            "risk_level": "safe",
            "risk_score": 3,
            "effective_mode": "quick",
            "created_at": "2026-07-29T10:00:00",
        }],
        [
            {
                "id": "completed-project",
                "target_name": "newer.zip",
                "status": "completed",
                "mode": "standard",
                "created_at": "2026-07-29T09:00:00",
                "finished_at": "2026-07-29T11:00:00",
                "result": {"risk_level": "high", "max_score": 88},
            },
            {
                "id": "cancelled-project",
                "target_name": "cancelled.zip",
                "status": "cancelled",
                "mode": "standard",
                "created_at": "2026-07-29T12:00:00",
                "result": None,
            },
        ],
        limit=4,
    )

    assert [item["record_type"] for item in recent] == ["project", "single"]
    assert recent[0]["target_name"] == "newer.zip"
    assert recent[0]["risk_level"] == "high"
    assert recent[0]["risk_score"] == 88
    assert all(item["id"] != "cancelled-project" for item in recent)


def test_deep_rescan_reuses_quick_engine_results(monkeypatch):
    orchestrator = DetectionOrchestrator()
    content = "def add(left, right):\n    return left + right\n"
    quick = orchestrator.scan("sample.py", content, "python", selected_mode="quick")

    def fail_if_repeated(*_args, **_kwargs):
        raise AssertionError("quick engine was executed twice")

    monkeypatch.setattr(orchestrator.rule_engine, "scan", fail_if_repeated)
    monkeypatch.setattr(orchestrator.xgb_engine, "scan", fail_if_repeated)
    monkeypatch.setattr(orchestrator.static_engine, "scan", fail_if_repeated)
    monkeypatch.setattr(orchestrator.reputation_engine, "scan", fail_if_repeated)
    monkeypatch.setattr(orchestrator.sandbox_engine, "scan", fail_if_repeated)

    semantic = {
        "name": "codet5p",
        "status": "completed",
        "decision": "benign",
        "probability": 0.05,
        "threshold": 0.5,
        "model_version": "codet5p-test",
        "duration_ms": 1,
        "metadata": {
            "primary_task": "malicious_intent",
            "task_probabilities": {"malicious_intent": 0.05},
            "task_thresholds": {"malicious_intent": 0.5},
            "task_versions": {"malicious_intent": "codet5p-test"},
        },
    }
    result = orchestrator.scan(
        "sample.py",
        content,
        "python",
        selected_mode="standard",
        precomputed_semantic=semantic,
        precomputed_quick_result=quick,
    )

    assert result["effective_mode"] == "standard"
    assert sum(engine["name"] == "codet5p" for engine in result["engines"]) == 1
    assert result["hashes"] == quick["hashes"]


def test_line_attribution_follows_explicit_detection_mode(monkeypatch):
    orchestrator = DetectionOrchestrator()
    attribution_flags = []

    def fake_xgb_scan(_content, _language, **kwargs):
        attribution_flags.append(kwargs["generate_line_attributions"])
        return []

    monkeypatch.setattr(orchestrator.xgb_engine, "scan", fake_xgb_scan)
    monkeypatch.setattr(
        orchestrator.gat_engine,
        "scan",
        lambda _content, _language: {
            "name": "gatv2", "status": "unavailable", "reason": "test",
        },
    )
    semantic = {
        "name": "codet5p",
        "status": "unavailable",
        "reason": "test",
    }

    orchestrator.scan("quick.py", "print('ok')", "python", selected_mode="quick")
    orchestrator.scan(
        "standard.py",
        "print('ok')",
        "python",
        selected_mode="standard",
        precomputed_semantic=semantic,
    )
    orchestrator.scan(
        "deep.py",
        "print('ok')",
        "python",
        selected_mode="deep",
        precomputed_semantic=semantic,
    )

    assert attribution_flags == [False, True, True]


def test_repetitive_rule_findings_are_bounded_in_public_result():
    content = "\n".join(
        f"$x{index} = base64_decode($_POST['x']); eval($x{index});"
        for index in range(150)
    )

    result = _scan_with_rule_engine("repetitive.php", content)
    rule_engine = next(
        engine for engine in result["engines"]
        if engine["name"] == "rule_engine"
    )

    assert len(rule_engine["findings"]) == 100
    assert rule_engine["metadata"]["findings_truncated"] is True
    assert rule_engine["metadata"]["total_finding_count"] > 100


def test_codet5p_project_requests_are_grouped_without_changing_output_indices():
    requests = [
        {"content": "first", "language": "python"},
        {"content": "second", "language": "java"},
        {"content": "third", "language": "python"},
    ]
    groups = _group_requests_by_route(
        {
            "malicious_intent": {
                "python": "codet5p-python",
                "java": "codet5p-java",
            },
        },
        requests,
    )

    assert [[index for index, _request in group] for group in groups] == [[0, 2], [1]]
    assert [[request["language"] for _index, request in group] for group in groups] == [
        ["python", "python"],
        ["java"],
    ]


def test_rule_based_sql_injection_is_active_while_vulnerability_model_stays_disabled():
    result = _scan_with_rule_engine(
        "vulnerable_sql.py",
        "cursor.execute('select * from users where id=' + request.args.get('id'))",
    )
    assert "SQL Injection" in result["categories"]
    assert result["final_decision"] == "vulnerable"
    assert result["confidence"] is None
    assert "combined_probability" not in result["vulnerability_risk"]
    assert result["vulnerability_risk"]["available"] is False
    assert result["vulnerability_risk"]["status"] == "disabled"
    assert result["vulnerability_risk"]["probability"] is None
    assert result["risk_score"] >= 35
    assert any("参数" in advice for advice in result["repair_suggestions"])
    assert any("OWASP SQL 注入" in item["title"] for item in result["remediation_references"])


def test_sql_injection_dataflow_covers_common_host_languages():
    samples = {
        "python": (
            "demo.py",
            "value = request.args.get('id')\n"
            "sql = 'SELECT * FROM users WHERE id=' + value\n"
            "cursor.execute(sql)",
        ),
        "java": (
            "Demo.java",
            'String value = request.getParameter("id");\n'
            'String sql = "SELECT * FROM users WHERE id=" + value;\n'
            "statement.executeQuery(sql);",
        ),
        "javascript": (
            "demo.js",
            "const value = req.query.id;\n"
            "const sql = `SELECT * FROM users WHERE id=${value}`;\n"
            "db.query(sql);",
        ),
        "php": (
            "demo.php",
            "$value = $_GET['id'];\n"
            "$sql = 'SELECT * FROM users WHERE id=' . $value;\n"
            "mysqli_query($db, $sql);",
        ),
        "go": (
            "demo.go",
            'value := r.URL.Query().Get("id")\n'
            'sql := fmt.Sprintf("SELECT * FROM users WHERE id=%s", value)\n'
            "db.Query(sql)",
        ),
    }

    for language, (filename, code) in samples.items():
        result = _scan_with_rule_engine(filename, code)
        sql_findings = [
            match for match in result["matches"]
            if match.get("category") == "SQL Injection"
        ]
        assert sql_findings, language
        assert result["final_decision"] == "vulnerable"
        assert result["vulnerability_risk"]["status"] == "disabled"


def test_parameterized_sql_is_not_reported_as_injection():
    result = scan_code(
        "safe_query.py",
        "value = request.args.get('id')\n"
        "cursor.execute('SELECT * FROM users WHERE id=?', (value,))",
        mode="quick",
    )

    assert "SQL Injection" not in result["categories"]


def test_local_remediation_catalog_has_source_attributed_language_guidance():
    stats = catalog_statistics()
    catalog = load_remediation_catalog()

    assert stats["categories"] >= 40
    assert stats["total_suggestions"] >= 300
    assert stats["language_suggestions"] >= 25
    assert all(
        entry.get("source_ids")
        and all(source_id in catalog["sources"] for source_id in entry["source_ids"])
        for entry in catalog["entries"].values()
    )


def test_owasp_top10_2025_has_auditable_baseline_coverage_without_absolute_claim():
    summary = coverage_summary()

    assert len(OWASP_TOP10_2025) == 10
    assert {item["id"] for item in OWASP_TOP10_2025} == {
        f"A{index:02d}:2025" for index in range(1, 11)
    }
    assert all(item["status"] == "baseline" for item in OWASP_TOP10_2025)
    assert all(item["detectors"] for item in OWASP_TOP10_2025)
    assert all(item["limitations"] for item in OWASP_TOP10_2025)
    assert summary["covered_categories"] == 10
    assert summary["absolute_coverage_claimed"] is False


def test_new_owasp_baseline_rules_detect_high_signal_security_failures():
    samples = {
        "tls.py": (
            "response = requests.get(url, verify=False)",
            "TLS Verification Disabled",
        ),
        "jwt.py": (
            "claims = jwt.decode(token, options={'verify_signature': False})",
            "JWT Verification Disabled",
        ),
        "password.py": (
            "if password == 'admin123':\n    grant_access()",
            "Plaintext Password Handling",
        ),
        "logging.py": (
            "logger.info('password=%s', password)",
            "Sensitive Data Logging",
        ),
        "crypto.py": (
            "token = Math.random().toString(16)",
            "Insecure Randomness",
        ),
        "error.py": (
            "try:\n    authorize()\nexcept Exception:\n    return True",
            "Fail Open Security Decision",
        ),
        "empty_error.py": (
            "try:\n    save()\nexcept Exception:\n    pass",
            "Empty Exception Handler",
        ),
    }

    for filename, (code, expected_category) in samples.items():
        result = _scan_with_rule_engine(filename, code)
        assert expected_category in result["categories"], filename
        assert result["owasp_categories"], filename
        assert result["owasp_top10_2025"] == {
            "coverage_level": "baseline",
            "covered_categories": 10,
            "total_categories": 10,
            "absolute_coverage_claimed": False,
        }


def test_new_owasp_baseline_rules_ignore_secure_equivalents():
    code = (
        "response = requests.get(url, timeout=10, verify=True)\n"
        "claims = jwt.decode(token, key, algorithms=['RS256'])\n"
        "token = secrets.token_urlsafe(32)\n"
        "logger.info('authentication completed for user_id=%s', user_id)\n"
    )

    result = scan_code("secure.py", code, mode="quick")

    assert "TLS Verification Disabled" not in result["categories"]
    assert "JWT Verification Disabled" not in result["categories"]
    assert "Insecure Randomness" not in result["categories"]
    assert "Sensitive Data Logging" not in result["categories"]


def test_xgboost_line_occlusion_locates_probability_contributing_line():
    content = "safe_setup()\ndangerous_remote_exec()\ncleanup()\n"

    def probability(candidate):
        return 0.9 if "dangerous_remote_exec" in candidate else 0.2

    attributions = _line_attributions(
        content,
        baseline_probability=probability(content),
        predict_probability=probability,
    )

    assert attributions[0]["line"] == 2
    assert attributions[0]["snippet"] == "dangerous_remote_exec()"
    assert attributions[0]["probability_drop"] == 0.7
    assert attributions[0]["contribution_percent"] == 100.0


def test_ai_line_attribution_is_marked_as_corroborated_only_near_evidence():
    engines = [{
        "name": "xgboost_malicious",
        "status": "completed",
        "decision": "malicious",
        "probability": 0.9,
        "threshold": 0.5,
        "metadata": {
            "raw_model_decision": "malicious",
            "line_attributions": [{
                "line": 8,
                "snippet": "execute(user_input)",
                "probability_drop": 0.4,
                "contribution_percent": 100.0,
            }],
        },
    }]
    evidence = [{
        "line": 8,
        "snippet": "execute(user_input)",
        "category": "Command Execution",
        "description": "外部输入进入命令执行接口。",
    }]

    merged, ai_only = merge_model_line_attributions(evidence, engines)

    assert merged[0]["evidence_basis"] == "ai_and_rule"
    assert merged[0]["ai_attribution"]["probability_drop"] == 0.4
    assert ai_only == []


def test_unconfirmed_ai_line_is_labeled_as_attention_signal_not_vulnerability():
    engines = [{
        "name": "xgboost_malicious",
        "status": "completed",
        "decision": "malicious",
        "probability": 0.88,
        "threshold": 0.5,
        "metadata": {
            "raw_model_decision": "malicious",
            "line_attributions": [{
                "line": 3,
                "snippet": "opaque_call(value)",
                "probability_drop": 0.12,
                "contribution_percent": 100.0,
            }],
        },
    }]

    merged, ai_only = merge_model_line_attributions([], engines)

    assert merged == []
    assert ai_only[0]["risk_type"] == "ai_signal"
    assert ai_only[0]["evidence_basis"] == "ai_only"
    assert ai_only[0]["owasp_category"] is None
    assert "不能直接定性" in ai_only[0]["harm"]


def test_risk_taxonomy_adds_api_and_supply_chain_scope_without_overmapping():
    ssrf = taxonomy_for_category("SSRF")
    download = taxonomy_for_category("Download and Execute")
    sql = taxonomy_for_category("SQL Injection")

    assert ssrf["api_security_category"] == "API7:2023 服务端请求伪造"
    assert "API安全" in ssrf["risk_domains"]
    assert "软件供应链" in download["risk_domains"]
    assert "恶意行为" in download["risk_domains"]
    assert sql["api_security_category"] is None


def test_quick_mode_runs_xgboost_and_low_cost_rules():
    result = scan_code("sample.py", "import os\nprint(os.getenv('HOME'))", mode="quick")
    engines = {engine["name"]: engine for engine in result["engines"]}
    assert engines["xgboost_malicious"]["status"] in {"completed", "unavailable"}
    assert (engines["xgboost_malicious"]["probability"] is not None) == (
        engines["xgboost_malicious"]["status"] == "completed"
    )
    assert "xgboost_vulnerability" not in engines
    assert engines["rule_engine"]["status"] == "completed"
    assert "static_evidence" not in engines
    assert "pe_static" not in engines


def test_vulnerability_route_is_disabled_without_deleting_artifacts():
    result = scan_code(
        "Demo.java",
        "class Demo { int add(int a, int b) { return a + b; } }",
        mode="quick",
    )
    assert not any(
        engine["name"] == "xgboost_vulnerability"
        for engine in result["engines"]
    )
    assert result["vulnerability_risk"]["status"] == "disabled"
    assert (BACKEND_DIR / "models" / "xgb_vulnerability_classifier.joblib").is_file()


def test_webshell_sample_is_malicious():
    code = "".join(["ev", "al($", "_POST['cmd']); sy", "stem($", "_GET['x']);"])
    result = _scan_with_rule_engine("webshell.php", code)
    assert "WebShell" in result["categories"]
    assert result["final_decision"] == "malicious"
    assert result["risk_score"] >= 65


def test_command_script_extensions_are_supported_and_detected():
    assert {
        ".ps1", ".psm1", ".psd1", ".bat", ".cmd", ".xhtml", ".hta",
    } <= SOURCE_EXTENSIONS
    assert detect_language(
        "loader.ps1",
        "IEX (New-Object Net.WebClient).DownloadString($u)",
    ) == "powershell"
    assert detect_language(
        "installer.cmd",
        "@echo off\r\nset target=%TEMP%\r\ncall :main",
    ) == "batch"
    assert detect_language(
        "landing.hta",
        "<html><script>function run(){ return 1; }</script></html>",
    ) == "html"
    powershell_graph = build_lightweight_graph(
        "IEX (New-Object Net.WebClient).DownloadString($url)",
        "powershell",
    )
    assert {
        "iex", "new-object", "downloadstring", "behavior_remote_execution_chain",
    } <= set(powershell_graph["dangerous_apis"])
    native_graph = build_lightweight_graph(
        "VirtualAlloc(0, n, 0x3000, 0x40); "
        "WriteProcessMemory(p, x, b, n, 0); CreateRemoteThread(p,0,0,x,0,0,0);",
        "cpp",
    )
    assert "behavior_process_injection_chain" in native_graph["dangerous_apis"]


def test_single_file_evidence_has_chinese_explanation_context_and_suspicion():
    result = _scan_with_rule_engine(
        "webshell.php",
        "safe = 1\neval($_POST['cmd']);\nprint(safe)",
    )
    assert result["evidence_items"][0]["evidence_basis"] == "ai_decision"
    evidence = next(
        item for item in result["evidence_items"]
        if item.get("line") == 2
    )
    assert evidence["line"] == 2
    assert evidence["risk_type"] == "malicious"
    assert evidence["suspicion_score"] == 100
    assert "网页后门" in evidence["description"]
    assert [row["line"] for row in evidence["code_context"]] == [1, 2, 3]
    assert next(row for row in evidence["code_context"] if row["is_target"])["code"] == "eval($_POST['cmd']);"
    assert any("动态执行" in advice for advice in result["repair_suggestions"])


def test_record_view_deduplicates_repair_suggestions_without_dropping_evidence():
    advice = "核对域名/IP/URL 的业务用途、所有权和历史信誉，避免仅凭字符串直接定性。"
    record = _public_record_view({
        "rule_matches": [
            {"risk_type": "context", "line": 1, "repair_advice": advice},
            {"risk_type": "context", "line": 2, "repair_advice": f"  {advice}  "},
        ],
        "engines": [],
        "engine_votes": {},
    })

    assert len(record["rule_matches"]) == 2
    assert record["repair_suggestions"] == [advice]


def test_record_view_keeps_sourced_rule_vulnerability_and_its_catalog_metadata():
    record = _public_record_view({
        "rule_matches": [{
            "source": "rule_engine",
            "risk_type": "vulnerable",
            "category": "SQL Injection",
            "repair_suggestions": ["使用参数化查询。", "限制数据库账户权限。"],
            "remediation_references": [{
                "title": "OWASP SQL 注入防护指南",
                "url": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
            }],
        }],
        "engines": [],
        "engine_votes": {},
    })

    assert len(record["rule_matches"]) == 1
    assert record["repair_suggestions"] == ["使用参数化查询。", "限制数据库账户权限。"]
    assert record["remediation_references"][0]["title"] == "OWASP SQL 注入防护指南"


def test_safe_sample_has_low_risk():
    result = scan_code(
        "safe_code.py",
        "cursor.execute('select * from users where id=?', (user_id,))",
        mode="quick",
    )
    assert result["risk_score"] < 35
    assert result["final_decision"] == "benign"


def test_hashes_are_reproducible_and_validated_model_owns_decision():
    content = "class Safe {}\n"
    result = scan_code("Sample.java", content, mode="quick")
    raw = content.encode("utf-8")
    assert result["hashes"] == {
        "md5": hashlib.md5(raw).hexdigest(),
        "sha1": hashlib.sha1(raw).hexdigest(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    fused = fuse_engine_results([{
        "name": "xgboost_malicious", "status": "completed", "decision": "malicious",
        "probability": 0.968, "threshold": 0.5, "findings": [],
    }])
    assert fused["final_decision"] == "malicious"
    assert fused["decision_basis"] == "ai_model"
    assert fused["decision_authority"] == "ai"
    assert fused["rule_fallback_used"] is False


def test_unvalidated_xgboost_fallback_is_advisory_only():
    fused = fuse_engine_results([{
        "name": "xgboost_malicious",
        "status": "completed",
        "decision": "review",
        "probability": 0.99,
        "threshold": 0.5,
        "findings": [],
        "metadata": {
            "advisory_only": True,
            "raw_model_decision": "malicious",
            "route_quality_gate_passed": False,
        },
    }])
    assert fused["final_decision"] == "unknown"
    assert fused["risk_score"] > 0
    assert fused["decision_basis"] == "unresolved"
    assert fused["rule_fallback_used"] is True
    assert fused["rule_fallback_reason"] == "ai_routes_not_validated"


def test_malicious_rule_cannot_override_decisive_ai_benign_result():
    fused = fuse_engine_results([
        {
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": "benign",
            "probability": 0.08,
            "threshold": 0.5,
            "findings": [],
        },
        {
            "name": "rule_engine",
            "status": "completed",
            "decision": "malicious",
            "findings": [{
                "risk_type": "malicious",
                "category": "WebShell",
                "severity": 10,
                "line": 3,
                "snippet": "eval(input)",
            }],
        },
    ])

    assert fused["final_decision"] == "benign"
    assert fused["decision_authority"] == "ai"
    assert fused["rule_disagrees_with_ai"] is True


def test_uncertain_ai_uses_rules_as_explicit_fallback():
    fused = fuse_engine_results([
        {
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": "malicious",
            "probability": 0.55,
            "threshold": 0.5,
            "metadata": {
                "uncertain_low": 0.45,
                "uncertain_high": 0.6,
            },
        },
        {
            "name": "rule_engine",
            "status": "completed",
            "decision": "malicious",
            "findings": [{
                "risk_type": "malicious",
                "category": "WebShell",
                "severity": 10,
                "line": 3,
                "snippet": "eval(input)",
            }],
        },
    ])

    assert fused["final_decision"] == "malicious"
    assert fused["decision_authority"] == "rule_fallback"
    assert fused["rule_fallback_reason"] == "ai_uncertain"


def test_conflicting_ai_models_use_rules_only_as_tiebreaker():
    fused = fuse_engine_results([
        {
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": "malicious",
            "probability": 0.91,
            "threshold": 0.5,
        },
        {
            "name": "codet5p",
            "status": "completed",
            "decision": "benign",
            "probability": 0.1,
            "threshold": 0.8,
        },
        {
            "name": "rule_engine",
            "status": "completed",
            "decision": "benign",
            "findings": [],
        },
    ])

    assert fused["final_decision"] == "unknown"
    assert fused["decision_authority"] == "unresolved"
    assert fused["ai_conflict"] is True
    assert fused["rule_fallback_reason"] == "ai_model_conflict"


def test_java_path_traversal_rule_is_active_without_enabling_vulnerability_model():
    code = """class Demo {
    void read(HttpServletRequest request) throws Exception {
        String param = URLDecoder.decode(request.getCookies()[0].getValue(), "UTF-8");
        String fileName = TESTFILES_DIR + param;
        FileInputStream fis = new FileInputStream(new File(fileName));
    }
}"""
    result = _scan_with_rule_engine("Demo.java", code)
    assert result["final_decision"] == "vulnerable"
    assert any(match["rule_id"] == "PATH-002" for match in result["matches"])
    assert result["vulnerability_risk"]["status"] == "disabled"


def test_standard_mode_uses_real_codet5p_artifact_when_available():
    statuses = {item["engine"]: item for item in runtime_status()}
    assert statuses["codet5p"]["status"] == "completed"
    result = scan_code("command.py", "import os\nos.system(user_input)", mode="standard")
    engine = next(item for item in result["engines"] if item["name"] == "codet5p")
    assert engine["status"] == "completed"
    assert engine["probability"] is not None
    assert engine["model_version"].startswith("codet5p-")
    assert result["malicious_intent"]["engine"] == "codet5p"
    assert result["confidence"] is None


def test_runtime_status_contains_registered_model_families():
    assert [item["engine"] for item in runtime_status()] == [
        "legacy_svm", "xgboost", "bytetcn", "gatv2", "codet5p",
    ]


def test_txt_file_is_supported_and_scanned():
    result = _scan_with_rule_engine(
        "notes.txt",
        "cursor.execute('select * from users where id=' + request.args.get('id'))",
    )
    assert result["language"] == "unknown"
    assert "SQL Injection" in result["categories"]
    assert result["final_decision"] == "vulnerable"


def test_plain_txt_displays_txt_when_content_language_is_unknown():
    content = "Meeting notes for Friday. Nothing executable here."

    assert detect_language("notes.txt", content) == "unknown"
    assert display_language("unknown", "notes.txt") == "text"


def test_txt_python_api_without_import_or_def_is_routed_as_python():
    content = "os.system(user_input)\neval(payload)\n"

    assert detect_language("payload.txt", content) == "python"
    result = scan_code("payload.txt", content, mode="quick")
    assert result["language"] == "python"
    assert result["display_language"] == "python"
    assert "Command Execution" in result["categories"]


def test_quick_rules_detect_python_popen_shell_true_in_txt():
    content = "from subprocess import Popen\nPopen(user_input, shell=True)\n"

    result = scan_code("payload.txt", content, mode="quick")

    assert result["language"] == "python"
    assert "Command Execution" in result["categories"]
    assert result["final_decision"] == "vulnerable"


def test_txt_php_content_is_routed_as_php():
    code = "<?php eval($_POST['cmd']); system($_GET['x']); ?>"
    assert detect_language("uploaded.txt", code) == "php"
    result = scan_code("uploaded.txt", code, mode="quick")
    assert result["language"] == "php"
    assert result["final_decision"] == "malicious"


def test_project_summary_ranks_high_risk_files():
    risky = scan_code("a.php", "eval($_POST['cmd']);", mode="quick")
    safe = scan_code("b.py", "def add(a, b): return a + b", mode="quick")
    summary = summarize_project("demo.zip", [safe, risky])
    assert summary["file_count"] == 2
    assert summary["high_risk_files"][0]["filename"] == "a.php"
    assert summary["risk_level"] in {"medium", "high", "critical"}


def test_project_summary_returns_every_ranked_file_for_frontend_pagination():
    results = [
        {
            "filename": f"file-{index:02d}.py",
            "language": "python",
            "risk_score": index,
            "risk_level": "safe",
            "final_decision": "benign",
            "categories": [],
            "engines": [],
        }
        for index in range(23)
    ]
    summary = summarize_project("all-files.zip", results)

    assert len(summary["high_risk_files"]) == 23
    assert [item["risk_score"] for item in summary["high_risk_files"]] == list(
        range(22, -1, -1)
    )


def test_project_summary_audits_ai_participation_and_rule_fallback():
    results = [
        {
            "filename": "ai.py",
            "language": "python",
            "risk_score": 80,
            "risk_level": "high",
            "final_decision": "malicious",
            "decision_authority": "ai",
            "ai_participated": True,
            "ai_model_names": ["xgboost_malicious"],
            "categories": [],
            "engines": [],
        },
        {
            "filename": "fallback.ts",
            "language": "typescript",
            "risk_score": 70,
            "risk_level": "high",
            "final_decision": "malicious",
            "decision_authority": "rule_fallback",
            "ai_participated": True,
            "ai_model_names": ["xgboost_malicious"],
            "categories": [],
            "engines": [],
        },
        {
            "filename": "tool.exe",
            "language": "binary",
            "risk_score": 0,
            "risk_level": "safe",
            "final_decision": "unknown",
            "decision_authority": "unresolved",
            "ai_participated": False,
            "categories": [],
            "engines": [],
        },
    ]

    summary = summarize_project("audit.zip", results)

    assert summary["ai_eligible_file_count"] == 2
    assert summary["ai_ineligible_file_count"] == 1
    assert summary["ai_participation_rate"] == 100.0
    assert summary["ai_primary_decision_rate"] == 50.0
    assert summary["rule_fallback_file_count"] == 1
    assert summary["ai_participation_target_met"] is True


def test_project_summary_includes_completed_gatv2_decision():
    summary = summarize_project("demo.zip", [], project_engines=[{
        "name": "gatv2", "status": "completed", "decision": "malicious",
        "probability": 0.82, "threshold": 0.21, "model_version": "gatv2-test",
    }])
    assert summary["final_decision"] == "malicious"
    assert summary["max_score"] == 82
    assert summary["project_engines"][0]["probability"] == 0.82


def test_project_xgboost_uses_validated_max_file_aggregation():
    results = [
        {
            "filename": "pkg/a.py",
            "language": "python",
            "engines": [{
                "name": "xgboost_malicious",
                "status": "completed",
                "decision": "review",
                "probability": 0.41,
                "threshold": 0.5,
                "model_version": "test",
                "metadata": {
                    "task": "malicious_intent",
                    "evaluation_scope": "project_or_package",
                    "route_quality_gate_passed": True,
                    "source_heldout_verified": True,
                    "advisory_only": True,
                    "raw_model_decision": "benign",
                },
            }],
            "risk_score": 0,
            "risk_level": "safe",
            "final_decision": "benign",
            "categories": [],
        },
        {
            "filename": "pkg/b.py",
            "language": "python",
            "engines": [{
                "name": "xgboost_malicious",
                "status": "completed",
                "decision": "review",
                "probability": 0.91,
                "threshold": 0.5,
                "model_version": "test",
                "metadata": {
                    "task": "malicious_intent",
                    "evaluation_scope": "project_or_package",
                    "route_quality_gate_passed": True,
                    "source_heldout_verified": True,
                    "advisory_only": True,
                    "raw_model_decision": "malicious",
                },
            }],
            "risk_score": 0,
            "risk_level": "safe",
            "final_decision": "benign",
            "categories": [],
        },
    ]
    engines = aggregate_project_xgboost(results)
    assert len(engines) == 1
    assert engines[0]["decision"] == "malicious"
    assert engines[0]["probability"] == 0.91
    assert engines[0]["metadata"]["top_file"] == "pkg/b.py"
    summary = summarize_project("demo.zip", results, project_engines=engines)
    assert summary["final_decision"] == "malicious"
    assert summary["max_score"] == 91


def test_project_summary_preserves_rule_based_vulnerability_decision():
    summary = summarize_project("demo.zip", [{
        "filename": "app.py",
        "language": "python",
        "engines": [],
        "risk_score": 66,
        "risk_level": "high",
        "final_decision": "vulnerable",
        "categories": ["SQL Injection"],
    }])

    assert summary["final_decision"] == "vulnerable"
    assert summary["decision_counts"]["vulnerable"] == 1
    assert summary["category_counts"]["SQL Injection"] == 1


def test_failed_source_heldout_project_route_is_not_aggregated():
    results = [{
        "filename": "Demo.java",
        "language": "java",
        "engines": [{
            "name": "xgboost_vulnerability",
            "status": "completed",
            "decision": "review",
            "probability": 0.99,
            "threshold": 0.2,
            "metadata": {
                "task": "vulnerability_risk",
                "evaluation_scope": "project_or_package",
                "route_quality_gate_passed": True,
                "source_heldout_verified": False,
                "advisory_only": True,
                "raw_model_decision": "vulnerable",
            },
        }],
    }]
    assert aggregate_project_xgboost(results) == []


def test_bytetcn_manifest_routes_only_independently_gated_task_languages():
    manifest = json.loads((BACKEND_DIR / "models" / "bytetcn_manifest.json").read_text(encoding="utf-8"))
    assert "python" in manifest["task_language_support"]["malicious_intent"]
    assert "java" in manifest["task_language_support"]["vulnerability_risk"]
    assert set(manifest["supported_languages"]) == {
        language
        for languages in manifest["task_language_support"].values()
        for language in languages
    }


def test_bytetcn_task_settings_select_language_route_and_keep_legacy_fallback():
    routed = {
        "task_models": {
            "malicious_intent": {
                "by_language": {
                    "python": {"file": "python.pt", "threshold": 0.8},
                    "php": {"file": "php.pt", "threshold": 0.6},
                }
            }
        }
    }
    assert _task_settings(routed, "malicious_intent", "php")["file"] == "php.pt"

    legacy = {"task_models": {"malicious_intent": {"file": "legacy.pt"}}}
    assert _task_settings(legacy, "malicious_intent", "python")["file"] == "legacy.pt"


def test_task_language_eligibility_requires_both_classes_in_every_split():
    records = {
        split: [
            {"language": "java", "vulnerability_risk": label}
            for label in ([0] * 20 + [1] * 20)
        ] + [
            {"language": "php", "vulnerability_risk": 1}
            for _ in range(40)
        ]
        for split in ("train", "validation", "test")
    }
    assert _eligible_languages(
        records, "vulnerability_risk",
        {"train": 20, "validation": 20, "test": 20},
    ) == ["java"]


def test_model_center_uses_real_current_and_historical_versions():
    view = model_center_view()
    assert [group["key"] for group in view["version_groups"]] == [
        "xgboost", "codet5p", "gatv2",
    ]
    groups = {group["key"]: group for group in view["version_groups"]}
    assert set(groups) == {"xgboost", "codet5p", "gatv2"}
    assert "codet5p-220m-base" in {
        version["version"] for version in groups["codet5p"]["versions"]
    }
    codet5p = groups["codet5p"]
    assert codet5p["active_version"].startswith("codet5p-active-")
    assert codet5p["active_version_label"] == codet5p["active_version"]
    published_versions = {
        version["version"]
        for version in codet5p["versions"]
        if version["published"]
    }
    assert set(codet5p["active_versions"]) <= published_versions
    assert codet5p["active_version"] in published_versions
    assert {
        language
        for version in codet5p["versions"]
        if version["published"]
        for task in version["tasks"]
        for language in (metric["language"] for metric in task["language_metrics"])
    } >= {
        "bash", "c", "cpp", "java", "javascript", "php", "powershell", "python",
    }
    assert groups["gatv2"]["active_version"].startswith("gatv2-")
    assert (BACKEND_DIR / "models" / "registry.json").is_file()
    assert (BACKEND_DIR / "models" / "bytetcn_manifest.json").is_file()
    assert all(
        set(task) >= {
            "accuracy", "precision", "false_positive_rate",
            "false_negative_rate", "f1", "language_metrics",
        }
        for group in groups.values() for version in group["versions"] for task in version["tasks"]
    )
    assert {
        task["task"]
        for group in groups.values()
        for version in group["versions"]
        for task in version["tasks"]
    } <= {"malicious_intent", "project_malicious_intent"}


def test_active_gatv2_routes_eleven_strict_gated_languages():
    manifest = json.loads((BACKEND_DIR / "models" / "gatv2_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "bash", "c", "config", "cpp", "csharp", "html",
        "java", "javascript", "php", "powershell", "python",
    }
    assert set(manifest["supported_languages"]) == expected
    assert set(manifest["language_models"]) == expected
    for language in expected:
        metrics = manifest["test_metrics_by_language"][language]
        assert metrics["precision"] >= 0.90
        assert metrics["false_positive_rate"] <= 0.10
        assert metrics["false_negative_rate"] <= 0.10
        assert (BACKEND_DIR / "models" / manifest["language_models"][language]["file"]).is_file()

    rows = {row["model"]: row for row in model_center_view()["performance_rows"]}
    assert "ByteCNN-TCN" not in rows
    assert rows["GATv2"]["scope"] == (
        "已验证语言合并测试集 · 项目级依赖图 · "
        "BASH / SHELL / C / CONFIG / C++ / C# / HTML / HTA / JAVA / JAVASCRIPT / PHP / POWERSHELL / PYTHON"
    )
    assert rows["GATv2"]["samples"] == 1542
    assert round(rows["GATv2"]["accuracy"], 6) == 0.953307
    assert round(rows["GATv2"]["precision"], 6) == 0.970402
    assert round(rows["GATv2"]["false_positive_rate"], 6) == 0.048276
    assert round(rows["GATv2"]["false_negative_rate"], 6) == 0.045738


def test_model_center_shows_quick_standard_deep_models_by_strengths_and_limits():
    view = model_center_view()
    assert [row["model"] for row in view["performance_rows"]] == [
        "XGBoost", "CodeT5+ 220M", "GATv2",
    ]
    rows = {row["model"]: row for row in view["performance_rows"]}
    assert set(rows) == {"XGBoost", "CodeT5+ 220M", "GATv2"}
    assert all(row["advantage"] != "暂无" and row["limitation"] != "暂无" for row in rows.values())
    assert "快速模式" in rows["XGBoost"]["advantage"]
    assert "跨文件关系强" in rows["GATv2"]["advantage"]
    assert "代码语义" in rows["CodeT5+ 220M"]["advantage"]
    assert all(row["advantage"].count("；") == 2 for row in rows.values())
    assert all(row["limitation"].count("；") == 2 for row in rows.values())

    template = (
        BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "models.html"
    ).read_text(encoding="utf-8")
    current_table = template.split('id="performance"', 1)[1].split('id="versions"', 1)[0]
    version_table = template.split('id="versions"', 1)[1].split('id="training-jobs"', 1)[0]
    assert "<strong>优点</strong>" in current_table
    assert "<strong>缺点</strong>" in current_table
    assert "<th>评测范围</th>" not in current_table
    assert "<th>准确率</th><th>精确率</th>" in current_table
    assert "data-model-language-select" not in current_table
    assert "data-version-language-select" in version_table
    assert "data-language-evaluation-note" not in version_table
    assert "n={{ language.samples }}" not in version_table
    assert "{% if language.full_metrics %}" in version_table
    assert "（已支持，指标不足）" not in version_table
    assert "<th>评测范围</th>" in version_table
    assert "<th>语言 / 评测范围</th>" not in version_table
    assert "selectattr('full_metrics')" in version_table
    assert "map(attribute='language_label')" in version_table
    assert "option.dataset.metricNote\n                ? option.dataset.languageLabel" in template
    assert version_table.count("data-six-row-select") >= 2
    assert "Math.min(6, select.options.length)" in template
    assert 'data-version-metric="accuracy"' in version_table
    assert "row.advantage" in current_table
    assert "row.limitation" in current_table

    groups = {group["key"]: group for group in view["version_groups"]}
    xg_active = next(
        version
        for version in groups["xgboost"]["versions"]
        if version["version"] == groups["xgboost"]["active_version"]
    )
    xg_languages = xg_active["tasks"][0]["language_metrics"]
    assert len(xg_languages) == 22
    assert {metric["language"] for metric in xg_languages} >= {
        "bash", "c", "cpp", "csharp", "go", "html", "java",
        "javascript", "php", "python", "ruby", "rust", "typescript",
    }
    assert {
        metric["language"]
        for metric in xg_languages
        if metric["full_metrics"]
    } == {
        "bash", "c", "config", "cpp", "html",
        "go", "java", "javascript", "php", "powershell",
        "python", "ruby", "rust",
    }
    assert all(metric["metric_note"] for metric in xg_languages)
    assert all(
        "n=" not in metric["metric_note"]
        for metric in xg_languages
        if metric["full_metrics"]
    )
    assert rows["XGBoost"]["samples"] == 5860
    assert rows["XGBoost"]["accuracy"] == 0.9797
    assert rows["XGBoost"]["precision"] == 0.9597
    assert rows["XGBoost"]["false_positive_rate"] == 0.0143
    assert rows["XGBoost"]["false_negative_rate"] == 0.0371

    for group_key, model_name in (
        ("xgboost", "XGBoost"),
        ("codet5p", "CodeT5+ 220M"),
        ("gatv2", "GATv2"),
    ):
        group = groups[group_key]
        active = next(
            version
            for version in group["versions"]
            if version["version"] == group["active_version"]
        )
        summary = active["tasks"][0]
        for metric_name in (
            "accuracy", "precision", "false_positive_rate",
            "false_negative_rate", "f1", "samples",
        ):
            assert rows[model_name][metric_name] == summary[metric_name]

    codet5p_summary = next(
        version
        for version in groups["codet5p"]["versions"]
        if version["version"] == groups["codet5p"]["active_version"]
    )["tasks"][0]
    assert codet5p_summary["samples"] == 4587
    assert round(codet5p_summary["accuracy"], 6) == 0.985394
    assert round(codet5p_summary["precision"], 6) == 0.978440
    assert round(codet5p_summary["false_positive_rate"], 6) == 0.011292
    assert round(codet5p_summary["false_negative_rate"], 6) == 0.020939


def test_model_center_runtime_and_training_options_hide_report_only_models():
    runtime = _visible_runtime_models([
        {"engine": "legacy_svm", "name": "TF-IDF / SVM 对照组"},
        {"engine": "gatv2", "name": "GATv2"},
        {"engine": "bytetcn", "name": "ByteCNN-TCN"},
        {"engine": "xgboost", "name": "XGBoost"},
        {"engine": "codet5p", "name": "CodeT5+ 220M"},
    ])
    assert [model["engine"] for model in runtime] == [
        "xgboost", "codet5p", "gatv2",
    ]

    options = _training_model_options(model_center_view())
    families = list(dict.fromkeys(option["family"] for option in options))
    assert families == ["xgboost", "codet5p", "gatv2"]


def test_model_center_overall_uses_all_language_test_samples():
    rows = _task_rows({
        "malicious_intent": {
            "accuracy": 0.4,
            "precision": 0.1,
            "false_positive_rate": 0.8,
            "false_negative_rate": 0.7,
            "f1": 0.2,
            "supported_languages": ["python"],
            "deployment": {
                "precision": 0.93,
                "false_positive_rate": 0.03,
                "false_negative_rate": 0.04,
                "f1": 0.94,
            },
        },
    })
    assert rows[0]["accuracy"] == 0.4
    assert rows[0]["precision"] == 0.93
    assert rows[0]["false_positive_rate"] == 0.03
    assert rows[0]["false_negative_rate"] == 0.04
    assert rows[0]["scope"] == "已验证语言合并测试集 · PYTHON"
    assert TASK_LABELS["project_malicious_intent"] == "恶意代码检测"


def test_training_steps_and_history_search_copy_are_consistent():
    model_template = (
        BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "models.html"
    ).read_text(encoding="utf-8")
    assert 'name="target_language"' not in model_template
    assert "选择目标语言" not in model_template
    assert '<span data-training-file-step>2. 选择本地训练集文件</span>' in model_template
    assert "从训练集的 language 字段或项目图的文件节点中自动识别语言" in model_template

    history_template = (
        BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "history.html"
    ).read_text(encoding="utf-8")
    assert 'placeholder="搜索文件名或哈希值"' in history_template
    assert "搜索文件名、哈希或风险类别" not in history_template


def test_file_upload_languages_follow_malware_source_prevalence_order():
    detection_template = (
        BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "index.html"
    ).read_text(encoding="utf-8")
    assert "支持 {{ single_file_language_labels|join('、') }} 及 EXE/DLL/SYS/OCX，最大 20 MB。" in detection_template


def test_auto_mode_copy_describes_mode_selection():
    detection_template = (
        BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "index.html"
    ).read_text(encoding="utf-8")
    assert (
        "('auto','自动模式','XGBoost + 规则引擎 + 按需 CodeT5+ 220M / GATv2')"
        in detection_template
    )
    assert (
        "('auto','自动模式','XGBoost + 按需 CodeT5+ 220M / 规则引擎')"
        in detection_template
    )
    assert "规则引擎 + XGBoost + 按需 CodeT5+ / GATv2" not in detection_template


def test_release_gate_requires_all_three_metrics():
    assert meets_quality_gate({
        "precision": 0.90,
        "false_positive_rate": 0.10,
        "false_negative_rate": 0.10,
        "quality_gate_passed": True,
    })
    assert not meets_quality_gate({
        "precision": 0.899,
        "false_positive_rate": 0.01,
        "false_negative_rate": 0.01,
        "quality_gate_passed": False,
    })
    assert not meets_quality_gate({
        "precision": 0.99,
        "false_positive_rate": 0.01,
        "false_negative_rate": 0.051,
        "quality_gate_passed": False,
    })


def test_failed_xgboost_candidate_does_not_replace_active_files(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    version_dir = model_dir / "xgb_registry" / "candidate"
    version_dir.mkdir(parents=True)
    registry_path = model_dir / "xgb_registry.json"
    runtime_metrics = model_dir / "xgb_metrics.json"
    runtime_metrics.write_text("active", encoding="utf-8")
    (version_dir / "xgb_metrics.json").write_text("candidate", encoding="utf-8")
    registry_path.write_text(json.dumps({
        "schema_version": 1,
        "active_version": "active-version",
        "versions": [{"version": "active-version"}],
    }), encoding="utf-8")
    monkeypatch.setattr(xgb_trainer, "MODEL_DIR", model_dir)
    monkeypatch.setattr(xgb_trainer, "REGISTRY_PATH", registry_path)
    xgb_trainer._register("candidate", version_dir, {
        "created_at": "2026-07-21T00:00:00+00:00",
        "dataset_sha256": "abc",
        "samples_training_eligible": 10,
        "tasks": {},
    }, activate=False)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["active_version"] == "active-version"
    assert runtime_metrics.read_text(encoding="utf-8") == "active"


def test_detection_center_single_file_form_has_upload_only():
    template = (BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "index.html").read_text(encoding="utf-8")
    record_template = (BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "record.html").read_text(encoding="utf-8")
    compare_template = (BACKEND_DIR.parent / "frontend" / "templates" / "attack" / "compare.html").read_text(encoding="utf-8")
    route = (BACKEND_DIR / "web" / "routes" / "attack_routes.py").read_text(encoding="utf-8")
    assert 'name="code_file"' in template
    assert 'name="code_text"' not in template
    assert 'request.form.get("code_text")' not in route
    assert '<span class="eyebrow">检测结果</span>' not in template
    assert "DETECTION RESULT" not in template
    assert "严重度：" not in template
    assert "综合风险分" not in template
    assert "risk-score-orb" in template
    assert "risk-score-orb" in record_template
    assert "record.repair_suggestions" in record_template
    assert "AI与规则一致" in template
    assert "AI重点关注" in template
    assert "<strong>危害</strong>" in template
    assert "<strong>检测依据</strong>" in template
    assert "贡献度只解释模型判断，不是漏洞概率" in template
    assert "record.evidence_items" in record_template
    assert "用于跨平台核对同一份文件；指纹本身不是恶意证据。" not in template
    assert "用于跨平台核对同一份文件；指纹本身不是恶意证据。" not in record_template
    assert "<span>{{ engine.status|zh }}</span>" in template
    assert "<td>{{ engine.status|zh }}</td>" in template
    assert "<span>{{ engine.status|zh }}</span>" in record_template
    assert "<span>{{ engine.status|zh }}</span>" in compare_template
    assert "本次已执行" in template
    assert "未启用能力" in template
    assert "本次已执行" in record_template
    assert "未启用能力" in record_template
    assert "<strong>{{ engine.name|zh }}</strong>" in template
    assert "engine.status == 'completed'" in template
    for heading in (
        "引擎名称",
        "执行状态",
        "恶意概率（0～1）",
        "判定阈值（0～1）",
        "耗时（毫秒）",
        "模型版本",
    ):
        assert heading in template
        assert heading in record_template
    assert "恶意概率（0～1）" in compare_template


def test_auxiliary_analysis_view_separates_completed_and_inactive_source_capabilities():
    view = _auxiliary_analysis_view({
        "language": "python",
        "engines": [
            {
                "name": "static_evidence",
                "status": "completed",
                "metadata": {"ioc_count": 2, "decoded_count": 1},
                "findings": [{"source": "behavior_chain"}],
            },
            {
                "name": "hash_reputation",
                "status": "unavailable",
                "reason": "external hash reputation is disabled",
            },
            {
                "name": "isolated_sandbox",
                "status": "skipped",
                "reason": "sandbox submission requires XIEZHI_SANDBOX_AUTO_SCAN=1",
            },
        ],
    })

    assert [item["name"] for item in view["executed"]] == [
        "字符串与 IOC",
        "静态去混淆",
        "行为链",
    ]
    inactive = {item["name"]: item["detail"] for item in view["inactive"]}
    assert inactive["PE/DLL 只读解析"] == "当前不是 EXE/DLL/SYS/OCX 文件"
    assert inactive["SHA256 外部信誉"] == "未配置/未查询"
    assert inactive["隔离动态沙箱"] == "已配置服务，但未开启自动提交"


def test_auxiliary_analysis_view_reports_binary_components_truthfully():
    view = _auxiliary_analysis_view({
        "language": "binary",
        "engines": [
            {
                "name": "pe_static",
                "status": "completed",
                "metadata": {"ioc_count": 1, "section_count": 3},
            },
        ],
    })

    assert [item["name"] for item in view["executed"]] == [
        "字符串与 IOC",
        "PE/DLL 只读解析",
    ]
    inactive = {item["name"]: item["detail"] for item in view["inactive"]}
    assert inactive["静态去混淆"] == "当前二进制文件不适用"
    assert inactive["行为链"] == "当前二进制文件不适用"
    assert inactive["SHA256 外部信誉"] == "未配置/未查询"
    assert inactive["隔离动态沙箱"] == "未配置/未提交"


def test_detection_center_copy_matches_runtime_models_and_upload_limits():
    template_root = BACKEND_DIR.parent / "frontend" / "templates" / "attack"
    index = (template_root / "index.html").read_text(encoding="utf-8")
    history = (template_root / "history.html").read_text(encoding="utf-8")

    assert "搜索、筛选、查看报告，并对两条明确选择的检测记录进行对比。" not in history
    assert "项目任务与单文件记录分开保存；点击可重新打开结果。" not in history
    assert "选择检测目标与可用模式，结果中的概率、状态、版本和耗时均来自后端实际执行。" not in index
    assert index.count("('quick','快速模式','XGBoost')") == 2
    assert "('quick','快速模式','XGBoost + 规则引擎')" not in index
    assert "规则与静态分析" not in index
    assert "ByteCNN-TCN" not in index
    assert "当前仅支持 ZIP（.zip）压缩包，最大 1 GB" in index
    assert 'name="project_zip" accept=".zip"' in index
    assert 'name="line_explanations"' not in index
    assert "<em>{{ state.reason }}</em>" not in index

    accept = index.split('name="code_file" accept="', 1)[1].split('"', 1)[0]
    assert accept == "{{ single_file_accept }}"
    upload_contract = _single_file_upload_contract()
    assert set(upload_contract["extensions"]) >= BINARY_EXTENSIONS
    assert upload_contract["accept"] == ",".join(sorted(upload_contract["extensions"]))

    frontend_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in template_root.glob("*.html")
    )
    assert "ByteCNN-TCN" not in frontend_text
    assert "bytetcn" not in frontend_text
    assert "CodeT5+" not in frontend_text.replace("CodeT5+ 220M", "")
    assert "PROJECT RESULT" not in index
    assert "仅显示本次项目图实际执行结果" not in index
    assert "data-paginated-table" in index
    assert 'data-page-size="10"' in index
    assert "最多显示 10 项" not in index

    project_scan = (template_root / "project_scan.html").read_text(encoding="utf-8")
    assert "data-paginated-table" in project_scan
    all_templates = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (BACKEND_DIR.parent / "frontend" / "templates").rglob("*.html")
    )
    for stale_english_copy in (
        "REAL SURFACE PARTICLES",
        "CREATE ACCOUNT",
        "RECORD COMPARISON",
        "REPORT #",
        "Detection mode",
        " is running in ",
        " for real status",
    ):
        assert stale_english_copy not in all_templates
    assert 'data-page-size="10"' in project_scan

    style = (
        BACKEND_DIR.parent / "frontend" / "static" / "css" / "style.css"
    ).read_text(encoding="utf-8")
    assert "text-align: center;" in style
    assert "vertical-align: middle;" in style
    assert '.page-jump input[type="number"]::-webkit-inner-spin-button' in style
    assert ".evidence-explanation-grid > .evidence-remediation" in style

    project_detail = (
        template_root / "project_file_detail.html"
    ).read_text(encoding="utf-8")
    assert "· 逐行遮挡定位" not in project_detail
    assert "处仅由AI重点关注" not in project_detail


def test_project_job_tray_does_not_restore_finished_history_and_keeps_close_by_stop():
    frontend_root = BACKEND_DIR.parent / "frontend"
    base = (frontend_root / "templates" / "base.html").read_text(encoding="utf-8")
    script = (frontend_root / "static" / "js" / "app.js").read_text(encoding="utf-8")

    assert "scan-job-tray-head" not in base
    assert "data-dismiss-job-tray" not in base
    assert "seenActiveJobs" in script
    assert "seenActiveJobs.has(jobId) && terminalStatuses.has(job.status)" in script
    assert 'class="scan-job-actions">${action}${dismiss}</div>' in script
    assert 'data-cancel-job="${jobId}"' in script
    assert 'data-dismiss-job="${jobId}"' in script
    assert "xiezhi-dismissed-scan-jobs" not in script


def test_frontend_and_web_training_expose_no_vulnerability_task():
    template_root = BACKEND_DIR.parent / "frontend" / "templates"
    model_template = (template_root / "attack" / "models.html").read_text(encoding="utf-8")
    frontend_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in template_root.rglob("*.html")
    )
    route = (BACKEND_DIR / "web" / "routes" / "attack_routes.py").read_text(encoding="utf-8")
    assert "漏洞风险" not in frontend_text
    assert "vulnerability_risk" not in frontend_text
    assert "TF-IDF" not in frontend_text
    assert "ByteCNN-TCN" not in frontend_text
    assert 'request.form.get("training_task"' not in route
    assert 'request.form.get("target_language"' not in route
    assert 'target_language = "all"' in route
    assert "<th>任务</th>" not in model_template
    assert 'name="training_task"' not in model_template
    assert model_template.count("<li>") == 4
    assert 'class="model-footnotes' in model_template
