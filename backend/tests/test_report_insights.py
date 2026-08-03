from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from attack_detection.data_pipeline import make_sample
from attack_detection.features.graph_builder import build_project_relationship_graph
from attack_detection.report import render_record_markdown
from attack_detection.remediation import remediation_for_finding
from attack_detection.report_insights import (
    build_evidence_groups,
    build_file_report_insights,
    build_project_report_insights,
)
from attack_detection.explainability import merge_model_line_attributions
from attack_detection.rules import _java_path_traversal_dataflow, _sql_injection_dataflow
from attack_detection.static_analysis.behavior_chains import detect_behavior_chains


def test_project_relationship_graph_only_draws_resolved_local_files():
    samples = [
        make_sample(
            '#include "util.h"\n#include <stdio.h>\nint main(void) { return util(); }',
            label="benign", category="runtime", language="c",
            source="test", package_name="demo", version="1",
            family="demo", split="runtime", file_path="src/main.c",
        ),
        make_sample(
            "int util(void) { return 0; }",
            label="benign", category="runtime", language="c",
            source="test", package_name="demo", version="1",
            family="demo", split="runtime", file_path="src/util.h",
        ),
    ]
    graph = build_project_relationship_graph(
        samples,
        {
            "src/main.c": {"risk_score": 72, "risk_level": "high", "language": "c"},
            "src/util.h": {"risk_score": 8, "risk_level": "safe", "language": "c"},
        },
    )

    paths = {node["id"]: node["path"] for node in graph["nodes"]}
    assert graph["node_count"] == 2
    assert graph["edge_count"] == 1
    assert [
        (paths[edge["source"]], paths[edge["target"]], edge["relation"])
        for edge in graph["edges"]
    ] == [("src/main.c", "src/util.h", "include")]


def test_project_report_exposes_graph_layout_and_timeout_reason():
    graph = {
        "node_count": 2,
        "edge_count": 1,
        "nodes": [
            {"id": "file-1", "path": "src/main.c", "name": "main.c", "risk_score": 72, "risk_level": "high", "degree": 1},
            {"id": "file-2", "path": "src/util.h", "name": "util.h", "risk_score": 8, "risk_level": "safe", "degree": 1},
        ],
        "edges": [{"source": "file-1", "target": "file-2", "relation": "include"}],
    }
    insights = build_project_report_insights({
        "project_relationship_graph": graph,
        "project_engines": [{
            "name": "gatv2",
            "status": "failed",
            "error": "GATv2 inference process failed: timed out after 12.0 seconds",
            "duration_ms": 12100,
        }],
    })

    assert insights["graph_model_status"]["label"] == "执行超时"
    assert "12.0 秒" in insights["graph_model_status"]["detail"]
    assert insights["relationship_graph"]["displayed_edge_count"] == 1
    assert insights["relationship_graph"]["edges"][0]["path"].startswith("M ")


def test_project_gat_status_exposes_its_own_decision():
    insights = build_project_report_insights({
        "project_engines": [{
            "name": "gatv2",
            "status": "completed",
            "decision": "benign",
            "probability": 0.456,
            "threshold": 0.5,
        }],
    })

    assert "恶意概率 45.6%" in insights["graph_model_status"]["detail"]
    assert "判定阈值 50.0%" in insights["graph_model_status"]["detail"]
    assert "GATv2判定 正常" in insights["graph_model_status"]["detail"]


def test_file_report_insights_separate_probability_threshold_and_risk_score():
    report = {
        "file_hash": "same-sha256",
        "risk_score": 91,
        "engines": [{
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": "malicious",
            "probability": 0.73,
            "threshold": 0.60,
            "model_version": "xgb-test",
        }, {
            "name": "rule_engine",
            "status": "completed",
            "decision": "not_applicable",
            "probability": None,
            "threshold": None,
        }],
        "evidence_items": [{
            "line": 8,
            "category": "Command Execution",
            "evidence_basis": "ai_and_rule",
            "ai_attribution": {
                "contribution_percent": 42.5,
                "probability_drop": 0.13,
            },
        }],
    }
    insights = build_file_report_insights(report)

    assert insights["decision_ledger"][0]["probability_percent"] == 73.0
    assert insights["decision_ledger"][0]["threshold_percent"] == 60.0
    assert insights["decision_ledger"][0]["margin"] == 13.0
    assert len(insights["decision_ledger"]) == 1
    assert insights["decision_ledger"][0]["authority_label"] == "参与AI判断"
    assert "model_column_chart" not in insights
    assert insights["evidence_chart"]["total"] == 1
    assert insights["evidence_chart"]["entries"][0]["label"] == "Command Execution"
    assert "evidence_basis_chart" not in insights
    assert "contribution_rows" not in insights
    assert "trend_chart" not in insights


def test_evidence_groups_merge_repeated_weakness_guidance_only():
    repeated_example = {
        "id": "CVE-2023-34362",
        "title": "SQL 注入案例",
        "summary": "公开案例摘要",
        "url": "https://example.test/CVE-2023-34362",
    }
    groups = build_evidence_groups([
        {
            "line": 3,
            "category": "SQL Injection",
            "cwe": "CWE-89",
            "harm": "攻击者可能读写数据库。",
            "repair_suggestions": ["使用参数化查询。"],
            "cve_examples": [repeated_example],
        },
        {
            "line": 9,
            "category": "SQL Injection",
            "cwe": "cwe-89",
            "harm": "攻击者可能读写数据库。",
            "repair_suggestions": ["  使用参数化查询。  "],
            "cve_examples": [repeated_example],
        },
        {
            "line": 12,
            "category": "SQL Injection",
            "cwe": "CWE-564",
            "repair_suggestions": ["使用 ORM 参数绑定。"],
        },
    ])

    assert len(groups) == 2
    assert groups[0]["count"] == 2
    assert [item["line"] for item in groups[0]["items"]] == [3, 9]
    assert groups[0]["harms"] == ["攻击者可能读写数据库。"]
    assert groups[0]["repair_suggestions"] == ["使用参数化查询。"]
    assert [item["id"] for item in groups[0]["cve_examples"]] == ["CVE-2023-34362"]
    assert groups[1]["cwe"] == "CWE-564"


def test_project_report_insights_build_distributions_and_rankings():
    report = {
        "project_name": "demo.zip",
        "level_counts": {"high": 1, "safe": 1},
        "language_counts": {"Python": 1, "JavaScript": 1},
        "category_counts": {"SQL Injection": 2, "Secret Exposure": 1},
        "file_results": [
            {"filename": "a.py", "risk_score": 88, "risk_level": "high", "decision_authority": "ai", "categories": ["SQL Injection", "SQL Injection"]},
            {"filename": "b.py", "risk_score": 12, "risk_level": "safe", "decision_authority": "unresolved", "categories": ["Secret Exposure"]},
        ],
    }
    insights = build_project_report_insights(report)

    assert insights["risk_chart"]["total"] == 2
    assert insights["risk_chart"]["entries"][0]["svg_path"].startswith("M ")
    assert insights["risk_chart"]["entries"][0]["start_percent"] == 0.0
    assert insights["language_chart"]["total"] == 2
    assert {item["label"] for item in insights["language_chart"]["entries"]} == {
        "Python", "JavaScript",
    }
    assert "authority_chart" not in insights
    assert insights["category_rows"][0]["label"] == "SQL Injection"
    assert insights["top_file_rows"][0]["label"] == "a.py"
    assert insights["top_file_rows"][0]["width"] == 88.0
    assert "trend_chart" not in insights


def test_report_radar_requires_exact_executed_version_and_complete_metrics():
    catalog = {
        "version_groups": [{
            "key": "codet5p",
            "name": "CodeT5+ 220M",
            "versions": [{
                "version": "codet5-exact",
                "tasks": [{
                    "task": "malicious_intent",
                    "language_metrics": [{
                        "language": "python",
                        "full_metrics": True,
                        "accuracy": 0.98,
                        "precision": 0.97,
                        "f1": 0.975,
                        "false_negative_rate": 0.04,
                        "false_positive_rate": 0.02,
                    }],
                }],
            }],
        }, {
            "key": "xgboost",
            "name": "XGBoost",
            "versions": [{
                "version": "xgb-exact",
                "tasks": [{
                    "task": "malicious_intent",
                    "scope": "已验证语言",
                    "language_metrics": [{
                        "language": "python",
                        "language_label": "Python",
                        "full_metrics": True,
                        "accuracy": 0.96,
                        "precision": 0.94,
                        "f1": 0.95,
                        "false_negative_rate": 0.08,
                        "false_positive_rate": 0.03,
                        "samples": 240,
                    }],
                }],
            }],
        }],
    }
    report = {
        "language": "python",
        "engines": [{
            "name": "codet5p",
            "status": "completed",
            "probability": 0.84,
            "threshold": 0.8,
            "model_version": "codet5-exact",
        }, {
            "name": "xgboost_malicious",
            "status": "completed",
            "probability": 0.81,
            "threshold": 0.6,
            "model_version": "xgb-exact",
        }],
    }

    radar = build_file_report_insights(report, catalog)["model_radar"]
    assert [series["model_name"] for series in radar["series"]] == [
        "CodeT5+ 220M", "XGBoost",
    ]
    assert [series["version"] for series in radar["series"]] == [
        "codet5-exact", "xgb-exact",
    ]
    assert radar["series"][0]["style_key"] == "codet5p"
    assert radar["series"][1]["style_key"] == "xgboost"
    assert "准确率98.0%" in radar["series"][0]["aria_metrics"]
    assert "特异度97.0%" in radar["series"][1]["aria_metrics"]

    report["engines"][1]["model_version"] = "xgb-other"
    radar = build_file_report_insights(report, catalog)["model_radar"]
    assert [series["model_name"] for series in radar["series"]] == ["CodeT5+ 220M"]
    report["engines"][0]["model_version"] = "codet5-other"
    assert build_file_report_insights(report, catalog)["model_radar"] is None


def test_project_risk_categories_include_uncategorized_ai_malicious_files_and_cap_at_ten():
    category_files = [
        {
            "filename": f"risk-{index}.py",
            "risk_score": 70,
            "risk_level": "high",
            "final_decision": "malicious",
            "categories": [f"Category {index:02d}"],
        }
        for index in range(12)
    ]
    category_files.append({
        "filename": "ai-only.js",
        "risk_score": 91,
        "risk_level": "critical",
        "final_decision": "malicious",
        "decision_authority": "ai",
        "categories": [],
    })

    insights = build_project_report_insights({
        "final_decision": "malicious",
        "file_results": category_files,
    })

    assert len(insights["category_rows"]) == 10
    assert any(
        item["label"] == "AI Semantic Risk"
        for item in insights["category_rows"]
    )


def test_static_flows_and_behavior_chains_expose_auditable_trace_steps():
    sql = _sql_injection_dataflow(
        "name = request.args.get('name')\n"
        "query = \"SELECT * FROM users WHERE name='\" + name + \"'\"\n"
        "cursor.execute(query)",
        "python",
    )
    path = _java_path_traversal_dataflow(
        "String name = request.getParameter(\"name\");\n"
        "String path = base + name;\n"
        "new FileInputStream(path);"
    )
    chain = detect_behavior_chains(
        "payload = requests.get('https://example.test/a')\nexec(payload.text)"
    )[0]

    assert [step["kind"] for step in sql["trace_steps"]][0] == "source"
    assert [step["kind"] for step in sql["trace_steps"]][-1] == "sink"
    assert [step["kind"] for step in path["trace_steps"]][-1] == "sink"
    assert len(chain["trace_steps"]) == 2


def test_ai_line_attribution_marks_matching_static_path_steps():
    evidence, ai_only = merge_model_line_attributions(
        [{
            "line": 2,
            "category": "SQL Injection",
            "trace_steps": [
                {"kind": "source", "line": 1, "snippet": "user = request.args['id']"},
                {"kind": "sink", "line": 2, "snippet": "cursor.execute(query)"},
            ],
        }],
        [{
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": "malicious",
            "probability": 0.8,
            "threshold": 0.6,
            "metadata": {
                "line_attributions": [
                    {"line": 1, "contribution_percent": 31.0, "probability_drop": 0.08},
                    {"line": 2, "contribution_percent": 44.0, "probability_drop": 0.12},
                ],
            },
        }],
    )

    assert ai_only == []
    assert [step.get("ai_supported") for step in evidence[0]["trace_steps"]] == [True, True]
    assert evidence[0]["trace_steps"][1]["ai_attribution"]["contribution_percent"] == 44.0


def test_markdown_report_contains_decision_ledger_boundary_and_trace():
    remediation = remediation_for_finding({"category": "SQL Injection"}, "python")
    record = {
        "filename": "demo.py",
        "display_language": "Python",
        "language": "python",
        "final_decision": "malicious",
        "decision_authority": "ai",
        "risk_level": "high",
        "risk_score": 80,
        "selected_mode": "standard",
        "effective_mode": "standard",
        "malicious_probability": 0.80,
        "categories": ["SQL Injection"],
        "attack_techniques": [],
        "hashes": {"md5": "m", "sha1": "s1", "sha256": "s256"},
        "file_hash": "s256",
        "created_at": "2026-01-01T00:00:00",
        "engines": [{
            "name": "xgboost_malicious", "status": "completed",
            "decision": "malicious", "probability": 0.8, "threshold": 0.6,
        }],
        "evidence_items": [{
            "line": 3, "category": "SQL Injection", "rule_id": "SQL-003",
            "description": "动态 SQL", "harm": "攻击者可能读写数据库。",
            "snippet": "cursor.execute(query)", "suspicion_score": 91,
            "cwe": remediation["cwe"],
            "cve_examples": remediation["cve_examples"],
            "repair_suggestions": remediation["suggestions"],
            "trace_steps": [{"kind": "sink", "stage": "数据库执行接口", "line": 3, "snippet": "cursor.execute(query)"}],
        }, {
            "line": 8, "category": "SQL Injection", "rule_id": "SQL-003",
            "description": "动态 SQL", "harm": "攻击者可能读写数据库。",
            "snippet": "cursor.execute(other_query)", "suspicion_score": 87,
            "cwe": remediation["cwe"],
            "cve_examples": remediation["cve_examples"],
            "repair_suggestions": remediation["suggestions"],
        }],
    }

    markdown = render_record_markdown(record)

    assert "## AI模型判定明细" in markdown
    assert "风险分说明：" not in markdown
    assert "判定职责" not in markdown
    assert "## 代码关联路径" in markdown
    assert "## 静态证据路径" not in markdown
    assert "恶意代码概率：" not in markdown
    assert "数据库执行接口" in markdown
    assert "### SQL 注入 · CWE-89 · 2 处" in markdown
    assert markdown.count("#### 典型例子") == 1
    assert markdown.count("#### 修复建议") == 1
    assert markdown.count(remediation["suggestions"][0]) == 1
    assert "CVE-2023-34362" in markdown
    assert "CWE / CVE 典型例子" not in markdown
    assert "不代表当前文件就是这些 CVE" not in markdown
    assert "检测依据" not in markdown
    assert "规则解释" not in markdown


def test_report_templates_parse_with_local_jinja_environment():
    templates = Path(__file__).resolve().parents[2] / "frontend" / "templates"
    environment = Environment(loader=FileSystemLoader(templates))
    environment.filters["zh"] = str

    for name in (
        "attack/index.html",
        "attack/record.html",
        "attack/project_file_detail.html",
        "attack/_file_report_insights.html",
        "attack/_project_report_insights.html",
        "attack/_report_chart_macros.html",
        "attack/_evidence_groups.html",
        "attack/_evidence_trace.html",
        "attack/_weakness_examples.html",
    ):
        environment.get_template(name)

    file_html = environment.get_template("attack/_file_report_insights.html").render(
        file_report_insights=build_file_report_insights({
            "engines": [{
                "name": "xgboost_malicious", "status": "completed",
                "decision": "benign", "probability": 0.4, "threshold": 0.6,
            }],
            "evidence_items": [],
        })
    )
    project_html = environment.get_template("attack/_project_report_insights.html").render(
        project_report_insights=build_project_report_insights({
            "project_name": "demo.zip", "level_counts": {"safe": 1},
            "file_results": [{"filename": "safe.py", "risk_score": 0, "risk_level": "safe"}],
        })
    )
    graph_project_html = environment.get_template("attack/_project_report_insights.html").render(
        project_report_insights=build_project_report_insights({
            "project_relationship_graph": {
                "node_count": 2,
                "edge_count": 1,
                "nodes": [
                    {"id": "file-1", "path": "src/main.c", "name": "main.c", "risk_score": 70, "risk_level": "high", "degree": 1},
                    {"id": "file-2", "path": "src/util.h", "name": "util.h", "risk_score": 4, "risk_level": "safe", "degree": 1},
                ],
                "edges": [{"source": "file-1", "target": "file-2", "relation": "include"}],
            },
            "project_engines": [{"name": "gatv2", "status": "completed", "probability": 0.3, "threshold": 0.8, "model_version": "gat-test"}],
        })
    )

    assert "低于阈值 20.0 个百分点" in file_html
    assert "AI模型判定明细" in file_html
    assert "模型概率与阈值" not in file_html
    assert "AI Decision" not in file_html
    assert "各AI模型的文件级概率、发布阈值和最终作用" not in file_html
    assert "快速主判候选与行级归因" not in file_html
    assert "恶意结论与风险分只来自可用AI模型" not in file_html
    assert "解释来源构成" not in file_html
    assert "AI定位的高贡献行" not in file_html
    assert "风险等级分布" in project_html
    assert "文件语言分布" in project_html
    assert "AI判定覆盖" not in project_html
    assert "Top 10 风险类别" in project_html
    assert "Top 10 风险文件" in project_html
    assert project_html.count("project-top10-card") == 2
    assert project_html.index("Top 10 风险类别") < project_html.index("Top 10 风险文件")
    for redundant_copy in (
        "按已检测文件的最终风险等级计数，每个文件仅计入一个等级",
        "柱状图用于比较有限语言品类；超过七类时合并为“其他语言”",
        "横条长度使用固定 0–100 风险分量纲，便于跨文件直接比较",
        "总体构成",
        "代码构成",
        "多分类排行",
        "柱高按本项目最大语言文件数归一化",
        "它描述模型能力边界，不代表本次样本的实时正确率",
    ):
        assert redundant_copy not in project_html
    assert "风险类别 Top 8" not in project_html
    assert "历史风险趋势" not in file_html
    assert "趋势" not in project_html
    assert "项目文件调用关系" in graph_project_html
    assert "main.c" in graph_project_html
    assert "util.h" in graph_project_html
    assert "GATv2 · 执行完成" in graph_project_html

    for template_name in (
        "attack/index.html",
        "attack/record.html",
        "attack/project_file_detail.html",
    ):
        template_source = (templates / template_name).read_text(encoding="utf-8")
        assert "恶意代码检测" not in template_source
        assert "第 {{ evidence.line or '未知' }} 行" not in template_source
        assert "第 {{ match.line or '未知' }} 行" not in template_source
        assert "AI判断与证据" not in template_source
        assert "|zh }}风险分" not in template_source

    assert ">风险分</span>" in (
        templates / "attack/index.html"
    ).read_text(encoding="utf-8")
    assert ">风险分</span>" in (
        templates / "attack/record.html"
    ).read_text(encoding="utf-8")
    assert ">风险分</span>" in (
        templates / "attack/project_file_detail.html"
    ).read_text(encoding="utf-8")

    index_source = (templates / "attack/index.html").read_text(encoding="utf-8")
    assert "文件检测报告" in index_source
    assert "单文件检测报告" not in index_source
    assert "扫描过程未产生警告" not in index_source
    assert "平均风险分" not in index_source
    assert index_source.count("project-stat-card") == 0
    assert "project-risk-details" in index_source
    assert "本次没有完成项目图模型分析" not in index_source
    assert "本次实际执行模型" not in index_source
    assert "AI参与率" not in index_source
    assert "扫描警告表示检测覆盖受到限制" in index_source
    assert "不代表发现了恶意代码" in index_source


def test_grouped_evidence_template_renders_one_guidance_block_per_weakness():
    templates = Path(__file__).resolve().parents[2] / "frontend" / "templates"
    environment = Environment(loader=FileSystemLoader(templates))
    environment.filters["zh"] = str
    grouped_html = environment.get_template("attack/_evidence_groups.html").render(
        evidence_groups=build_evidence_groups([{
            "line": 4,
            "category": "SQL Injection",
            "cwe": "CWE-89",
            "suspicion_score": 92,
            "snippet": "cursor.execute(query)",
            "harm": "攻击者可能读写数据库。",
            "repair_suggestions": ["使用参数化查询。"],
        }, {
            "line": 7,
            "category": "SQL Injection",
            "cwe": "CWE-89",
            "suspicion_score": 88,
            "snippet": "cursor.execute(other_query)",
            "harm": "攻击者可能读写数据库。",
            "repair_suggestions": ["使用参数化查询。"],
        }]),
        evidence_total=2,
        evidence_empty_message="没有证据",
    )

    assert grouped_html.count('class="evidence-item evidence-group motion-card"') == 1
    assert grouped_html.count("使用参数化查询。") == 1
    assert "第 4 行" in grouped_html and "第 7 行" in grouped_html
    assert "可疑度 92/100" in grouped_html
    assert "检测依据" not in grouped_html
    assert "规则解释" not in grouped_html


def test_detection_report_routes_do_not_load_history_for_insight_cards():
    route_source = (
        Path(__file__).resolve().parents[1]
        / "web"
        / "routes"
        / "attack_routes.py"
    ).read_text(encoding="utf-8")

    assert "build_file_report_insights(\n            result," in route_source
    assert "build_file_report_insights(\n            public_record," in route_source
    assert "model_center_view() if result else None" in route_source
    assert "build_file_report_insights(\n            result,\n            list_record" not in route_source
    assert "build_file_report_insights(\n            public_record,\n            list_record" not in route_source
