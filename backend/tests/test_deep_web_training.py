import json
from types import SimpleNamespace

import pytest

import attack_detection.engines.codet5p_engine as codet5p_engine_module
from attack_detection.engines.codet5p_engine import CodeT5PEngine
from attack_detection.model_center import model_center_view
from attack_detection.training import deep_web_trainer
from web.routes.attack_routes import _training_model_options, _validate_training_dataset


def test_training_options_include_only_visible_product_families_and_codet5p_base():
    options = _training_model_options(model_center_view())
    families = list(dict.fromkeys(item["family"] for item in options))
    assert families == ["xgboost", "codet5p", "gatv2"]
    codet5p = [
        item for item in options
        if item["family"] == "codet5p"
    ]
    assert "codet5p-220m-base" in {item["version"] for item in codet5p}


def test_gatv2_accepts_language_annotated_graph_jsonl_but_xgboost_requires_code_schema(tmp_path):
    dataset = tmp_path / "graphs.jsonl"
    dataset.write_text(json.dumps({
        "nodes": [
            {"id": "package:demo", "type": "package"},
            {"id": "file:main", "type": "file", "language": "go"},
        ],
        "edges": [],
        "label": "benign",
        "split": "train",
    }) + "\n", encoding="utf-8")

    _validate_training_dataset(dataset, "gatv2")
    with pytest.raises(ValueError, match="code"):
        _validate_training_dataset(dataset, "xgboost")


def test_all_code_model_datasets_require_language_field(tmp_path):
    dataset = tmp_path / "code.jsonl"
    record = {
        "code": "package main",
        "label": "benign",
        "split": "train",
        "review_status": "approved",
        "label_confidence": 1.0,
    }
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    for family in ("xgboost", "codet5p", "gatv2"):
        with pytest.raises(ValueError, match="language"):
            _validate_training_dataset(dataset, family)
    record["language"] = "go"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    for family in ("xgboost", "codet5p", "gatv2"):
        _validate_training_dataset(dataset, family)


def test_gatv2_graph_requires_language_on_a_file_node(tmp_path):
    dataset = tmp_path / "graphs.jsonl"
    dataset.write_text(json.dumps({
        "nodes": [{"id": "package:demo", "type": "package"}],
        "edges": [],
        "label": "benign",
        "split": "train",
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="language"):
        _validate_training_dataset(dataset, "gatv2")


def test_codet5p_project_batch_shapes_completed_and_unavailable_results(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "codet5p_registry.json").write_text(
        json.dumps({
            "active_routes": {
                "malicious_intent": {"javascript": "codet5p-test"},
            },
        }),
        encoding="utf-8",
    )
    deep_python = tmp_path / "python.exe"
    deep_python.write_bytes(b"test")
    infer_outputs = {
        "results": [
            {
                "status": "completed",
                "decision": "malicious",
                "probability": 0.93,
                "threshold": 0.8,
                "model_version": "codet5p-test",
                "primary_task": "malicious_intent",
                "task_probabilities": {"malicious_intent": 0.93},
                "task_thresholds": {"malicious_intent": 0.8},
                "task_versions": {"malicious_intent": "codet5p-test"},
                "duration_ms": 12,
            },
            {
                "status": "unavailable",
                "reason": "no active CodeT5+ route for language python",
            },
        ],
    }

    monkeypatch.setattr(codet5p_engine_module, "MODEL_DIR", model_dir)
    monkeypatch.setattr(codet5p_engine_module, "DEFAULT_DEEP_PYTHON", deep_python)
    monkeypatch.setattr(
        codet5p_engine_module,
        "_run_persistent_batch",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(infer_outputs),
            stderr="",
        ),
    )

    results = CodeT5PEngine().scan_batch([
        {"content": "eval(payload)", "language": "javascript"},
        {"content": "print('ok')", "language": "python"},
    ])

    assert results[0]["status"] == "completed"
    assert results[0]["name"] == "codet5p"
    assert results[0]["metadata"]["primary_task"] == "malicious_intent"
    assert results[0]["metadata"]["task_probabilities"] == {"malicious_intent": 0.93}
    assert results[1]["status"] == "unavailable"
    assert "python" in results[1]["reason"]


def test_gatv2_release_gate_checks_precision_and_both_error_rates():
    assert deep_web_trainer._metrics_pass({
        "precision": 0.9,
        "false_positive_rate": 0.1,
        "false_negative_rate": 0.1,
    })
    assert not deep_web_trainer._metrics_pass({
        "precision": 0.89,
        "false_positive_rate": 0.01,
        "false_negative_rate": 0.01,
    })
    assert not deep_web_trainer._metrics_pass({
        "precision": 0.99,
        "false_positive_rate": 0.11,
        "false_negative_rate": 0.01,
    })
    assert not deep_web_trainer._metrics_pass({
        "precision": 0.99,
        "false_positive_rate": 0.01,
        "false_negative_rate": 0.11,
    })


def test_gatv2_publish_uses_versioned_weight_and_atomic_manifest(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    output = tmp_path / "output"
    model_dir.mkdir()
    output.mkdir()
    (output / "gatv2_classifier.pt").write_bytes(b"weights")
    manifest = {
        "model_version": "gatv2-test-version",
        "files": ["gatv2_classifier.pt"],
    }
    monkeypatch.setattr(deep_web_trainer, "MODEL_DIR", model_dir)

    deep_web_trainer._publish_gatv2(output, manifest)

    runtime = json.loads((model_dir / "gatv2_manifest.json").read_text(encoding="utf-8"))
    assert runtime["artifact"] == "gatv2_classifier__gatv2-test-version.pt"
    assert (model_dir / runtime["artifact"]).read_bytes() == b"weights"


def test_bytetcn_publish_routes_tasks_to_versioned_weight(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    output = tmp_path / "output"
    model_dir.mkdir()
    output.mkdir()
    (output / "bytetcn_multitask.pt").write_bytes(b"weights")
    manifest = {
        "model_version": "bytetcn-test-version",
        "files": ["bytetcn_multitask.pt"],
        "config": {"max_length": 128},
        "task_language_support": {
            "malicious_intent": ["python"],
            "vulnerability_risk": ["java"],
        },
    }
    monkeypatch.setattr(deep_web_trainer, "MODEL_DIR", model_dir)

    deep_web_trainer._publish_bytetcn(output, manifest)

    runtime = json.loads((model_dir / "bytetcn_manifest.json").read_text(encoding="utf-8"))
    assert runtime["files"] == ["bytetcn_multitask__bytetcn-test-version.pt"]
    assert runtime["task_models"]["malicious_intent"]["file"] == runtime["files"][0]
    assert runtime["task_models"]["vulnerability_risk"]["file"] == runtime["files"][0]
    assert (model_dir / runtime["files"][0]).read_bytes() == b"weights"
