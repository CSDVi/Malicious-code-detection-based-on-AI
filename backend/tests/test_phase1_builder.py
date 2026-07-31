from attack_detection.data_pipeline import deduplicate, generate_evasion_suite, make_sample
from attack_detection.phase1_builder import _language


def test_exact_cross_label_conflicts_are_quarantined():
    benign = make_sample("print('same')", label="benign", category="safe", language="python")
    malicious = make_sample("print('same')", label="malicious", category="payload", language="python")
    output, report = deduplicate([benign, malicious])
    assert output == []
    assert report["conflict_samples_quarantined"] == 2
    assert report["label_conflicts"][0]["labels"] == ["benign", "malicious"]


def test_language_mapping_keeps_configuration_separate():
    assert _language("package/index.ts") == "typescript"
    assert _language("package/package.json") == "config"


def test_training_metadata_round_trips_on_sample():
    sample = make_sample(
        "eval(payload)",
        label="malicious",
        category="Obfuscated Payload",
        language="javascript",
        behavior_labels=["Obfuscated Payload"],
        cwe_labels=["CWE-95"],
        label_confidence=0.95,
        review_status="source_verified",
    )
    assert sample.behavior_labels == ("Obfuscated Payload",)
    assert sample.cwe_labels == ("CWE-95",)
    assert sample.label_confidence == 0.95


def test_evasion_variant_follows_parent_split():
    sample = make_sample(
        "eval('abcdefghijk')",
        label="malicious",
        category="Obfuscated Payload",
        language="javascript",
        family="npm:example",
        split="train",
        label_confidence=0.95,
        review_status="source_verified",
    )
    variants = generate_evasion_suite([sample], limit=4)
    assert variants
    assert {item.split for item in variants} == {"train"}
    assert {item.parent_sample_hash for item in variants} == {sample.sample_hash}
    assert {item.review_status for item in variants} == {"generated_variant"}
