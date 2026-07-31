from attack_detection.practiceset_layout import resolve_practiceset_layout


def test_legacy_practiceset_layout_keeps_existing_paths(tmp_path):
    layout = resolve_practiceset_layout(tmp_path)

    assert layout.organized is False
    assert layout.vulnerability == tmp_path
    assert layout.javascript == tmp_path
    assert layout.other == tmp_path


def test_organized_practiceset_layout_resolves_categories_and_languages(tmp_path):
    (tmp_path / "vulnerability_detection").mkdir()
    (tmp_path / "malware_detection").mkdir()

    layout = resolve_practiceset_layout(tmp_path)

    assert layout.organized is True
    assert layout.vulnerability == tmp_path / "vulnerability_detection"
    assert layout.java == tmp_path / "malware_detection" / "java"
    assert layout.javascript == tmp_path / "malware_detection" / "javascript"
    assert layout.php == tmp_path / "malware_detection" / "php"
    assert layout.python == tmp_path / "malware_detection" / "python"
    assert layout.other == tmp_path / "malware_detection" / "other"
