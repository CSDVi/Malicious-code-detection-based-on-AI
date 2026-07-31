from pathlib import PurePosixPath

from pathlib import Path

from attack_detection.safe_extract import _archive_output_name, _safe_member, _wanted_member


def test_safe_extract_rejects_unsafe_and_non_source_members():
    assert _safe_member("../escape.js") is None
    assert _safe_member("C:/escape.js") is None
    assert _safe_member("/absolute/escape.js") is None
    assert _wanted_member(PurePosixPath("package/index.js"))
    assert _wanted_member(PurePosixPath("package/package.json"))
    assert not _wanted_member(PurePosixPath("package/payload.exe"))
    assert not _wanted_member(PurePosixPath("package/nested.zip"))
    assert not _wanted_member(PurePosixPath("package/node_modules/dependency.js"))


def test_nested_archives_receive_stable_unique_output_names():
    root = Path("archives")
    first = _archive_output_name(root / "aa" / "sample.zip", root)
    second = _archive_output_name(root / "bb" / "sample.zip", root)
    assert first.startswith("sample__")
    assert first == _archive_output_name(root / "aa" / "sample.zip", root)
    assert first != second
